#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${PROJECT_ROOT}/el-mlffs"
BATCH_INDEX="1"

TASK1_GPU_IDS="${TASK1_GPU_IDS:-0,1,2,3}"
TASK2_GPU_IDS="${TASK2_GPU_IDS:-4,5,6,7}"
NPROC_PER_TASK="${NPROC_PER_TASK:-4}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${WORK_DIR}/checkpoints/base_models_a100_8gpu_50ksteps}"
TRAIN_FILE="${TRAIN_FILE:-data/train.extxyz}"
VAL_FILE="${VAL_FILE:-data/test.extxyz}"

AUTO_BATCH_START="${AUTO_BATCH_START:-4}"
AUTO_BATCH_STEP="${AUTO_BATCH_STEP:-1}"
AUTO_BATCH_MAX="${AUTO_BATCH_MAX:-64}"
MEMORY_TARGET_RATIO="${MEMORY_TARGET_RATIO:-0.80}"
TARGET_TOTAL_STEPS="${TARGET_TOTAL_STEPS:-20000}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-50.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
SEED="${SEED:-42}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
PYG_WHEEL_URL="${PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}"
ENV_SETUP_LOCK="${ENV_SETUP_LOCK:-${PROJECT_ROOT}/.horm_env_setup.lock}"


if mamba env list | awk '{print $1}' | grep -qx "horm"; then
  echo "Detected existing mamba env: horm. Skip setup.sh."
else
  echo "Mamba env horm not found. Running setup.sh."
  (
    cd "$HOME"
    # source your environment setup script here
  )
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}"
set +u
eval "$(mamba shell hook --shell bash)"
mamba activate horm
set -u

(
  flock -x 9
  python -m pip install -q -r "${PROJECT_ROOT}/requirements-conservative-meta.txt"
  if python -c "import torch_geometric" >/dev/null 2>&1; then
    echo "Detected torch_geometric in env horm."
  else
    echo "torch_geometric missing. Installing PyG dependencies."
    python -m pip install -q torch-cluster -f "${PYG_WHEEL_URL}"
    python -m pip install -q torch-scatter -f "${PYG_WHEEL_URL}"
    python -m pip install -q torch-sparse -f "${PYG_WHEEL_URL}"
    python -m pip install -q torch-geometric
  fi
) 9>"${ENV_SETUP_LOCK}"

mkdir -p "${WORK_DIR}/checkpoints/meta_models/conservative_combo"
mkdir -p "${WORK_DIR}/logs/conservative_combo_batch"

TASK_SPECS=()
TASK_SPECS+=("1|dp|1|dp")
TASK_SPECS+=("2|nep|1|nep")
TASK_SPECS+=("3|mtp|1|mtp")
TASK_SPECS+=("4|soap|1|soap")
TASK_SPECS+=("5|painn|1|painn")

launch_task() {
  local task_spec="$1"
  local gpu_ids="$2"

  IFS='|' read -r task_index combo_tag num_models base_models_str <<< "$task_spec"
  local save_path="checkpoints/meta_models/conservative_combo/${task_index}_${combo_tag}.pth"
  local log_path="${WORK_DIR}/logs/conservative_combo_batch/${task_index}_${combo_tag}.log"
  local master_port=$((MASTER_PORT_BASE + task_index))
  local -a base_models
  read -r -a base_models <<< "$base_models_str"
  echo "[Batch ${BATCH_INDEX}] Launch task ${task_index} on GPUs ${gpu_ids} | models=${base_models_str}" >&2
  (
    cd "${WORK_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    exec torchrun --standalone --nproc_per_node="${NPROC_PER_TASK}" --master-port="${master_port}" "${PROJECT_ROOT}/train_conservative_meta_with_bases.py" \
      --base-model-dir "${BASE_MODEL_DIR}" \
      --base-models "${base_models[@]}" \
      --data-file "${TRAIN_FILE}" \
      --val-data-file "${VAL_FILE}" \
      --batch-size "${AUTO_BATCH_START}" \
      --auto-batch-size \
      --batch-size-probe-start "${AUTO_BATCH_START}" \
      --batch-size-probe-step "${AUTO_BATCH_STEP}" \
      --batch-size-probe-max "${AUTO_BATCH_MAX}" \
      --memory-target-ratio "${MEMORY_TARGET_RATIO}" \
      --target-total-steps "${TARGET_TOTAL_STEPS}" \
      --lr "${LR}" \
      --min-lr "${MIN_LR}" \
      --energy-weight "${ENERGY_WEIGHT}" \
      --force-weight "${FORCE_WEIGHT}" \
      --grad-clip-norm "${GRAD_CLIP_NORM}" \
      --huber-delta "${HUBER_DELTA}" \
      --save-path "${save_path}" \
      --seed "${SEED}" \
      --split-strategy "${SPLIT_STRATEGY}" \
      --num-workers "${NUM_WORKERS}"
  ) >"${log_path}" 2>&1 &
  LAST_LAUNCHED_PID="$!"
}

wait_for_group() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "${status}"
}

num_tasks="${#TASK_SPECS[@]}"
task_ptr=0
while [ "${task_ptr}" -lt "${num_tasks}" ]; do
  pids=()

  launch_task "${TASK_SPECS[task_ptr]}" "${TASK1_GPU_IDS}"
  pids+=("${LAST_LAUNCHED_PID}")
  task_ptr=$((task_ptr + 1))

  if [ "${task_ptr}" -lt "${num_tasks}" ]; then
    launch_task "${TASK_SPECS[task_ptr]}" "${TASK2_GPU_IDS}"
    pids+=("${LAST_LAUNCHED_PID}")
    task_ptr=$((task_ptr + 1))
  fi

  if ! wait_for_group "${pids[@]}"; then
    echo "[Batch ${BATCH_INDEX}] A task failed. Stopping this batch script."
    exit 1
  fi
done

echo "[Batch ${BATCH_INDEX}] All tasks completed."
