#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_FILE="${TRAIN_FILE:-${SCRIPT_DIR}/el-mlffs/data/train.extxyz}"
VAL_FILE="${VAL_FILE:-${SCRIPT_DIR}/el-mlffs/data/test.extxyz}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/el-mlffs/checkpoints/base_models_a100_8gpu_50ksteps}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/el-mlffs/logs/base_models_a100_8gpu_50ksteps}"

TARGET_TOTAL_STEPS="${TARGET_TOTAL_STEPS:-50000}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
CUTOFF="${CUTOFF:-5.0}"
INITIAL_BATCH_SIZE="${INITIAL_BATCH_SIZE:-128}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4096}"
TARGET_MEMORY_FRACTION="${TARGET_MEMORY_FRACTION:-0.85}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OMP_NUM_THREADS_PER_JOB="${OMP_NUM_THREADS_PER_JOB:-4}"

read -r -a MODELS <<< "${MODELS:-schnet painn dp nep mtp soap mace}"
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"

if [[ "${#GPU_IDS[@]}" -lt "${#MODELS[@]}" ]]; then
  echo "Need at least ${#MODELS[@]} GPU ids, got ${#GPU_IDS[@]}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "Launching 8-GPU base-model jobs"
echo "Models: ${MODELS[*]}"
echo "GPU IDs: ${GPU_IDS[*]}"
echo "Target total steps per model: ${TARGET_TOTAL_STEPS}"
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${OUTPUT_DIR}"

declare -a PIDS=()
declare -a NAMES=()

cleanup() {
  local code=$?
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Stopping child jobs..."
    for pid in "${PIDS[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait || true
  fi
  exit "${code}"
}

trap cleanup INT TERM

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  log_file="${LOG_DIR}/${model}.log"

  echo "[launch] model=${model} gpu=${gpu} log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS_PER_JOB}"
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_one_base_model.py" \
      --model-name "${model}" \
      --data-file "${TRAIN_FILE}" \
      --val-data-file "${VAL_FILE}" \
      --cutoff "${CUTOFF}" \
      --target-total-steps "${TARGET_TOTAL_STEPS}" \
      --lr "${LR}" \
      --min-lr "${MIN_LR}" \
      --initial-batch-size "${INITIAL_BATCH_SIZE}" \
      --max-batch-size "${MAX_BATCH_SIZE}" \
      --target-memory-fraction "${TARGET_MEMORY_FRACTION}" \
      --num-workers "${NUM_WORKERS}" \
      --output-dir "${OUTPUT_DIR}" \
      > "${log_file}" 2>&1
  ) &
  PIDS+=("$!")
  NAMES+=("${model}")
done

status=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  model="${NAMES[$idx]}"
  if wait "${pid}"; then
    echo "[done] ${model}"
  else
    echo "[fail] ${model} (see ${LOG_DIR}/${model}.log)" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "All base-model jobs finished successfully."
