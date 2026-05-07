#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_FILE="${DATA_FILE:-/mnt/bn/bangchen1/HORM3/data/peptide_dft_wb97x_5k.lmdb}"
REFS_JSON="${REFS_JSON:-/mnt/bn/bangchen1/HORM3/data/peptide_dft_wb97x_5k.refs.json}"
DATASET_BACKEND="${DATASET_BACKEND:-peptide_dft_lmdb}"
ENERGY_FIELD="${ENERGY_FIELD:-ae}"
MAX_FORCE_NORM="${MAX_FORCE_NORM:-100.0}"
PEPTIDE_CACHE_DIR="${PEPTIDE_CACHE_DIR:-${PROJECT_ROOT}/el-mlffs/data/peptide_dft_cache}"
CUTOFF="${CUTOFF:-5.0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${PROJECT_ROOT}/el-mlffs/checkpoints/peptide_dft_wb97x_5k_base_models}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-${PROJECT_ROOT}/el-mlffs/checkpoints/peptide_dft_wb97x_5k_meta_models}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/el-mlffs/logs/peptide_dft_wb97x_5k}"

ENV_NAME="${ENV_NAME:-horm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.2.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-50.0}"
TARGET_TOTAL_STEPS="${TARGET_TOTAL_STEPS:-50000}"
INITIAL_BATCH_SIZE="${INITIAL_BATCH_SIZE:-16}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
TARGET_MEMORY_FRACTION="${TARGET_MEMORY_FRACTION:-0.75}"
MEMORY_HEADROOM_FRACTION="${MEMORY_HEADROOM_FRACTION:-0.05}"

META_LR="${META_LR:-5e-4}"
META_TARGET_TOTAL_STEPS="${META_TARGET_TOTAL_STEPS:-20000}"
META_BATCH_SIZE="${META_BATCH_SIZE:-4}"
META_AUTO_BATCH="${META_AUTO_BATCH:-1}"
META_BATCH_PROBE_START="${META_BATCH_PROBE_START:-4}"
META_BATCH_PROBE_MAX="${META_BATCH_PROBE_MAX:-32}"
META_MEMORY_TARGET_RATIO="${META_MEMORY_TARGET_RATIO:-0.70}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

DATASET_KWARG_ARGS=(
  --dataset-kwarg "energy_field=${ENERGY_FIELD}"
  --dataset-kwarg "max_force_norm=${MAX_FORCE_NORM}"
  --dataset-kwarg "cache_dir=${PEPTIDE_CACHE_DIR}"
)

BASE_MODELS=(dp nep mtp soap painn schnet mace)

setup_env() {
  if [[ "${SKIP_ENV_SETUP:-0}" == "1" ]]; then
    return
  fi
  if ! command -v mamba >/dev/null 2>&1; then
    echo "mamba not found in PATH." >&2
    exit 1
  fi
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}"
  set +u
  eval "$(mamba shell hook --shell bash)"
  set -u
  if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Detected existing mamba env: ${ENV_NAME}"
  else
    mamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
  set +u
  mamba activate "${ENV_NAME}"
  set -u
  python -m pip install --upgrade pip setuptools wheel
  python - <<'PY' || python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
import torch
print(torch.__version__)
PY
  python - <<'PY' || python -m pip install numpy==1.26.4 ase e3nn tqdm lmdb torch-geometric
import ase, e3nn, lmdb, torch_geometric
print("runtime deps ok")
PY
}

activate_env_only() {
  if ! command -v mamba >/dev/null 2>&1; then
    echo "mamba not found in PATH." >&2
    exit 1
  fi
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}"
  set +u
  eval "$(mamba shell hook --shell bash)"
  mamba activate "${ENV_NAME}"
  set -u
}

run_base_model() {
  local model_name="$1"
  local gpu_id="$2"
  local log_prefix="$3"

  mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_DIR}"
  local done_file="${BASE_OUTPUT_DIR}/${model_name}.done"
  local log_file="${LOG_DIR}/${log_prefix}_${model_name}.log"
  local ckpt_path="${BASE_OUTPUT_DIR}/${model_name}_torch.pth"
  if [[ -f "${done_file}" && -f "${ckpt_path}" ]]; then
    echo "Skip ${model_name}: ${ckpt_path} already completed."
    return 0
  fi

  echo "Launch peptide DFT base model: model=${model_name} gpu=${gpu_id} data=${DATA_FILE} energy_field=${ENERGY_FIELD}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    python "${PROJECT_ROOT}/train_one_base_model.py" \
      --model-name "${model_name}" \
      --dataset-backend "${DATASET_BACKEND}" \
      --data-file "${DATA_FILE}" \
      --val-data-file "" \
      --cutoff "${CUTOFF}" \
      --lr "${LR}" \
      --min-lr "${MIN_LR}" \
      --energy-weight "${ENERGY_WEIGHT}" \
      --force-weight "${FORCE_WEIGHT}" \
      --train-ratio "${TRAIN_RATIO}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --initial-batch-size "${INITIAL_BATCH_SIZE}" \
      --max-batch-size "${MAX_BATCH_SIZE}" \
      --target-memory-fraction "${TARGET_MEMORY_FRACTION}" \
      --memory-headroom-fraction "${MEMORY_HEADROOM_FRACTION}" \
      --target-total-steps "${TARGET_TOTAL_STEPS}" \
      --world-size 1 \
      --output-dir "${BASE_OUTPUT_DIR}" \
      "${DATASET_KWARG_ARGS[@]}" \
      2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"
  if [[ "${status}" -eq 0 ]]; then
    touch "${done_file}"
  fi
  return "${status}"
}
