#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

MODEL_NAME="schnet"
TASK_INDEX="6"
GPU_ID="${GPU_ID:-0}"
TARGET_TOTAL_STEPS="${BASE_TARGET_TOTAL_STEPS:-50000}"
LR="${BASE_LR:-5e-4}"
MIN_LR="${BASE_MIN_LR:-1e-6}"
ENERGY_WEIGHT="${BASE_ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${BASE_FORCE_WEIGHT:-50.0}"
CUTOFF="${CUTOFF:-5.0}"
INITIAL_BATCH_SIZE="${BASE_INITIAL_BATCH_SIZE:-128}"
MAX_BATCH_SIZE="${BASE_MAX_BATCH_SIZE:-4096}"
TARGET_MEMORY_FRACTION="${BASE_TARGET_MEMORY_FRACTION:-0.70}"
MEMORY_HEADROOM_FRACTION="${BASE_MEMORY_HEADROOM_FRACTION:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"

mkdir -p "${BASE_OUTPUT_DIR}" "${BASE_LOG_DIR}"
DONE_FILE="${BASE_OUTPUT_DIR}/.schnet.done"
CKPT_PATH="${BASE_OUTPUT_DIR}/schnet_torch.pth"
LOG_FILE="${BASE_LOG_DIR}/6_schnet.log"

if [[ -f "${DONE_FILE}" && -f "${CKPT_PATH}" ]]; then
  echo "Skip schnet: completed marker exists at ${DONE_FILE}"
  exit 0
fi

setup_nve_env
require_paths

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cat <<EOF
Launching NVE cutoff-safe base model
  model=${MODEL_NAME}
  gpu=${GPU_ID}
  train=${TRAIN_FILE}
  val=${VAL_FILE}
  output_dir=${BASE_OUTPUT_DIR}
  checkpoint=${CKPT_PATH}
  log=${LOG_FILE}
  target_total_steps=${TARGET_TOTAL_STEPS}
  lr=${LR} min_lr=${MIN_LR}
  force_weight=${FORCE_WEIGHT} cutoff=${CUTOFF}
EOF

python "${PROJECT_ROOT}/train_one_base_model.py"   --model-name "${MODEL_NAME}"   --data-file "${TRAIN_FILE}"   --val-data-file "${VAL_FILE}"   --cutoff "${CUTOFF}"   --lr "${LR}"   --min-lr "${MIN_LR}"   --energy-weight "${ENERGY_WEIGHT}"   --force-weight "${FORCE_WEIGHT}"   --train-ratio "${TRAIN_RATIO}"   --seed "${SEED}"   --num-workers "${NUM_WORKERS}"   --initial-batch-size "${INITIAL_BATCH_SIZE}"   --max-batch-size "${MAX_BATCH_SIZE}"   --target-memory-fraction "${TARGET_MEMORY_FRACTION}"   --memory-headroom-fraction "${MEMORY_HEADROOM_FRACTION}"   --target-total-steps "${TARGET_TOTAL_STEPS}"   --output-dir "${BASE_OUTPUT_DIR}"   2>&1 | tee "${LOG_FILE}"

status="${PIPESTATUS[0]}"
if [[ "${status}" -eq 0 && -f "${CKPT_PATH}" ]]; then
  touch "${DONE_FILE}"
fi
exit "${status}"
