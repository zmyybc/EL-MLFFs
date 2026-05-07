#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

GPU_IDS="${GPU_IDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29617}"
BASE_MODELS=(dp nep mtp soap painn schnet mace)
BASE_CHECKPOINT_TEMPLATE="${BASE_CHECKPOINT_TEMPLATE:-{model}_torch.pth}"
SAVE_PATH="${SAVE_PATH:-${META_OUTPUT_DIR}/conservative_meta_7model_nve_cutoff_safe.pth}"
LOG_FILE="${LOG_FILE:-${META_LOG_DIR}/8_conservative_meta_7model.log}"
DONE_FILE="${DONE_FILE:-${META_OUTPUT_DIR}/.conservative_meta_7model.done}"

TARGET_TOTAL_STEPS="${META_TARGET_TOTAL_STEPS:-20000}"
LR="${META_LR:-5e-4}"
MIN_LR="${META_MIN_LR:-1e-6}"
ENERGY_WEIGHT="${META_ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${META_FORCE_WEIGHT:-50.0}"
CUTOFF="${CUTOFF:-5.0}"
BATCH_SIZE="${META_BATCH_SIZE:-4}"
AUTO_BATCH_START="${META_AUTO_BATCH_START:-4}"
AUTO_BATCH_STEP="${META_AUTO_BATCH_STEP:-1}"
AUTO_BATCH_MAX="${META_AUTO_BATCH_MAX:-32}"
MEMORY_TARGET_RATIO="${META_MEMORY_TARGET_RATIO:-0.50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_CLIP_NORM="${META_GRAD_CLIP_NORM:-10.0}"
HUBER_DELTA="${META_HUBER_DELTA:-1.0}"
SEED="${SEED:-42}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random}"

mkdir -p "${META_OUTPUT_DIR}" "${META_LOG_DIR}"

if [[ -f "${DONE_FILE}" && -f "${SAVE_PATH}" ]]; then
  echo "Skip conservative meta: completed marker exists at ${DONE_FILE}"
  exit 0
fi

setup_nve_env
require_paths

for model in "${BASE_MODELS[@]}"; do
  ckpt="${BASE_OUTPUT_DIR}/${model}_torch.pth"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing NVE cutoff-safe base checkpoint: ${ckpt}" >&2
    echo "Run scripts 1-7 first, or set BASE_OUTPUT_DIR to the completed checkpoint directory." >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cat <<EOF
Launching NVE cutoff-safe conservative meta model
  base_model_dir=${BASE_OUTPUT_DIR}
  base_models=${BASE_MODELS[*]}
  gpus=${GPU_IDS}
  nproc_per_node=${NPROC_PER_NODE}
  train=${TRAIN_FILE}
  val=${VAL_FILE}
  save=${SAVE_PATH}
  log=${LOG_FILE}
  target_total_steps=${TARGET_TOTAL_STEPS}
  lr=${LR} min_lr=${MIN_LR}
  force_weight=${FORCE_WEIGHT} cutoff=${CUTOFF}
  auto_batch_start=${AUTO_BATCH_START} max=${AUTO_BATCH_MAX} memory_target=${MEMORY_TARGET_RATIO}
EOF

(
  cd "${WORK_DIR}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" --master-port="${MASTER_PORT}" "${PROJECT_ROOT}/train_conservative_meta_with_bases.py" \
    --base-model-dir "${BASE_OUTPUT_DIR}" \
    --base-checkpoint-template "${BASE_CHECKPOINT_TEMPLATE}" \
    --base-models "${BASE_MODELS[@]}" \
    --data-file "${TRAIN_FILE}" \
    --val-data-file "${VAL_FILE}" \
    --cutoff "${CUTOFF}" \
    --batch-size "${BATCH_SIZE}" \
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
    --save-path "${SAVE_PATH}" \
    --seed "${SEED}" \
    --split-strategy "${SPLIT_STRATEGY}" \
    --num-workers "${NUM_WORKERS}"
) 2>&1 | tee "${LOG_FILE}"

status="${PIPESTATUS[0]}"
if [[ "${status}" -eq 0 && -f "${SAVE_PATH}" ]]; then
  touch "${DONE_FILE}"
fi
exit "${status}"
