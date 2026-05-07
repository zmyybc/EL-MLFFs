#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
setup_env
activate_env_only

BASE_MODELS_5=(nep mtp soap schnet mace)
SAVE_PATH="${META_OUTPUT_DIR}/conservative_meta_5model_no_dp_painn.pth"
LOG_FILE="${LOG_DIR}/10_meta_conservative_5model_no_dp_painn.log"

mkdir -p "${META_OUTPUT_DIR}" "${LOG_DIR}"

AUTO_BATCH_ARGS=()
if [[ "${META_AUTO_BATCH}" == "1" ]]; then
  AUTO_BATCH_ARGS+=(--auto-batch-size)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

echo "Launch peptide DFT conservative meta without dp/painn"
echo "models=${BASE_MODELS_5[*]}"
echo "nproc=${NPROC_PER_NODE} gpus=${CUDA_VISIBLE_DEVICES} data=${DATA_FILE}"
echo "save_path=${SAVE_PATH}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${PROJECT_ROOT}/train_conservative_meta_with_bases.py" \
    --base-models "${BASE_MODELS_5[@]}" \
    --base-model-dir "${BASE_OUTPUT_DIR}" \
    --base-checkpoint-template "{model}_torch.pth" \
    --dataset-backend "${DATASET_BACKEND}" \
    --data-file "${DATA_FILE}" \
    --val-data-file "" \
    --cutoff "${CUTOFF}" \
    --batch-size "${META_BATCH_SIZE}" \
    --lr "${META_LR}" \
    --min-lr "${MIN_LR}" \
    --energy-weight "${ENERGY_WEIGHT}" \
    --force-weight "${FORCE_WEIGHT}" \
    --train-ratio "${TRAIN_RATIO}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size-probe-start "${META_BATCH_PROBE_START}" \
    --batch-size-probe-max "${META_BATCH_PROBE_MAX}" \
    --memory-target-ratio "${META_MEMORY_TARGET_RATIO}" \
    --target-total-steps "${META_TARGET_TOTAL_STEPS}" \
    --save-path "${SAVE_PATH}" \
    "${AUTO_BATCH_ARGS[@]}" \
    "${DATASET_KWARG_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
