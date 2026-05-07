#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
setup_env
activate_env_only

BASE_MODELS_4=(nep mtp soap schnet)
META_WEIGHT_DECAY="${META_WEIGHT_DECAY:-1e-5}"
META_LR="${META_LR:-1e-4}"
META_BATCH_SIZE="${META_BATCH_SIZE:-1}"
META_AUTO_BATCH="${META_AUTO_BATCH:-1}"
META_BATCH_PROBE_START="${META_BATCH_PROBE_START:-1}"
META_BATCH_PROBE_MAX="${META_BATCH_PROBE_MAX:-4}"
META_MEMORY_TARGET_RATIO="${META_MEMORY_TARGET_RATIO:-0.70}"
SAVE_PATH="${META_OUTPUT_DIR}/conservative_meta_4model_no_dp_painn_mace_l2_connected.pth"
LOG_FILE="${LOG_DIR}/12_meta_conservative_4model_no_dp_painn_mace_l2_connected.log"

mkdir -p "${META_OUTPUT_DIR}" "${LOG_DIR}"

AUTO_BATCH_ARGS=()
if [[ "${META_AUTO_BATCH}" == "1" ]]; then
  AUTO_BATCH_ARGS+=(--auto-batch-size)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29621}"

echo "Launch peptide DFT strict conservative meta without dp/painn/mace"
echo "models=${BASE_MODELS_4[*]}"
echo "nproc=${NPROC_PER_NODE} gpus=${CUDA_VISIBLE_DEVICES} master_port=${MASTER_PORT} data=${DATA_FILE}"
echo "save_path=${SAVE_PATH}"
echo "weight_decay=${META_WEIGHT_DECAY}"
echo "differentiate_force_features=1"

python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/train_conservative_meta_with_bases.py" \
    --base-models "${BASE_MODELS_4[@]}" \
    --base-model-dir "${BASE_OUTPUT_DIR}" \
    --base-checkpoint-template "{model}_torch.pth" \
    --dataset-backend "${DATASET_BACKEND}" \
    --data-file "${DATA_FILE}" \
    --val-data-file "" \
    --cutoff "${CUTOFF}" \
    --batch-size "${META_BATCH_SIZE}" \
    --lr "${META_LR}" \
    --min-lr "${MIN_LR}" \
    --weight-decay "${META_WEIGHT_DECAY}" \
    --energy-weight "${ENERGY_WEIGHT}" \
    --force-weight "${FORCE_WEIGHT}" \
    --train-ratio "${TRAIN_RATIO}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
    --batch-size-probe-start "${META_BATCH_PROBE_START}" \
    --batch-size-probe-max "${META_BATCH_PROBE_MAX}" \
    --memory-target-ratio "${META_MEMORY_TARGET_RATIO}" \
    --target-total-steps "${META_TARGET_TOTAL_STEPS}" \
    --differentiate-force-features \
    --save-path "${SAVE_PATH}" \
    "${AUTO_BATCH_ARGS[@]}" \
    "${DATASET_KWARG_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
