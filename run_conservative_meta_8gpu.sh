#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}/el-mlffs"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${WORK_DIR}/checkpoints/base_models_a100_8gpu_50ksteps}"
BASE_MODELS="${BASE_MODELS:-dp nep mtp soap painn schnet mace}"
TRAIN_FILE="${TRAIN_FILE:-data/train.extxyz}"
VAL_FILE="${VAL_FILE:-data/test.extxyz}"
SAVE_PATH="${SAVE_PATH:-checkpoints/meta_models/conservative_meta_current_bases_8gpu.pth}"

BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-5e-4}"
MIN_LR="${MIN_LR:-1e-6}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-50.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
SEED="${SEED:-42}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random}"

mkdir -p "${WORK_DIR}/$(dirname "${SAVE_PATH}")"

cd "${WORK_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/train_conservative_meta_with_bases.py" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --base-models ${BASE_MODELS} \
  --data-file "${TRAIN_FILE}" \
  --val-data-file "${VAL_FILE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
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
