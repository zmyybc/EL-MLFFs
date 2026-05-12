#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${PROJECT_ROOT}/el-mlffs"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
MASTER_PORT="${MASTER_PORT:-29632}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-${WORK_DIR}/checkpoints/base_models_a100_8gpu_50ksteps}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/el-mlffs/data/train.extxyz}"
VAL_FILE="${VAL_FILE:-${PROJECT_ROOT}/el-mlffs/data/test.extxyz}"
SAVE_PATH="${SAVE_PATH:-checkpoints/meta_models/conservative_meta_current_bases_8gpu_connected.pth}"
LOG_FILE="${LOG_FILE:-${WORK_DIR}/logs/conservative_meta_current_bases_8gpu_connected.log}"

AUTO_BATCH_START="${AUTO_BATCH_START:-1}"
AUTO_BATCH_STEP="${AUTO_BATCH_STEP:-1}"
AUTO_BATCH_MAX="${AUTO_BATCH_MAX:-8}"
MEMORY_TARGET_RATIO="${MEMORY_TARGET_RATIO:-0.70}"
TARGET_TOTAL_STEPS="${TARGET_TOTAL_STEPS:-20000}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-6}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCE_WEIGHT="${FORCE_WEIGHT:-50.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
HUBER_DELTA="${HUBER_DELTA:-1.0}"
SEED="${SEED:-42}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random}"
ENV_NAME="${ENV_NAME:-horm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.2.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYG_WHEEL_URL="${PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}"

BASE_MODELS=(dp nep mtp soap painn schnet mace)

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba not found in PATH." >&2
  exit 1
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}"
set +u
eval "$(mamba shell hook --shell bash)"
set -u

if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Detected existing mamba env: ${ENV_NAME}."
else
  echo "Creating mamba env: ${ENV_NAME}"
  mamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

set +u
mamba activate "${ENV_NAME}"
set -u

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
python -m pip install numpy==1.26.4 ase e3nn matplotlib seaborn tqdm
python -m pip install torch-cluster -f "${PYG_WHEEL_URL}"
python -m pip install torch-scatter -f "${PYG_WHEEL_URL}"
python -m pip install torch-sparse -f "${PYG_WHEEL_URL}"
python -m pip install torch-geometric

for model_name in "${BASE_MODELS[@]}"; do
  checkpoint="${BASE_MODEL_DIR}/${model_name}_torch.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing base checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Missing train file: ${TRAIN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Missing val file: ${VAL_FILE}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}/$(dirname "${SAVE_PATH}")" "$(dirname "${LOG_FILE}")"

echo "========================================"
echo "Old-data 7-model Connected Conservative Meta"
echo "========================================"
echo "models:        ${BASE_MODELS[*]}"
echo "gpu_ids:       ${GPU_IDS}"
echo "nproc:         ${NPROC_PER_NODE}"
echo "train_file:    ${TRAIN_FILE}"
echo "val_file:      ${VAL_FILE}"
echo "base_model_dir:${BASE_MODEL_DIR}"
echo "save_path:     ${WORK_DIR}/${SAVE_PATH}"
echo "log_file:      ${LOG_FILE}"
echo "differentiate_force_features=1"
echo "lr=${LR} target_total_steps=${TARGET_TOTAL_STEPS} grad_clip_norm=${GRAD_CLIP_NORM}"
echo "========================================"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
cd "${WORK_DIR}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master-port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/train_conservative_meta_with_bases.py" \
    --base-model-dir "${BASE_MODEL_DIR}" \
    --base-models "${BASE_MODELS[@]}" \
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
    --differentiate-force-features \
    --save-path "${SAVE_PATH}" \
    --seed "${SEED}" \
    --split-strategy "${SPLIT_STRATEGY}" \
    --num-workers "${NUM_WORKERS}" \
    2>&1 | tee "${LOG_FILE}"
