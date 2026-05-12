#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ENV_NAME="${ENV_NAME:-horm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.2.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYG_WHEEL_URL="${PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${PROJECT_ROOT}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/oc20}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${DATA_ROOT}/processed_lmdb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/el-mlffs/checkpoints/oc20_base_models}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/el-mlffs/logs/oc20_base_models}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
MASTER_PORT="${MASTER_PORT:-29631}"

TARGET_TOTAL_STEPS="${TARGET_TOTAL_STEPS:-20000}"
CUTOFF="${CUTOFF:-6.0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PREP_WORKERS="${DATA_PREP_WORKERS:-8}"
STORE_EDGES="${STORE_EDGES:-0}"
REF_ENERGY="${REF_ENERGY:-1}"
TARGET_MEMORY_FRACTION="${TARGET_MEMORY_FRACTION:-0.78}"
MEMORY_HEADROOM_FRACTION="${MEMORY_HEADROOM_FRACTION:-0.05}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${PROCESSED_ROOT}"

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba not found in PATH" >&2
  exit 1
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}"
set +u
eval "$(mamba shell hook --shell bash)"
set -u

if mamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Detected existing mamba env: ${ENV_NAME}"
else
  echo "Creating mamba env: ${ENV_NAME}"
  mamba create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

set +u
mamba activate "${ENV_NAME}"
set -u

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
python -m pip install numpy==1.26.4 ase e3nn matplotlib seaborn tqdm lmdb
python -m pip install torch-cluster -f "${PYG_WHEEL_URL}"
python -m pip install torch-scatter -f "${PYG_WHEEL_URL}"
python -m pip install torch-sparse -f "${PYG_WHEEL_URL}"
python -m pip install torch-geometric

cd "${WORK_DIR}"

PREP_FLAGS=()
if [ "${STORE_EDGES}" = "1" ]; then
  PREP_FLAGS+=("--store-edges")
fi
if [ "${REF_ENERGY}" = "1" ]; then
  PREP_FLAGS+=("--ref-energy")
fi

if [ ! -f "${PROCESSED_ROOT}/2M/metadata.json" ] ||    [ ! -f "${PROCESSED_ROOT}/val_id/metadata.json" ] ||    [ ! -f "${PROCESSED_ROOT}/val_ood_ads/metadata.json" ] ||    [ ! -f "${PROCESSED_ROOT}/val_ood_cat/metadata.json" ] ||    [ ! -f "${PROCESSED_ROOT}/val_ood_both/metadata.json" ]; then
  python "${PROJECT_ROOT}/scripts/prepare_oc20_s2ef_for_elmlffs.py" \
    --oc20-root "${DATA_ROOT}" \
    --processed-root "${PROCESSED_ROOT}" \
    --splits 2M val_id val_ood_ads val_ood_cat val_ood_both \
    --cutoff "${CUTOFF}" \
    --num-workers "${DATA_PREP_WORKERS}" \
    "${PREP_FLAGS[@]}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

python "${PROJECT_ROOT}/train_oc20_base_multigpu.py" \
  --model-name "painn" \
  --dataset-backend oc20_lmdb \
  --data-file "${PROCESSED_ROOT}/2M" \
  --val-data-file "${PROCESSED_ROOT}/val_id" \
  --extra-val-file "ood_ads=${PROCESSED_ROOT}/val_ood_ads" \
  --extra-val-file "ood_cat=${PROCESSED_ROOT}/val_ood_cat" \
  --extra-val-file "ood_both=${PROCESSED_ROOT}/val_ood_both" \
  --cutoff "${CUTOFF}" \
  --lr "3e-4" \
  --min-lr "1e-6" \
  --energy-weight "1.0" \
  --force-weight "50.0" \
  --initial-batch-size "4" \
  --max-batch-size "128" \
  --target-memory-fraction "${TARGET_MEMORY_FRACTION}" \
  --memory-headroom-fraction "${MEMORY_HEADROOM_FRACTION}" \
  --target-total-steps "${TARGET_TOTAL_STEPS}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --master-port "${MASTER_PORT}" \
  --num-workers "${NUM_WORKERS}" \
  --output-dir "${OUTPUT_ROOT}/painn" \
  --save-name "painn_compact.pth" \
  --model-kwarg "hidden_channels=96" \
  --model-kwarg "num_layers=2" \
  --model-kwarg "num_basis=24" \

  2>&1 | tee "${LOG_ROOT}/5_painn_compact.log"
