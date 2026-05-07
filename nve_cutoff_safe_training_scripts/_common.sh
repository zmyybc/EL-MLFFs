#!/usr/bin/env bash
# Shared environment setup for NVE cutoff-safe retraining scripts.
set -euo pipefail

export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export no_proxy="${no_proxy:-byted.org}"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/bangchen/EL-MLFFs}"
WORK_DIR="${PROJECT_ROOT}/el-mlffs"
ENV_NAME="${ENV_NAME:-horm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.2.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
ENV_SETUP_LOCK="${ENV_SETUP_LOCK:-${PROJECT_ROOT}/.nve_cutoff_safe_env_setup.lock}"
ENV_READY_MARKER="${ENV_READY_MARKER:-${PROJECT_ROOT}/.nve_cutoff_safe_env_ready}"

TRAIN_FILE="${TRAIN_FILE:-${WORK_DIR}/data/train.extxyz}"
VAL_FILE="${VAL_FILE:-${WORK_DIR}/data/test.extxyz}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${WORK_DIR}/checkpoints/base_models_nve_cutoff_safe}"
META_OUTPUT_DIR="${META_OUTPUT_DIR:-${WORK_DIR}/checkpoints/meta_models/nve_cutoff_safe}"
BASE_LOG_DIR="${BASE_LOG_DIR:-${WORK_DIR}/logs/base_models_nve_cutoff_safe}"
META_LOG_DIR="${META_LOG_DIR:-${WORK_DIR}/logs/meta_models_nve_cutoff_safe}"

setup_nve_env() {
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

  (
    flock -x 9
    if [[ ! -f "${ENV_READY_MARKER}" ]]; then
      python -m pip install --upgrade pip setuptools wheel
      if ! python - <<'PY_CHECK_TORCH'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('torch') is not None else 1)
PY_CHECK_TORCH
      then
        python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
      fi
      python -m pip install -r "${PROJECT_ROOT}/requirements-conservative-meta.txt"
      touch "${ENV_READY_MARKER}"
    else
      echo "Environment marker exists: ${ENV_READY_MARKER}"
    fi
  ) 9>"${ENV_SETUP_LOCK}"
}

require_paths() {
  [[ -f "${TRAIN_FILE}" ]] || { echo "Missing TRAIN_FILE=${TRAIN_FILE}" >&2; exit 1; }
  [[ -f "${VAL_FILE}" ]] || { echo "Missing VAL_FILE=${VAL_FILE}" >&2; exit 1; }
  [[ -f "${PROJECT_ROOT}/train_one_base_model.py" ]] || { echo "Missing train_one_base_model.py" >&2; exit 1; }
  [[ -f "${PROJECT_ROOT}/train_conservative_meta_with_bases.py" ]] || { echo "Missing train_conservative_meta_with_bases.py" >&2; exit 1; }
}
