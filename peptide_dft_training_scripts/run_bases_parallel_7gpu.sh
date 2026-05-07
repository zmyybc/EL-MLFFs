#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
setup_env

PIDS=()
for idx in "${!BASE_MODELS[@]}"; do
  script_idx=$((idx + 1))
  gpu_id="${GPU_IDS:-0,1,2,3,4,5,6}"
  gpu_id="$(echo "${gpu_id}" | tr ',' '\n' | sed -n "$((idx + 1))p")"
  if [[ -z "${gpu_id}" ]]; then
    echo "Missing GPU id for model index ${idx}" >&2
    exit 1
  fi
  SKIP_ENV_SETUP=1 GPU_ID="${gpu_id}" bash "${SCRIPT_DIR}/${script_idx}.sh" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"

