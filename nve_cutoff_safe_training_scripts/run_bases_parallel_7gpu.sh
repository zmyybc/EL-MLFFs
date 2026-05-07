#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids=()
for spec in \
  "0 1_dp" \
  "1 2_nep" \
  "2 3_mtp" \
  "3 4_soap" \
  "4 5_painn" \
  "5 6_schnet" \
  "6 7_mace"; do
  read -r gpu task <<< "${spec}"
  echo "Launching ${task}.sh on GPU ${gpu}"
  (GPU_ID="${gpu}" bash "${SCRIPT_DIR}/${task}.sh") &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one base-model task failed." >&2
  exit "${status}"
fi

echo "All base-model tasks completed. You can now run 8_meta_conservative.sh."
