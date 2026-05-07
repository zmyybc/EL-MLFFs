#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sequential single-GPU default. Override GPU_ID per command if desired.
for task in 1_dp 2_nep 3_mtp 4_soap 5_painn 6_schnet 7_mace; do
  bash "${SCRIPT_DIR}/${task}.sh"
done

bash "${SCRIPT_DIR}/8_meta_conservative.sh"
