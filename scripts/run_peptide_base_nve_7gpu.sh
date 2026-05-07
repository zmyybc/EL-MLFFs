#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/tiger/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME:-horm}"

MODELS=(dp nep mtp soap painn schnet mace)
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/reports/peptide_base_nve_smoke}"
STEPS="${STEPS:-4000}"
TIMESTEP_FS="${TIMESTEP_FS:-0.25}"
TEMPERATURE_K="${TEMPERATURE_K:-150}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
TRAJ_INTERVAL="${TRAJ_INTERVAL:-200}"
RELAX_STEPS="${RELAX_STEPS:-300}"
RELAX_FMAX="${RELAX_FMAX:-0.2}"
STOP_DRIFT_MEV_ATOM="${STOP_DRIFT_MEV_ATOM:-50}"
STOP_MIN_DISTANCE="${STOP_MIN_DISTANCE:-0.45}"

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
PIDS=()
for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_ARRAY[$idx]}"
  log_file="${OUTPUT_DIR}/${model}.log"
  echo "launch base NVE model=${model} gpu=${gpu} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python "${PROJECT_ROOT}/scripts/run_peptide_base_nve.py" \
    --model "${model}" \
    --output-dir "${OUTPUT_DIR}" \
    --steps "${STEPS}" \
    --timestep-fs "${TIMESTEP_FS}" \
    --temperature-k "${TEMPERATURE_K}" \
    --log-interval "${LOG_INTERVAL}" \
    --traj-interval "${TRAJ_INTERVAL}" \
    --relax-steps "${RELAX_STEPS}" \
    --relax-fmax "${RELAX_FMAX}" \
    --stop-drift-mev-atom "${STOP_DRIFT_MEV_ATOM}" \
    --stop-min-distance "${STOP_MIN_DISTANCE}" \
    > "${log_file}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

OUTPUT_DIR_FOR_SUMMARY="${OUTPUT_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR_FOR_SUMMARY"])
rows = []
for path in sorted(root.glob("*/summary.json")):
    rows.append(json.loads(path.read_text()))
for row in rows:
    print(
        f"{row['model']:6s} steps={row['completed_steps']:5d} "
        f"time_fs={row['completed_time_fs']:8.1f} "
        f"drift={row['final_abs_drift_mev_atom']:9.3f} meV/atom "
        f"maxdrift={row['max_abs_drift_mev_atom']:9.3f} "
        f"T={row['final_temperature_K']:8.2f} K "
        f"mind={row['min_distance_min_A']:6.3f} A "
        f"stop={row['stopped_reason'] or 'completed'}"
    )
PY

exit "${status}"
