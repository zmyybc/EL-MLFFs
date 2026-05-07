#!/usr/bin/env bash
# Master script to run all Direct Ensemble experiments.
# Usage: bash run_all_direct_experiments.sh [--parallel-lsm] [--skip-baselines] [--skip-lsm]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_ROOT}/el-mlffs"
REPORTS_DIR="${PROJECT_ROOT}/reports"

# GPU configuration — auto-detect available GPUs, default to 2 (dual V100)
if command -v nvidia-smi &>/dev/null; then
  DETECTED_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  GPU_IDS="${GPU_IDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd ',')}"
else
  DETECTED_GPU_COUNT=0
  GPU_IDS="${GPU_IDS:-0,1}"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"

# Auto-tune batch size for V100 (16GB/32GB) vs A100 (40GB/80GB)
V100_BS_FACTOR="${V100_BS_FACTOR:-1}"
V100_DETECTED=false
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr '[:upper:]' '[:lower:]')
  if [[ "${GPU_NAME}" == *"v100"* ]]; then
    V100_BS_FACTOR=2  # divide by 2
    V100_DETECTED=true
  fi
fi

# Parse flags
PARALLEL_LSM=false
SKIP_BASELINES=false
SKIP_LSM=false
SKIP_COMBO=false
for arg in "$@"; do
  case "$arg" in
    --parallel-lsm) PARALLEL_LSM=true ;;
    --skip-baselines) SKIP_BASELINES=true ;;
    --skip-lsm) SKIP_LSM=true ;;
    --skip-combo) SKIP_COMBO=true ;;
  esac
done

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Pre-flight checks ──────────────────────────────────────────────────────
log_info "Pre-flight checks..."

# Check GPU availability
if command -v nvidia-smi &>/dev/null; then
  AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  log_info "Detected ${AVAILABLE_GPUS} GPUs"
else
  log_warn "nvidia-smi not found — assuming CPU-only mode for baselines"
  AVAILABLE_GPUS=0
fi

# Check base model checkpoints
BASE_MODEL_DIR="${WORK_DIR}/checkpoints/base_models_a100_8gpu_50ksteps"
MISSING_CKPTS=0
for model in dp nep mtp soap painn schnet mace; do
  if [[ ! -f "${BASE_MODEL_DIR}/${model}_torch.pth" ]]; then
    log_error "Missing base model checkpoint: ${model}_torch.pth"
    MISSING_CKPTS=$((MISSING_CKPTS + 1))
  fi
done
if [[ ${MISSING_CKPTS} -gt 0 ]]; then
  log_error "${MISSING_CKPTS} base model checkpoints missing. Aborting."
  exit 1
fi
log_info "All 7 base model checkpoints found."

# Check data files
for f in "${WORK_DIR}/data/train.extxyz" "${WORK_DIR}/data/test.extxyz"; do
  if [[ ! -f "$f" ]]; then
    log_error "Missing data file: $f"
    exit 1
  fi
done
log_info "Data files found."

# Create output directories
mkdir -p "${REPORTS_DIR}/direct_combo_metrics"
mkdir -p "${REPORTS_DIR}/joint_comparison"
mkdir -p "${WORK_DIR}/checkpoints/meta_models/direct_combo"
mkdir -p "${WORK_DIR}/logs"

# ─── Helper: run with GPU subset ────────────────────────────────────────────
run_with_gpus() {
  local gpu_ids="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu_ids}" "$@"
}

# ─── Phase 1: Baselines (can run in parallel) ───────────────────────────────
phase1_baselines() {
  log_info "=== Phase 1: Baselines ==="
  log_info "AVG, LWA, LNC baselines can run in parallel on 1 GPU or CPU"

  if [[ "${SKIP_BASELINES}" == "true" ]]; then
    log_warn "Skipping baselines (--skip-baselines)"
    return
  fi

  # Use first GPU for baselines (or CPU)
  BASELINE_GPU="${GPU_ARRAY[0]}"

  # AVG baseline
  log_info "[1/3] Running AVG baseline..."
  python "${PROJECT_ROOT}/eval_avg_baseline.py" \
    --batch-size 8 \
    --device "cuda:${BASELINE_GPU}" \
    2>&1 | tee "${WORK_DIR}/logs/avg_baseline.log"

  # LWA baseline
  log_info "[2/3] Running LWA baseline..."
  python "${PROJECT_ROOT}/eval_lwa_baseline.py" \
    --batch-size 8 \
    --device "cuda:${BASELINE_GPU}" \
    2>&1 | tee "${WORK_DIR}/logs/lwa_baseline.log"

  # LNC baseline
  log_info "[3/3] Running LNC baseline..."
  python "${PROJECT_ROOT}/eval_lnc_baseline.py" \
    --batch-size 8 \
    --lnc-epochs 100 \
    --device "cuda:${BASELINE_GPU}" \
    2>&1 | tee "${WORK_DIR}/logs/lnc_baseline.log"

  log_info "Phase 1 complete."
}

# ─── Phase 2: LSM models ────────────────────────────────────────────────────
phase2_lsm() {
  log_info "=== Phase 2: LSM Models ==="

  if [[ "${SKIP_LSM}" == "true" ]]; then
    log_warn "Skipping LSM (--skip-lsm)"
    return
  fi

  # Auto-tune batch size for V100 vs A100
  LSM_DIRECT_BS=$((8 / V100_BS_FACTOR))
  LSM_CONSERV_BS=$((4 / V100_BS_FACTOR))
  if [[ ${LSM_DIRECT_BS} -lt 1 ]]; then LSM_DIRECT_BS=1; fi
  if [[ ${LSM_CONSERV_BS} -lt 1 ]]; then LSM_CONSERV_BS=1; fi

  if [[ "${V100_DETECTED}" == "true" ]]; then
    log_warn "V100 detected — batch size halved (Direct=${LSM_DIRECT_BS}, Conserv=${LSM_CONSERV_BS})"
  fi

  if [[ ${NUM_GPUS} -ge 8 && "${PARALLEL_LSM}" == "true" ]]; then
    # Run both LSMs in parallel on different GPU groups
    log_info "Running LSM-Direct and LSM-Conserv in parallel (4 GPUs each)"

    LSM1_GPUS="${GPU_ARRAY[0]},${GPU_ARRAY[1]},${GPU_ARRAY[2]},${GPU_ARRAY[3]}"
    LSM2_GPUS="${GPU_ARRAY[4]},${GPU_ARRAY[5]},${GPU_ARRAY[6]},${GPU_ARRAY[7]}"

    (
      run_with_gpus "${LSM1_GPUS}" torchrun --standalone --nproc_per_node=4 \
        "${PROJECT_ROOT}/train_lsm_direct.py" \
        --batch-size "${LSM_DIRECT_BS}" \
        --target-total-steps 50000 \
        2>&1 | tee "${WORK_DIR}/logs/lsm_direct.log"
    ) &
    LSM1_PID=$!

    (
      run_with_gpus "${LSM2_GPUS}" torchrun --standalone --nproc_per_node=4 \
        "${PROJECT_ROOT}/train_lsm_conserv.py" \
        --batch-size "${LSM_CONSERV_BS}" \
        --target-total-steps 50000 \
        2>&1 | tee "${WORK_DIR}/logs/lsm_conserv.log"
    ) &
    LSM2_PID=$!

    wait "${LSM1_PID}"
    wait "${LSM2_PID}"

  elif [[ ${NUM_GPUS} -ge 4 ]]; then
    # Sequential LSM training on all available GPUs
    LSM_GPUS="${GPU_ARRAY[0]},${GPU_ARRAY[1]},${GPU_ARRAY[2]},${GPU_ARRAY[3]}"

    log_info "[1/2] Training LSM-Direct (4 GPUs, ~2-4 hours)..."
    run_with_gpus "${LSM_GPUS}" torchrun --standalone --nproc_per_node=4 \
      "${PROJECT_ROOT}/train_lsm_direct.py" \
      --batch-size "${LSM_DIRECT_BS}" \
      --target-total-steps 50000 \
      2>&1 | tee "${WORK_DIR}/logs/lsm_direct.log"

    log_info "[2/2] Training LSM-Conserv (4 GPUs, ~3-5 hours)..."
    run_with_gpus "${LSM_GPUS}" torchrun --standalone --nproc_per_node=4 \
      "${PROJECT_ROOT}/train_lsm_conserv.py" \
      --batch-size "${LSM_CONSERV_BS}" \
      --target-total-steps 50000 \
      2>&1 | tee "${WORK_DIR}/logs/lsm_conserv.log"

  elif [[ ${NUM_GPUS} -ge 2 ]]; then
    # Dual GPU (e.g., V100 x2)
    LSM_GPUS="${GPU_ARRAY[0]},${GPU_ARRAY[1]}"

    log_info "[1/2] Training LSM-Direct (2 GPUs, batch=${LSM_DIRECT_BS})..."
    run_with_gpus "${LSM_GPUS}" torchrun --standalone --nproc_per_node=2 \
      "${PROJECT_ROOT}/train_lsm_direct.py" \
      --batch-size "${LSM_DIRECT_BS}" \
      --target-total-steps 50000 \
      2>&1 | tee "${WORK_DIR}/logs/lsm_direct.log"

    log_info "[2/2] Training LSM-Conserv (2 GPUs, batch=${LSM_CONSERV_BS})..."
    run_with_gpus "${LSM_GPUS}" torchrun --standalone --nproc_per_node=2 \
      "${PROJECT_ROOT}/train_lsm_conserv.py" \
      --batch-size "${LSM_CONSERV_BS}" \
      --target-total-steps 50000 \
      2>&1 | tee "${WORK_DIR}/logs/lsm_conserv.log"

  else
    log_warn "Only ${NUM_GPUS} GPU(s) available. LSM training may be slow or OOM."
    log_warn "Consider reducing --batch-size or --hidden-dim."

    LSM_GPUS="${GPU_IDS}"
    NPROC="${NUM_GPUS}"

    log_info "[1/2] Training LSM-Direct (${NPROC} GPU(s))..."
    run_with_gpus "${LSM_GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
      "${PROJECT_ROOT}/train_lsm_direct.py" \
      --batch-size "${LSM_DIRECT_BS}" \
      --target-total-steps 50000 \
      2>&1 | tee "${WORK_DIR}/logs/lsm_direct.log"

    log_info "[2/2] Training LSM-Conserv (${NPROC} GPU(s))..."
    run_with_gpus "${LSM_GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
      "${PROJECT_ROOT}/train_lsm_conserv.py" \
      --batch-size "${LSM_CONSERV_BS}" \
      --target-total-steps 50000 \
      2>&1 | tee "${WORK_DIR}/logs/lsm_conserv.log"
  fi

  log_info "Phase 2 complete."
}

# ─── Phase 3: Direct Combo Sweep ────────────────────────────────────────────
phase3_combo() {
  log_info "=== Phase 3: Direct Combo Sweep (127 models) ==="

  if [[ "${SKIP_COMBO}" == "true" ]]; then
    log_warn "Skipping combo sweep (--skip-combo)"
    return
  fi

  # Configure combo batch scripts for detected GPU count
  export NPROC_PER_TASK="${NPROC_PER_TASK:-${NUM_GPUS}}"
  if [[ ${NUM_GPUS} -ge 8 ]]; then
    export TASK1_GPU_IDS="${TASK1_GPU_IDS:-0,1,2,3}"
    export TASK2_GPU_IDS="${TASK2_GPU_IDS:-4,5,6,7}"
    log_info "Combo mode: 2 tasks in parallel (4 GPUs each)"
    log_info "Estimated total time: 3-6 days on 8xA100."
  elif [[ ${NUM_GPUS} -ge 4 ]]; then
    export TASK1_GPU_IDS="${TASK1_GPU_IDS:-0,1,2,3}"
    export TASK2_GPU_IDS=""
    log_info "Combo mode: 1 task at a time (4 GPUs)"
    log_info "Estimated total time: ~6-10 days on 4-GPU machine."
  elif [[ ${NUM_GPUS} -ge 2 ]]; then
    export TASK1_GPU_IDS="${TASK1_GPU_IDS:-0,1}"
    export TASK2_GPU_IDS=""
    export NPROC_PER_TASK=2
    log_warn "Only ${NUM_GPUS} GPUs — combo will run single-task with 2 GPUs."
    log_warn "Batch size auto-reduced for V100 if detected."
    log_warn "Estimated total time: ~10-20 days on dual V100. Consider using more GPUs."
  else
    export TASK1_GPU_IDS="${TASK1_GPU_IDS:-0}"
    export TASK2_GPU_IDS=""
    export NPROC_PER_TASK=1
    log_warn "Only 1 GPU — combo will take a very long time."
  fi

  log_info "Submitting 30 batch scripts sequentially..."
  log_info "Press Ctrl+C within 5 seconds to cancel and submit manually."
  sleep 5

  for script in "${PROJECT_ROOT}/direct_combo_batch_submit_scripts_30"/*.sh; do
    log_info "Running batch: $(basename "$script")"
    bash "$script"
  done

  log_info "Phase 3 complete."
}

# ─── Phase 4: Collect Results ───────────────────────────────────────────────
phase4_collect() {
  log_info "=== Phase 4: Collect Results ==="

  log_info "[1/3] Collecting direct combo metrics..."
  python "${PROJECT_ROOT}/eval_direct_combo_metrics.py" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/meta_models/direct_combo" \
    --output-dir "${REPORTS_DIR}/direct_combo_metrics"

  log_info "[2/3] Generating joint comparison plot..."
  python "${PROJECT_ROOT}/plot_direct_vs_conservative.py" \
    --output-dir "${REPORTS_DIR}/joint_comparison"

  log_info "[3/3] Results summary:"
  echo ""
  echo "  Direct combo metrics: ${REPORTS_DIR}/direct_combo_metrics/"
  echo "  Joint comparison:     ${REPORTS_DIR}/joint_comparison/"
  echo "  AVG baseline:         ${REPORTS_DIR}/avg_baseline_metrics.csv"
  echo "  LWA baseline:         ${REPORTS_DIR}/lwa_baseline_metrics.csv"
  echo "  LNC baseline:         ${REPORTS_DIR}/lnc_baseline_metrics.csv"
  echo "  LSM-Direct:           ${WORK_DIR}/checkpoints/lsm_direct.pth"
  echo "  LSM-Conserv:          ${WORK_DIR}/checkpoints/lsm_conserv.pth"
  echo ""

  log_info "All phases complete!"
}

# ─── Resource Summary ───────────────────────────────────────────────────────
print_resource_summary() {
  if [[ "${V100_DETECTED}" == "true" ]]; then
    local gpu_label="V100"
  else
    local gpu_label="A100"
  fi

  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║     Direct Ensemble Experiment — Resource Summary            ║"
  echo "╠══════════════════════════════════════════════════════════════╣"
  echo "║ GPUs detected:     ${NUM_GPUS}  (${gpu_label})                                    ║"
  echo "║                                                              ║"
  echo "║ Phase          │ GPUs   │ Time (est.)              │ Mode   ║"
  echo "║ ───────────────┼────────┼────────────────────────┼───────── ║"
  echo "║ Baselines      │ 1      │ ~4 hours               │ single  ║"
  echo "║ LSM-Direct     │ 2-4    │ ~3-6 hours             │ DDP     ║"
  echo "║ LSM-Conserv    │ 2-4    │ ~4-8 hours             │ DDP     ║"
  echo "║ Direct Combo   │ 1-8    │ ~10-120 hours          │ batch   ║"
  echo "║ (127 models)   │        │ (depends on GPU count) │         ║"
  echo "║ ───────────────┼────────┼────────────────────────┼───────── ║"
  if [[ ${NUM_GPUS} -ge 8 ]]; then
    echo "║ Your config    │ ${NUM_GPUS}      │ ~100-130 hours total   │ optimal ║"
  elif [[ ${NUM_GPUS} -ge 4 ]]; then
    echo "║ Your config    │ ${NUM_GPUS}      │ ~150-200 hours total   │ reduced ║"
  elif [[ ${NUM_GPUS} -ge 2 ]]; then
    echo "║ Your config    │ ${NUM_GPUS}      │ ~250-400 hours total   │ minimal ║"
  else
    echo "║ Your config    │ ${NUM_GPUS}      │ ~500+ hours total      │ slow    ║"
  fi
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Options:"
  echo "  --parallel-lsm   Run LSM-Direct and LSM-Conserv simultaneously (needs 8 GPUs)"
  echo "  --skip-baselines Skip AVG/LWA/LNC baselines"
  echo "  --skip-lsm       Skip LSM training"
  echo "  --skip-combo     Skip 127 combo sweep"
  echo ""
}

# ─── Main ───────────────────────────────────────────────────────────────────
main() {
  print_resource_summary

  log_info "Starting all phases in sequence..."
  log_info "Flags: parallel-lsm=${PARALLEL_LSM}, skip-baselines=${SKIP_BASELINES}, skip-lsm=${SKIP_LSM}, skip-combo=${SKIP_COMBO}"

  phase1_baselines
  phase2_lsm
  phase3_combo
  phase4_collect
}

main "$@"
