#!/usr/bin/env bash
# Training Ablation Experiments — these all REQUIRE training.
# For speed benchmarking (no training), use benchmark_speed.py instead.
# Usage: bash run_ablation_experiments.sh [--phase <name>] [--gpu <id>]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_ROOT}/el-mlffs"
REPORTS_DIR="${PROJECT_ROOT}/reports"

# GPU setup — auto-detect or use env var
if command -v nvidia-smi &>/dev/null; then
  DETECTED_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  GPU_IDS="${GPU_IDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd ',')}"
else
  DETECTED_GPU_COUNT=1
  GPU_IDS="${GPU_IDS:-0}"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"

GPU_ID="${GPU_ID:-0}"
DEVICE="cuda:${GPU_ID}"

# V100 auto-tune
V100_BS_FACTOR="${V100_BS_FACTOR:-1}"
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr '[:upper:]' '[:lower:]')
  if [[ "${GPU_NAME}" == *"v100"* ]]; then
    V100_BS_FACTOR=2
  fi
fi
AB_BS=$((8 / V100_BS_FACTOR))
if [[ ${AB_BS} -lt 1 ]]; then AB_BS=1; fi

# Parse flags
PHASE="all"
for arg in "$@"; do
  case "$arg" in
    --phase)   shift; PHASE="$1"; shift ;;
    --gpu)     shift; GPU_ID="$1"; DEVICE="cuda:${GPU_ID}"; shift ;;
    --gpu-id)  shift; GPU_ID="$1"; DEVICE="cuda:${GPU_ID}"; shift ;;
  esac
done

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

mkdir -p "${REPORTS_DIR}/ablation"
mkdir -p "${WORK_DIR}/logs"

# ─────────────────────────────────────────────────────────────────────────────
# Phase A: Ensemble Architecture Ablations (Direct)
# ─────────────────────────────────────────────────────────────────────────────
ablation_direct_architecture() {
  log_info "=== Ablation A: Direct Ensemble Architecture ==="

  local base_models="painn schnet mace"
  local steps=5000

  # A1: hidden_scalar_channels
  for hsc in 32 64 128 256; do
    log_info "[A1] Direct hidden_scalar_channels=${hsc}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models ${base_models} \
      --target-total-steps ${steps} \
      --lr 1e-3 \
      --batch-size ${AB_BS} \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_hsc_${hsc}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_hsc_${hsc}.log"
  done

  # A2: num_layers
  for nl in 2 3 4 6; do
    log_info "[A2] Direct num_layers=${nl}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models ${base_models} \
      --target-total-steps ${steps} \
      --lr 1e-3 \
      --batch-size ${AB_BS} \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_nl_${nl}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_nl_${nl}.log"
  done

  # A3: force_weight
  for fw in 10.0 25.0 50.0 100.0; do
    log_info "[A3] Direct force_weight=${fw}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models ${base_models} \
      --target-total-steps ${steps} \
      --lr 1e-3 \
      --batch-size ${AB_BS} \
      --force-weight ${fw} \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_fw_${fw}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_fw_${fw}.log"
  done

  log_info "Phase A complete."
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase B: Ensemble Size Ablation (Direct, subset of combos)
# ─────────────────────────────────────────────────────────────────────────────
ablation_ensemble_size() {
  log_info "=== Ablation B: Ensemble Size ==="

  local steps=5000

  # B1: 1-model ensembles
  for model in dp nep mtp soap painn schnet mace; do
    log_info "[B1] Single model: ${model}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models "${model}" \
      --target-total-steps ${steps} \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_1model_${model}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_1model_${model}.log"
  done

  # B2: 3-model ensembles (representative combos)
  local combos=(
    "dp nep mtp"
    "painn schnet mace"
    "dp soap mace"
  )
  for combo in "${combos[@]}"; do
    tag=$(echo "${combo}" | tr ' ' '_')
    log_info "[B2] 3-model ensemble: ${combo}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models ${combo} \
      --target-total-steps ${steps} \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_3model_${tag}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_3model_${tag}.log"
  done

  # B3: 5-model ensembles
  log_info "[B3] 5-model ensemble"
  python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
    --base-models dp nep mtp soap mace \
    --target-total-steps ${steps} \
    --save-path "${WORK_DIR}/checkpoints/ablation/direct_5model.pth" \
    2>&1 | tee "${WORK_DIR}/logs/ablation_direct_5model.log"

  # B4: 7-model ensemble (full)
  log_info "[B4] 7-model ensemble (full)"
  python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
    --base-models dp nep mtp soap painn schnet mace \
    --target-total-steps ${steps} \
    --save-path "${WORK_DIR}/checkpoints/ablation/direct_7model.pth" \
    2>&1 | tee "${WORK_DIR}/logs/ablation_direct_7model.log"

  log_info "Phase B complete."
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase C: Learning Rate & Batch Size Ablation
# ─────────────────────────────────────────────────────────────────────────────
ablation_lr_batch() {
  log_info "=== Ablation C: LR & Batch Size ==="

  local base_models="painn schnet mace"
  local steps=5000

  # C1: LR sweep
  for lr in 5e-4 1e-3 2e-3 5e-3; do
    log_info "[C1] LR=${lr}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models ${base_models} \
      --target-total-steps ${steps} \
      --lr ${lr} \
      --batch-size 8 \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_lr_${lr}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_lr_${lr}.log"
  done

  # C2: Batch size sweep
  for bs in 4 8 16 32; do
    log_info "[C2] batch_size=${bs}"
    python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
      --base-models ${base_models} \
      --target-total-steps ${steps} \
      --lr 1e-3 \
      --batch-size ${bs} \
      --save-path "${WORK_DIR}/checkpoints/ablation/direct_bs_${bs}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_direct_bs_${bs}.log"
  done

  log_info "Phase C complete."
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase D: Conservative vs Direct (same base models, fixed steps)
# ─────────────────────────────────────────────────────────────────────────────
ablation_direct_vs_conserv() {
  log_info "=== Ablation D: Direct vs Conservative (same config) ==="

  local base_models="painn schnet mace"
  local steps=5000

  log_info "[D1] Direct ensemble"
  python "${PROJECT_ROOT}/train_direct_meta_with_bases.py" \
    --base-models ${base_models} \
    --target-total-steps ${steps} \
    --batch-size ${AB_BS} \
    --save-path "${WORK_DIR}/checkpoints/ablation/direct_vs_conserv_direct.pth" \
    2>&1 | tee "${WORK_DIR}/logs/ablation_direct_vs_conserv_direct.log"

  log_info "[D2] Conservative ensemble"
  python "${PROJECT_ROOT}/train_torch_ensemble.py" \
    --base-models ${base_models} \
    --architecture conservative \
    --target-total-steps ${steps} \
    --batch-size ${AB_BS} \
    --save-path "${WORK_DIR}/checkpoints/ablation/direct_vs_conserv_conservative.pth" \
    2>&1 | tee "${WORK_DIR}/logs/ablation_direct_vs_conserv_conservative.log"

  log_info "Phase D complete."
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase E: LSM-Direct Ablations
# ─────────────────────────────────────────────────────────────────────────────
ablation_lsm_direct() {
  log_info "=== Ablation E: LSM-Direct Architecture ==="

  local steps=5000
  local lsm_bs=$((8 / V100_BS_FACTOR))
  if [[ ${lsm_bs} -lt 1 ]]; then lsm_bs=1; fi

  # Use DDP if multiple GPUs available
  if [[ ${NUM_GPUS} -ge 2 ]]; then
    local lsm_gpus="${GPU_ARRAY[0]},${GPU_ARRAY[1]}"
    local nproc=2
  else
    local lsm_gpus="${GPU_ARRAY[0]}"
    local nproc=1
  fi

  # E1: hidden_dim
  for hd in 256 512 768; do
    log_info "[E1] LSM-Direct hidden_dim=${hd} (${nproc} GPU(s), batch=${lsm_bs})"
    CUDA_VISIBLE_DEVICES="${lsm_gpus}" torchrun --standalone --nproc_per_node=${nproc} \
      "${PROJECT_ROOT}/train_lsm_direct.py" \
      --target-total-steps ${steps} \
      --batch-size ${lsm_bs} \
      --hidden-dim ${hd} \
      --save-path "${WORK_DIR}/checkpoints/ablation/lsm_direct_hd_${hd}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_lsm_direct_hd_${hd}.log"
  done

  # E2: num_layers
  for nl in 6 8 10 12; do
    log_info "[E2] LSM-Direct num_layers=${nl} (${nproc} GPU(s), batch=${lsm_bs})"
    CUDA_VISIBLE_DEVICES="${lsm_gpus}" torchrun --standalone --nproc_per_node=${nproc} \
      "${PROJECT_ROOT}/train_lsm_direct.py" \
      --target-total-steps ${steps} \
      --batch-size ${lsm_bs} \
      --num-layers ${nl} \
      --save-path "${WORK_DIR}/checkpoints/ablation/lsm_direct_nl_${nl}.pth" \
      2>&1 | tee "${WORK_DIR}/logs/ablation_lsm_direct_nl_${nl}.log"
  done

  log_info "Phase E complete."
}

# ─────────────────────────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────
main() {
  log_info "Training Ablation Experiments"
  log_info "GPUs detected: ${NUM_GPUS}"
  log_info "Selected phase: ${PHASE}"
  log_info "Device: ${DEVICE}"
  if [[ "${GPU_NAME:-}" == *"v100"* ]]; then
    log_warn "V100 detected — batch size auto-reduced to ${AB_BS}"
  fi

  case "${PHASE}" in
    all)
      ablation_direct_architecture
      ablation_ensemble_size
      ablation_lr_batch
      ablation_direct_vs_conserv
      ablation_lsm_direct
      ;;
    architecture|a) ablation_direct_architecture ;;
    size|b)       ablation_ensemble_size ;;
    lr-batch|c)   ablation_lr_batch ;;
    compare|d)    ablation_direct_vs_conserv ;;
    lsm|e)        ablation_lsm_direct ;;
    *)
      log_error "Unknown phase: ${PHASE}"
      echo "Valid phases: all, architecture (a), size (b), lr-batch (c), compare (d), lsm (e)"
      exit 1
      ;;
  esac

  log_info "All requested ablation phases complete."
}

main "$@"
