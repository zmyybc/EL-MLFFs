from __future__ import annotations

import csv
import itertools
from pathlib import Path


BASE_MODELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")
BATCH_SCRIPT_OUTPUT_DIR = "direct_combo_batch_submit_scripts_30"
BATCH_MANIFEST_FILENAME = "batch_manifest.csv"
TASK_MANIFEST_FILENAME = "task_manifest.csv"
REQUIREMENTS_FILENAME = "requirements-conservative-meta.txt"
HORM_SETUP_PATH = "/mnt/bn/bangchen/HORM/setup.sh"
HORM_SETUP_DIR = "/mnt/bn/bangchen/HORM"
HORM_ENV_NAME = "horm"
NUM_BATCH_SCRIPTS = 30


def iter_model_combinations() -> list[tuple[str, ...]]:
    combinations: list[tuple[str, ...]] = []
    for size in range(1, len(BASE_MODELS) + 1):
        combinations.extend(itertools.combinations(BASE_MODELS, size))
    return combinations


def build_task_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for index, combo in enumerate(iter_model_combinations(), start=1):
        combo_tag = "_".join(combo)
        rows.append(
            {
                "task_index": index,
                "num_models": len(combo),
                "base_models": " ".join(combo),
                "combo_tag": combo_tag,
                "default_save_path": f"el-mlffs/checkpoints/meta_models/direct_combo/{index}_{combo_tag}.pth",
            }
        )
    return rows


def split_into_batches(task_rows: list[dict[str, str | int]], num_batches: int) -> list[list[dict[str, str | int]]]:
    total_tasks = len(task_rows)
    base_batch_size = total_tasks // num_batches
    remainder = total_tasks % num_batches

    batches: list[list[dict[str, str | int]]] = []
    start = 0
    for batch_idx in range(num_batches):
        batch_size = base_batch_size + (1 if batch_idx < remainder else 0)
        end = start + batch_size
        batches.append(task_rows[start:end])
        start = end
    return batches


def build_script_text(batch_index: int, batch_tasks: list[dict[str, str | int]]) -> str:
    task_specs = "\n".join(
        f'TASK_SPECS+=("{task["task_index"]}|{task["combo_tag"]}|{task["num_models"]}|{task["base_models"]}")'
        for task in batch_tasks
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
WORK_DIR="${{PROJECT_ROOT}}/el-mlffs"
BATCH_INDEX="{batch_index}"

TASK1_GPU_IDS="${{TASK1_GPU_IDS:-0,1,2,3}}"
TASK2_GPU_IDS="${{TASK2_GPU_IDS:-4,5,6,7}}"
NPROC_PER_TASK="${{NPROC_PER_TASK:-4}}"
BASE_MODEL_DIR="${{BASE_MODEL_DIR:-${{WORK_DIR}}/checkpoints/base_models_a100_8gpu_50ksteps}}"
TRAIN_FILE="${{TRAIN_FILE:-data/train.extxyz}}"
VAL_FILE="${{VAL_FILE:-data/test.extxyz}}"

AUTO_BATCH_START="${{AUTO_BATCH_START:-4}}"
AUTO_BATCH_STEP="${{AUTO_BATCH_STEP:-1}}"
AUTO_BATCH_MAX="${{AUTO_BATCH_MAX:-64}}"
MEMORY_TARGET_RATIO="${{MEMORY_TARGET_RATIO:-0.80}}"
TARGET_TOTAL_STEPS="${{TARGET_TOTAL_STEPS:-20000}}"
LR="${{LR:-1e-3}}"
MIN_LR="${{MIN_LR:-1e-6}}"
FORCE_WEIGHT="${{FORCE_WEIGHT:-50.0}}"
NUM_WORKERS="${{NUM_WORKERS:-4}}"
GRAD_CLIP_NORM="${{GRAD_CLIP_NORM:-10.0}}"
HUBER_DELTA="${{HUBER_DELTA:-1.0}}"
SEED="${{SEED:-42}}"
SPLIT_STRATEGY="${{SPLIT_STRATEGY:-random}}"
MASTER_PORT_BASE="${{MASTER_PORT_BASE:-29500}}"
PYG_WHEEL_URL="${{PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}}"
ENV_SETUP_LOCK="${{ENV_SETUP_LOCK:-${{PROJECT_ROOT}}/.horm_env_setup.lock}}"

export http_proxy="${{http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}}"
export https_proxy="${{https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}}"
export no_proxy="${{no_proxy:-byted.org}}"

if mamba env list | awk '{{print $1}}' | grep -qx "{HORM_ENV_NAME}"; then
  echo "Detected existing mamba env: {HORM_ENV_NAME}. Skip setup.sh."
else
  echo "Mamba env {HORM_ENV_NAME} not found. Running setup.sh."
  (
    cd "{HORM_SETUP_DIR}"
    bash "{HORM_SETUP_PATH}"
  )
fi

export MAMBA_ROOT_PREFIX="${{MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}}"
set +u
eval "$(mamba shell hook --shell bash)"
mamba activate {HORM_ENV_NAME}
set -u

(
  flock -x 9
  python -m pip install -q -r "${{PROJECT_ROOT}}/{REQUIREMENTS_FILENAME}"
  if python -c "import torch_geometric" >/dev/null 2>&1; then
    echo "Detected torch_geometric in env {HORM_ENV_NAME}."
  else
    echo "torch_geometric missing. Installing PyG dependencies."
    python -m pip install -q torch-cluster -f "${{PYG_WHEEL_URL}}"
    python -m pip install -q torch-scatter -f "${{PYG_WHEEL_URL}}"
    python -m pip install -q torch-sparse -f "${{PYG_WHEEL_URL}}"
    python -m pip install -q torch-geometric
  fi
) 9>"${{ENV_SETUP_LOCK}}"

mkdir -p "${{WORK_DIR}}/checkpoints/meta_models/direct_combo"
mkdir -p "${{WORK_DIR}}/logs/direct_combo_batch"

TASK_SPECS=()
{task_specs}

launch_task() {{
  local task_spec="$1"
  local gpu_ids="$2"

  IFS='|' read -r task_index combo_tag num_models base_models_str <<< "$task_spec"
  local save_path="checkpoints/meta_models/direct_combo/${{task_index}}_${{combo_tag}}.pth"
  local log_path="${{WORK_DIR}}/logs/direct_combo_batch/${{task_index}}_${{combo_tag}}.log"
  local master_port=$((MASTER_PORT_BASE + task_index))
  local -a base_models
  read -r -a base_models <<< "$base_models_str"
  echo "[Batch ${{BATCH_INDEX}}] Launch task ${{task_index}} on GPUs ${{gpu_ids}} | models=${{base_models_str}}" >&2
  (
    cd "${{WORK_DIR}}"
    export CUDA_VISIBLE_DEVICES="${{gpu_ids}}"
    exec torchrun --standalone --nproc_per_node="${{NPROC_PER_TASK}}" --master-port="${{master_port}}" "${{PROJECT_ROOT}}/train_direct_meta_with_bases.py" \\
      --base-model-dir "${{BASE_MODEL_DIR}}" \\
      --base-models "${{base_models[@]}}" \\
      --data-file "${{TRAIN_FILE}}" \\
      --val-data-file "${{VAL_FILE}}" \\
      --batch-size "${{AUTO_BATCH_START}}" \\
      --auto-batch-size \\
      --batch-size-probe-start "${{AUTO_BATCH_START}}" \\
      --batch-size-probe-step "${{AUTO_BATCH_STEP}}" \\
      --batch-size-probe-max "${{AUTO_BATCH_MAX}}" \\
      --memory-target-ratio "${{MEMORY_TARGET_RATIO}}" \\
      --target-total-steps "${{TARGET_TOTAL_STEPS}}" \\
      --lr "${{LR}}" \\
      --min-lr "${{MIN_LR}}" \\
      --force-weight "${{FORCE_WEIGHT}}" \\
      --grad-clip-norm "${{GRAD_CLIP_NORM}}" \\
      --huber-delta "${{HUBER_DELTA}}" \\
      --save-path "${{save_path}}" \\
      --seed "${{SEED}}" \\
      --split-strategy "${{SPLIT_STRATEGY}}" \\
      --num-workers "${{NUM_WORKERS}}"
  ) >"${{log_path}}" 2>&1 &
  LAST_LAUNCHED_PID="$!"
}}

wait_for_group() {{
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "${{status}}"
}}

num_tasks="${{#TASK_SPECS[@]}}"
task_ptr=0
while [ "${{task_ptr}}" -lt "${{num_tasks}}" ]; do
  pids=()

  launch_task "${{TASK_SPECS[task_ptr]}}" "${{TASK1_GPU_IDS}}"
  pids+=("${{LAST_LAUNCHED_PID}}")
  task_ptr=$((task_ptr + 1))

  if [ "${{task_ptr}}" -lt "${{num_tasks}}" ]; then
    launch_task "${{TASK_SPECS[task_ptr]}}" "${{TASK2_GPU_IDS}}"
    pids+=("${{LAST_LAUNCHED_PID}}")
    task_ptr=$((task_ptr + 1))
  fi

  if ! wait_for_group "${{pids[@]}}"; then
    echo "[Batch ${{BATCH_INDEX}}] A task failed. Stopping this batch script."
    exit 1
  fi
done

echo "[Batch ${{BATCH_INDEX}}] All tasks completed."
"""


def main() -> None:
    project_root = Path(__file__).resolve().parent
    script_dir = project_root / BATCH_SCRIPT_OUTPUT_DIR
    script_dir.mkdir(parents=True, exist_ok=True)

    task_rows = build_task_rows()
    batches = split_into_batches(task_rows, NUM_BATCH_SCRIPTS)

    batch_manifest_rows: list[dict[str, str | int]] = []
    for batch_index, batch_tasks in enumerate(batches, start=1):
        script_path = script_dir / f"{batch_index}.sh"
        script_path.write_text(build_script_text(batch_index, batch_tasks), encoding="utf-8")
        script_path.chmod(0o755)
        batch_manifest_rows.append(
            {
                "batch_index": batch_index,
                "num_tasks": len(batch_tasks),
                "task_indices": " ".join(str(task["task_index"]) for task in batch_tasks),
                "base_model_groups": " ; ".join(str(task["base_models"]) for task in batch_tasks),
                "script_path": str(script_path.relative_to(project_root)),
            }
        )

    task_manifest_path = script_dir / TASK_MANIFEST_FILENAME
    with task_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_index", "num_models", "base_models", "combo_tag", "default_save_path"],
        )
        writer.writeheader()
        writer.writerows(task_rows)

    batch_manifest_path = script_dir / BATCH_MANIFEST_FILENAME
    with batch_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["batch_index", "num_tasks", "task_indices", "base_model_groups", "script_path"],
        )
        writer.writeheader()
        writer.writerows(batch_manifest_rows)

    print(f"Generated {len(batch_manifest_rows)} batch submit scripts in {script_dir}")
    print(f"Task manifest written to {task_manifest_path}")
    print(f"Batch manifest written to {batch_manifest_path}")


if __name__ == "__main__":
    main()
