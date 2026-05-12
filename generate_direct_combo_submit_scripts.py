from __future__ import annotations

import csv
import itertools
import os
from pathlib import Path


BASE_MODELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")
SCRIPT_OUTPUT_DIR = "direct_combo_submit_scripts"
MANIFEST_FILENAME = "manifest.csv"
REQUIREMENTS_FILENAME = "requirements-conservative-meta.txt"
HORM_SETUP_PATH = ""
HORM_SETUP_DIR = ""
HORM_ENV_NAME = "horm"


def iter_model_combinations() -> list[tuple[str, ...]]:
    combinations: list[tuple[str, ...]] = []
    for size in range(1, len(BASE_MODELS) + 1):
        combinations.extend(itertools.combinations(BASE_MODELS, size))
    return combinations


def build_script_text(index: int, combo: tuple[str, ...]) -> str:
    combo_tag = "_".join(combo)
    combo_models = " ".join(f'"{model}"' for model in combo)
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
WORK_DIR="${{PROJECT_ROOT}}/el-mlffs"

GPU_IDS="${{GPU_IDS:-0,1,6,7}}"
NPROC_PER_NODE="${{NPROC_PER_NODE:-4}}"
BASE_MODEL_DIR="${{BASE_MODEL_DIR:-${{WORK_DIR}}/checkpoints/base_models_a100_8gpu_50ksteps}}"
TRAIN_FILE="${{TRAIN_FILE:-data/train.extxyz}}"
VAL_FILE="${{VAL_FILE:-data/test.extxyz}}"
SAVE_PATH="${{SAVE_PATH:-checkpoints/meta_models/direct_combo/{index}_{combo_tag}.pth}}"

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
PYG_WHEEL_URL="${{PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}}"
ENV_SETUP_LOCK="${{ENV_SETUP_LOCK:-${{PROJECT_ROOT}}/.horm_env_setup.lock}}"

BASE_MODELS=({combo_models})


if mamba env list | awk '{{print $1}}' | grep -qx "{HORM_ENV_NAME}"; then
  echo "Detected existing mamba env: {HORM_ENV_NAME}. Skip setup.sh."
else
  echo "Mamba env {HORM_ENV_NAME} not found. Running setup.sh."
  if [ -n "{HORM_SETUP_DIR}" ] && [ -n "{HORM_SETUP_PATH}" ]; then
    (
      cd "{HORM_SETUP_DIR}"
      bash "{HORM_SETUP_PATH}"
    )
  fi
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

mkdir -p "${{WORK_DIR}}/$(dirname "${{SAVE_PATH}}")"

cd "${{WORK_DIR}}"
export CUDA_VISIBLE_DEVICES="${{GPU_IDS}}"

exec torchrun --standalone --nproc_per_node="${{NPROC_PER_NODE}}" "${{PROJECT_ROOT}}/train_direct_meta_with_bases.py" \\
  --base-model-dir "${{BASE_MODEL_DIR}}" \\
  --base-models "${{BASE_MODELS[@]}}" \\
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
  --save-path "${{SAVE_PATH}}" \\
  --seed "${{SEED}}" \\
  --split-strategy "${{SPLIT_STRATEGY}}" \\
  --num-workers "${{NUM_WORKERS}}"
"""


def main() -> None:
    project_root = Path(__file__).resolve().parent
    script_dir = project_root / SCRIPT_OUTPUT_DIR
    script_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    for index, combo in enumerate(iter_model_combinations(), start=1):
        script_path = script_dir / f"{index}.sh"
        script_path.write_text(build_script_text(index, combo), encoding="utf-8")
        script_path.chmod(0o755)
        manifest_rows.append(
            {
                "index": index,
                "num_models": len(combo),
                "base_models": " ".join(combo),
                "script_path": str(script_path.relative_to(project_root)),
                "default_save_path": f"el-mlffs/checkpoints/meta_models/direct_combo/{index}_{'_'.join(combo)}.pth",
            }
        )

    manifest_path = script_dir / MANIFEST_FILENAME
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "num_models", "base_models", "script_path", "default_save_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} submit scripts in {script_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
