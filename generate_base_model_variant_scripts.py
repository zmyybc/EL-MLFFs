from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "base_model_variant_submit_scripts"
MANIFEST_NAME = "manifest.csv"
README_NAME = "README.md"

BASE_MODELS = ["schnet", "painn", "dp", "nep", "mtp", "soap", "mace"]
VARIANTS = [
    {
        "name": "baseline",
        "description": "Current default setup for reference.",
        "lr": "5e-4",
        "min_lr": "1e-6",
        "energy_weight": "1.0",
        "force_weight": "50.0",
        "cutoff": "5.0",
    },
    {
        "name": "low_lr_high_force",
        "description": "More conservative learning rate with slightly stronger force supervision.",
        "lr": "2e-4",
        "min_lr": "5e-7",
        "energy_weight": "1.0",
        "force_weight": "60.0",
        "cutoff": "5.0",
    },
    {
        "name": "energy_balanced_wide_cutoff",
        "description": "Wider cutoff with more energy weight and lighter force weight.",
        "lr": "5e-4",
        "min_lr": "1e-6",
        "energy_weight": "2.0",
        "force_weight": "35.0",
        "cutoff": "5.5",
    },
]


def build_script(index: int, model_name: str, variant: dict[str, str]) -> str:
    variant_name = variant["name"]
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"

PYTHON_BIN="${{PYTHON_BIN:-python}}"
GPU_ID="${{GPU_ID:-0}}"
TRAIN_FILE="${{TRAIN_FILE:-${{PROJECT_ROOT}}/el-mlffs/data/train.extxyz}}"
VAL_FILE="${{VAL_FILE:-${{PROJECT_ROOT}}/el-mlffs/data/test.extxyz}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-${{PROJECT_ROOT}}/el-mlffs/checkpoints/base_model_variants}}"
LOG_DIR="${{LOG_DIR:-${{PROJECT_ROOT}}/el-mlffs/logs/base_model_variants}}"

MODEL_NAME="{model_name}"
VARIANT_NAME="{variant_name}"
TARGET_TOTAL_STEPS="${{TARGET_TOTAL_STEPS:-50000}}"
LR="${{LR:-{variant["lr"]}}}"
MIN_LR="${{MIN_LR:-{variant["min_lr"]}}}"
ENERGY_WEIGHT="${{ENERGY_WEIGHT:-{variant["energy_weight"]}}}"
FORCE_WEIGHT="${{FORCE_WEIGHT:-{variant["force_weight"]}}}"
CUTOFF="${{CUTOFF:-{variant["cutoff"]}}}"
INITIAL_BATCH_SIZE="${{INITIAL_BATCH_SIZE:-128}}"
MAX_BATCH_SIZE="${{MAX_BATCH_SIZE:-4096}}"
TARGET_MEMORY_FRACTION="${{TARGET_MEMORY_FRACTION:-0.85}}"
NUM_WORKERS="${{NUM_WORKERS:-4}}"
SEED="${{SEED:-42}}"
TRAIN_RATIO="${{TRAIN_RATIO:-0.9}}"

SAVE_DIR="${{OUTPUT_ROOT}}/${{MODEL_NAME}}/${{VARIANT_NAME}}"
CKPT_PATH="${{SAVE_DIR}}/${{MODEL_NAME}}_torch.pth"
LOG_FILE="${{LOG_DIR}}/{index:02d}_${{MODEL_NAME}}_${{VARIANT_NAME}}.log"

mkdir -p "${{SAVE_DIR}}" "${{LOG_DIR}}"

if [[ -f "${{CKPT_PATH}}" ]]; then
  echo "Skip: checkpoint already exists at ${{CKPT_PATH}}"
  exit 0
fi

echo "Launching base-model variant"
echo "  model=${{MODEL_NAME}}"
echo "  variant=${{VARIANT_NAME}}"
echo "  gpu=${{GPU_ID}}"
echo "  lr=${{LR}} min_lr=${{MIN_LR}}"
echo "  energy_weight=${{ENERGY_WEIGHT}} force_weight=${{FORCE_WEIGHT}} cutoff=${{CUTOFF}}"
echo "  target_total_steps=${{TARGET_TOTAL_STEPS}}"
echo "  log=${{LOG_FILE}}"
echo "  ckpt=${{CKPT_PATH}}"

export CUDA_VISIBLE_DEVICES="${{GPU_ID}}"

"${{PYTHON_BIN}}" "${{PROJECT_ROOT}}/train_one_base_model.py" \\
  --model-name "${{MODEL_NAME}}" \\
  --data-file "${{TRAIN_FILE}}" \\
  --val-data-file "${{VAL_FILE}}" \\
  --cutoff "${{CUTOFF}}" \\
  --lr "${{LR}}" \\
  --min-lr "${{MIN_LR}}" \\
  --energy-weight "${{ENERGY_WEIGHT}}" \\
  --force-weight "${{FORCE_WEIGHT}}" \\
  --train-ratio "${{TRAIN_RATIO}}" \\
  --seed "${{SEED}}" \\
  --num-workers "${{NUM_WORKERS}}" \\
  --initial-batch-size "${{INITIAL_BATCH_SIZE}}" \\
  --max-batch-size "${{MAX_BATCH_SIZE}}" \\
  --target-memory-fraction "${{TARGET_MEMORY_FRACTION}}" \\
  --target-total-steps "${{TARGET_TOTAL_STEPS}}" \\
  --output-dir "${{SAVE_DIR}}" \\
  2>&1 | tee "${{LOG_FILE}}"

exit "${{PIPESTATUS[0]}}"
"""


def build_readme(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Base Model Variant Scripts",
        "",
        "This directory contains one script per `(base model, hyperparameter variant)` pair.",
        "",
        "Training backend:",
        "- Uses `train_one_base_model.py`.",
        "- Includes the existing CUDA memory probing / batch-size autotune logic.",
        "- Converts `target_total_steps` into epochs automatically.",
        "",
        "Variants:",
    ]
    for variant in VARIANTS:
        lines.append(
            f"- `{variant['name']}`: {variant['description']} "
            f"(lr={variant['lr']}, min_lr={variant['min_lr']}, "
            f"energy_weight={variant['energy_weight']}, force_weight={variant['force_weight']}, cutoff={variant['cutoff']})"
        )
    lines.extend(
        [
            "",
            "Examples:",
            "```bash",
            "bash base_model_variant_submit_scripts/01_schnet_baseline.sh",
            "GPU_ID=3 TARGET_TOTAL_STEPS=70000 bash base_model_variant_submit_scripts/14_mtp_low_lr_high_force.sh",
            "```",
            "",
            "Outputs go to:",
            "- `el-mlffs/checkpoints/base_model_variants/<model>/<variant>/<model>_torch.pth`",
            "- `el-mlffs/logs/base_model_variants/*.log`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    index = 1
    for model_name in BASE_MODELS:
        for variant in VARIANTS:
            filename = f"{index:02d}_{model_name}_{variant['name']}.sh"
            script_path = OUTPUT_DIR / filename
            script_path.write_text(build_script(index, model_name, variant), encoding="utf-8")
            script_path.chmod(0o755)
            rows.append(
                {
                    "index": str(index),
                    "model_name": model_name,
                    "variant_name": variant["name"],
                    "lr": variant["lr"],
                    "min_lr": variant["min_lr"],
                    "energy_weight": variant["energy_weight"],
                    "force_weight": variant["force_weight"],
                    "cutoff": variant["cutoff"],
                    "script_path": str(script_path.relative_to(ROOT)),
                    "checkpoint_path": f"el-mlffs/checkpoints/base_model_variants/{model_name}/{variant['name']}/{model_name}_torch.pth",
                    "log_path": f"el-mlffs/logs/base_model_variants/{index:02d}_{model_name}_{variant['name']}.log",
                }
            )
            index += 1

    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "model_name",
                "variant_name",
                "lr",
                "min_lr",
                "energy_weight",
                "force_weight",
                "cutoff",
                "script_path",
                "checkpoint_path",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_DIR / README_NAME).write_text(build_readme(rows), encoding="utf-8")
    print(f"Generated {len(rows)} scripts in {OUTPUT_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
