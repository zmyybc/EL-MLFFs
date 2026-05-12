from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "base_model_variant_numeric_scripts"
MANIFEST_NAME = "manifest.csv"
README_NAME = "README.md"

COMMON_TRAINING = {
    "lr": "5e-4",
    "min_lr": "1e-6",
    "energy_weight": "1.0",
    "force_weight": "50.0",
    "cutoff": "5.0",
}

MODEL_VARIANTS = {
    "schnet": [
        {
            "name": "baseline",
            "description": "Default SchNet architecture.",
            "model_kwargs": {},
        },
        {
            "name": "compact",
            "description": "Smaller SchNet with narrower channels and fewer interactions.",
            "model_kwargs": {"hidden_channels": 96, "num_filters": 96, "num_interactions": 2, "num_gaussians": 40},
        },
        {
            "name": "tiny",
            "description": "Further reduced SchNet with narrower channels and fewer basis functions.",
            "model_kwargs": {"hidden_channels": 64, "num_filters": 64, "num_interactions": 2, "num_gaussians": 32},
        },
    ],
    "painn": [
        {
            "name": "baseline",
            "description": "Default PaiNN architecture.",
            "model_kwargs": {},
        },
        {
            "name": "compact",
            "description": "Smaller PaiNN with fewer layers and basis functions.",
            "model_kwargs": {"hidden_channels": 96, "num_layers": 2, "num_basis": 24},
        },
        {
            "name": "tiny",
            "description": "Further reduced PaiNN with fewer channels and basis functions.",
            "model_kwargs": {"hidden_channels": 64, "num_layers": 2, "num_basis": 16},
        },
    ],
    "dp": [
        {
            "name": "baseline",
            "description": "Default DP-style architecture.",
            "model_kwargs": {},
        },
        {
            "name": "compact",
            "description": "Smaller DP descriptor and MLP width.",
            "model_kwargs": {
                "hidden_channels": 96,
                "descriptor_embed_dim": 48,
                "descriptor_axis_dim": 12,
                "descriptor_type_emb_dim": 12,
            },
        },
        {
            "name": "tiny",
            "description": "Further reduced DP descriptor and MLP width.",
            "model_kwargs": {
                "hidden_channels": 64,
                "descriptor_embed_dim": 32,
                "descriptor_axis_dim": 8,
                "descriptor_type_emb_dim": 8,
            },
        },
    ],
    "nep": [
        {
            "name": "baseline",
            "description": "Default NEP-style architecture.",
            "model_kwargs": {},
        },
        {
            "name": "compact",
            "description": "Smaller NEP with reduced radial and angular basis orders.",
            "model_kwargs": {
                "hidden_channels": 96,
                "n_max_radial": 8,
                "n_max_angular": 4,
                "basis_radial": 8,
                "basis_angular": 6,
            },
        },
        {
            "name": "tiny",
            "description": "Further reduced NEP with smaller radial and angular basis orders.",
            "model_kwargs": {
                "hidden_channels": 64,
                "n_max_radial": 6,
                "n_max_angular": 3,
                "basis_radial": 6,
                "basis_angular": 4,
            },
        },
    ],
    "mtp": [
        {
            "name": "baseline",
            "description": "Default MTP-style architecture.",
            "model_kwargs": {},
        },
        {
            "name": "compact",
            "description": "Smaller MTP with reduced basis size and narrower hidden layers.",
            "model_kwargs": {
                "hidden_channels": 96,
                "descriptor_num_basis": 12,
                "descriptor_type_emb_dim": 6,
                "descriptor_radial_hidden_dim": 48,
            },
        },
        {
            "name": "tiny",
            "description": "Further reduced MTP with smaller basis size and narrower hidden layers.",
            "model_kwargs": {
                "hidden_channels": 64,
                "descriptor_num_basis": 8,
                "descriptor_type_emb_dim": 4,
                "descriptor_radial_hidden_dim": 32,
            },
        },
    ],
    "soap": [
        {
            "name": "baseline",
            "description": "Default SOAP-style architecture.",
            "model_kwargs": {},
        },
        {
            "name": "compact",
            "description": "Smaller SOAP power spectrum with fewer radial channels.",
            "model_kwargs": {"hidden_channels": 96, "num_radial": 6},
        },
        {
            "name": "tiny",
            "description": "Further reduced SOAP power spectrum with fewer radial channels.",
            "model_kwargs": {"hidden_channels": 64, "num_radial": 4},
        },
    ],
    "mace": [
        {
            "name": "baseline",
            "description": "Capped MACE baseline kept under 3M parameters.",
            "model_kwargs": {"num_layers": 3, "num_basis": 24, "hidden_scalar_dim": 84, "hidden_vector_dim": 42},
        },
        {
            "name": "compact",
            "description": "Smaller MACE with fewer channels, still under the 3M cap.",
            "model_kwargs": {"num_layers": 3, "num_basis": 20, "hidden_scalar_dim": 80, "hidden_vector_dim": 40},
        },
        {
            "name": "tiny",
            "description": "Further reduced MACE with fewer interactions and narrower irreps.",
            "model_kwargs": {"num_layers": 2, "num_basis": 16, "hidden_scalar_dim": 64, "hidden_vector_dim": 32},
        },
    ],
}


def build_model_kwarg_args(model_kwargs: dict[str, int | float | str]) -> str:
    if not model_kwargs:
        return "MODEL_KWARG_ARGS=()"

    lines = ["MODEL_KWARG_ARGS=("]
    for key, value in sorted(model_kwargs.items()):
        lines.append(f'  --model-kwarg "{key}={value}"')
    lines.append(")")
    return "\n".join(lines)


def build_script(index: int, model_name: str, variant: dict[str, object]) -> str:
    variant_name = str(variant["name"])
    model_kwarg_block = build_model_kwarg_args(variant["model_kwargs"])
    model_kwargs_json = json.dumps(variant["model_kwargs"], sort_keys=True)
    return f"""#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"

GPU_ID="${{GPU_ID:-0}}"
TRAIN_FILE="${{TRAIN_FILE:-${{PROJECT_ROOT}}/el-mlffs/data/train.extxyz}}"
VAL_FILE="${{VAL_FILE:-${{PROJECT_ROOT}}/el-mlffs/data/test.extxyz}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-${{PROJECT_ROOT}}/el-mlffs/checkpoints/base_model_variants}}"
LOG_DIR="${{LOG_DIR:-${{PROJECT_ROOT}}/el-mlffs/logs/base_model_variants}}"

MODEL_NAME="{model_name}"
VARIANT_NAME="{variant_name}"
TARGET_TOTAL_STEPS="${{TARGET_TOTAL_STEPS:-50000}}"
LR="${{LR:-{COMMON_TRAINING["lr"]}}}"
MIN_LR="${{MIN_LR:-{COMMON_TRAINING["min_lr"]}}}"
ENERGY_WEIGHT="${{ENERGY_WEIGHT:-{COMMON_TRAINING["energy_weight"]}}}"
FORCE_WEIGHT="${{FORCE_WEIGHT:-{COMMON_TRAINING["force_weight"]}}}"
CUTOFF="${{CUTOFF:-{COMMON_TRAINING["cutoff"]}}}"
INITIAL_BATCH_SIZE="${{INITIAL_BATCH_SIZE:-128}}"
MAX_BATCH_SIZE="${{MAX_BATCH_SIZE:-4096}}"
TARGET_MEMORY_FRACTION="${{TARGET_MEMORY_FRACTION:-0.80}}"
MEMORY_HEADROOM_FRACTION="${{MEMORY_HEADROOM_FRACTION:-0.05}}"
NUM_WORKERS="${{NUM_WORKERS:-4}}"
SEED="${{SEED:-42}}"
TRAIN_RATIO="${{TRAIN_RATIO:-0.9}}"
ENV_NAME="${{ENV_NAME:-horm}}"
PYTHON_VERSION="${{PYTHON_VERSION:-3.11}}"
TORCH_VERSION="${{TORCH_VERSION:-2.2.1}}"
TORCH_INDEX_URL="${{TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}}"
PYG_WHEEL_URL="${{PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}}"

SAVE_DIR="${{OUTPUT_ROOT}}/${{MODEL_NAME}}/${{VARIANT_NAME}}"
CKPT_PATH="${{SAVE_DIR}}/${{MODEL_NAME}}_torch.pth"
DONE_FILE="${{SAVE_DIR}}/.done"
LOG_FILE="${{LOG_DIR}}/{index:02d}_${{MODEL_NAME}}_${{VARIANT_NAME}}.log"

{model_kwarg_block}

mkdir -p "${{SAVE_DIR}}" "${{LOG_DIR}}"

if [[ -f "${{DONE_FILE}}" ]]; then
  echo "Skip: completed marker already exists at ${{DONE_FILE}}"
  exit 0
fi

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba not found in PATH." >&2
  exit 1
fi

export MAMBA_ROOT_PREFIX="${{MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}}"
set +u
eval "$(mamba shell hook --shell bash)"
set -u

if mamba env list | awk '{{print $1}}' | grep -qx "${{ENV_NAME}}"; then
  echo "Detected existing mamba env: ${{ENV_NAME}}."
else
  echo "Creating mamba env: ${{ENV_NAME}}"
  mamba create -y -n "${{ENV_NAME}}" "python=${{PYTHON_VERSION}}"
fi

set +u
mamba activate "${{ENV_NAME}}"
set -u

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${{TORCH_INDEX_URL}}" "torch==${{TORCH_VERSION}}"
python -m pip install numpy==1.26.4 ase e3nn matplotlib seaborn tqdm
python -m pip install torch-cluster -f "${{PYG_WHEEL_URL}}"
python -m pip install torch-scatter -f "${{PYG_WHEEL_URL}}"
python -m pip install torch-sparse -f "${{PYG_WHEEL_URL}}"
python -m pip install torch-geometric

echo "Launching base-model architecture variant"
echo "  model=${{MODEL_NAME}}"
echo "  variant=${{VARIANT_NAME}}"
echo "  gpu=${{GPU_ID}}"
echo "  model_kwargs={model_kwargs_json}"
echo "  lr=${{LR}} min_lr=${{MIN_LR}}"
echo "  energy_weight=${{ENERGY_WEIGHT}} force_weight=${{FORCE_WEIGHT}} cutoff=${{CUTOFF}}"
echo "  target_total_steps=${{TARGET_TOTAL_STEPS}}"
echo "  log=${{LOG_FILE}}"
echo "  ckpt=${{CKPT_PATH}}"

export CUDA_VISIBLE_DEVICES="${{GPU_ID}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

python "${{PROJECT_ROOT}}/train_one_base_model.py" \\
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
  --memory-headroom-fraction "${{MEMORY_HEADROOM_FRACTION}}" \\
  --target-total-steps "${{TARGET_TOTAL_STEPS}}" \\
  --output-dir "${{SAVE_DIR}}" \\
  "${{MODEL_KWARG_ARGS[@]}}" \\
  2>&1 | tee "${{LOG_FILE}}"

status="${{PIPESTATUS[0]}}"
if [[ "${{status}}" -eq 0 ]]; then
  touch "${{DONE_FILE}}"
fi
exit "${{status}}"
"""


def build_readme() -> str:
    lines = [
        "# Numeric Base Model Variant Scripts",
        "",
        "This directory contains 21 single-GPU scripts named `1.sh` through `21.sh`.",
        "",
        "Each script trains exactly one `(base model, architecture variant)` pair.",
        "",
        "Training backend:",
        "- Uses `train_one_base_model.py`.",
        "- Keeps the CUDA memory probing / batch-size autotune logic.",
        "- Converts `target_total_steps` into epochs automatically.",
        "- Configures environment inside each script with proxy exports, `mamba create/activate`, and direct `pip install` commands.",
        "",
        "Architecture changes now target constructor kwargs such as `hidden_channels`, `num_layers`, `num_basis`,",
        "`num_interactions`, `num_gaussians`, and descriptor-specific widths/orders depending on the model family.",
        "",
        "Use `manifest.csv` to map script numbers to model / variant details.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    index = 1
    for model_name, variants in MODEL_VARIANTS.items():
        for variant in variants:
            filename = f"{index}.sh"
            script_path = OUTPUT_DIR / filename
            script_path.write_text(build_script(index, model_name, variant), encoding="utf-8")
            script_path.chmod(0o755)
            rows.append(
                {
                    "index": str(index),
                    "model_name": model_name,
                    "variant_name": str(variant["name"]),
                    "description": str(variant["description"]),
                    "model_kwargs": json.dumps(variant["model_kwargs"], sort_keys=True),
                    "lr": COMMON_TRAINING["lr"],
                    "min_lr": COMMON_TRAINING["min_lr"],
                    "energy_weight": COMMON_TRAINING["energy_weight"],
                    "force_weight": COMMON_TRAINING["force_weight"],
                    "cutoff": COMMON_TRAINING["cutoff"],
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
                "description",
                "model_kwargs",
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

    (OUTPUT_DIR / README_NAME).write_text(build_readme(), encoding="utf-8")
    print(f"Generated {len(rows)} scripts in {OUTPUT_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
