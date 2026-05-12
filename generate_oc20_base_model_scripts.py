#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "oc20_base_model_scripts"

MODEL_SPECS = [
    {
        "index": 1,
        "model_name": "dp",
        "tag": "dp_compact",
        "initial_batch_size": 8,
        "max_batch_size": 192,
        "model_kwargs": {
            "hidden_channels": 96,
            "descriptor_embed_dim": 48,
            "descriptor_axis_dim": 12,
            "descriptor_type_emb_dim": 12,
        },
    },
    {
        "index": 2,
        "model_name": "nep",
        "tag": "nep_compact",
        "initial_batch_size": 8,
        "max_batch_size": 160,
        "model_kwargs": {
            "hidden_channels": 96,
            "n_max_radial": 8,
            "n_max_angular": 4,
            "basis_radial": 8,
            "basis_angular": 6,
        },
    },
    {
        "index": 3,
        "model_name": "mtp",
        "tag": "mtp_compact",
        "initial_batch_size": 8,
        "max_batch_size": 192,
        "model_kwargs": {
            "hidden_channels": 96,
            "descriptor_num_basis": 12,
            "descriptor_type_emb_dim": 8,
            "descriptor_radial_hidden_dim": 48,
        },
    },
    {
        "index": 4,
        "model_name": "soap",
        "tag": "soap_compact",
        "initial_batch_size": 8,
        "max_batch_size": 128,
        "model_kwargs": {
            "hidden_channels": 96,
            "num_radial": 6,
        },
    },
    {
        "index": 5,
        "model_name": "painn",
        "tag": "painn_compact",
        "initial_batch_size": 4,
        "max_batch_size": 128,
        "model_kwargs": {
            "hidden_channels": 96,
            "num_layers": 2,
            "num_basis": 24,
        },
    },
    {
        "index": 6,
        "model_name": "schnet",
        "tag": "schnet_compact",
        "initial_batch_size": 4,
        "max_batch_size": 128,
        "model_kwargs": {
            "hidden_channels": 96,
            "num_filters": 96,
            "num_interactions": 3,
            "num_gaussians": 40,
        },
    },
    {
        "index": 7,
        "model_name": "mace",
        "tag": "mace_compact",
        "initial_batch_size": 2,
        "max_batch_size": 48,
        "model_kwargs": {
            "num_layers": 2,
            "num_basis": 16,
            "hidden_scalar_dim": 64,
            "hidden_vector_dim": 32,
        },
    },
]


SCRIPT_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"

ENV_NAME="${{ENV_NAME:-horm}}"
PYTHON_VERSION="${{PYTHON_VERSION:-3.11}}"
TORCH_VERSION="${{TORCH_VERSION:-2.2.1}}"
TORCH_INDEX_URL="${{TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}}"
PYG_WHEEL_URL="${{PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.2.1+cu121.html}}"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
WORK_DIR="${{PROJECT_ROOT}}"
DATA_ROOT="${{DATA_ROOT:-${{PROJECT_ROOT}}/data/oc20}}"
PROCESSED_ROOT="${{PROCESSED_ROOT:-${{DATA_ROOT}}/processed_lmdb}}"
OUTPUT_ROOT="${{OUTPUT_ROOT:-${{PROJECT_ROOT}}/el-mlffs/checkpoints/oc20_base_models}}"
LOG_ROOT="${{LOG_ROOT:-${{PROJECT_ROOT}}/el-mlffs/logs/oc20_base_models}}"

GPU_IDS="${{GPU_IDS:-0,1,2,3}}"
IFS=',' read -r -a GPU_ARRAY <<< "${{GPU_IDS}}"
NPROC_PER_NODE="${{NPROC_PER_NODE:-${{#GPU_ARRAY[@]}}}}"
MASTER_PORT="${{MASTER_PORT:-29631}}"

TARGET_TOTAL_STEPS="${{TARGET_TOTAL_STEPS:-20000}}"
CUTOFF="${{CUTOFF:-6.0}}"
NUM_WORKERS="${{NUM_WORKERS:-8}}"
DATA_PREP_WORKERS="${{DATA_PREP_WORKERS:-8}}"
STORE_EDGES="${{STORE_EDGES:-0}}"
REF_ENERGY="${{REF_ENERGY:-1}}"
TARGET_MEMORY_FRACTION="${{TARGET_MEMORY_FRACTION:-0.78}}"
MEMORY_HEADROOM_FRACTION="${{MEMORY_HEADROOM_FRACTION:-0.05}}"

mkdir -p "${{OUTPUT_ROOT}}" "${{LOG_ROOT}}" "${{PROCESSED_ROOT}}"

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba not found in PATH" >&2
  exit 1
fi

export MAMBA_ROOT_PREFIX="${{MAMBA_ROOT_PREFIX:-$(dirname "$(dirname "$(command -v mamba)")")}}"
set +u
eval "$(mamba shell hook --shell bash)"
set -u

if mamba env list | awk '{{print $1}}' | grep -qx "${{ENV_NAME}}"; then
  echo "Detected existing mamba env: ${{ENV_NAME}}"
else
  echo "Creating mamba env: ${{ENV_NAME}}"
  mamba create -y -n "${{ENV_NAME}}" "python=${{PYTHON_VERSION}}"
fi

set +u
mamba activate "${{ENV_NAME}}"
set -u

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${{TORCH_INDEX_URL}}" "torch==${{TORCH_VERSION}}"
python -m pip install numpy==1.26.4 ase e3nn matplotlib seaborn tqdm lmdb
python -m pip install torch-cluster -f "${{PYG_WHEEL_URL}}"
python -m pip install torch-scatter -f "${{PYG_WHEEL_URL}}"
python -m pip install torch-sparse -f "${{PYG_WHEEL_URL}}"
python -m pip install torch-geometric

cd "${{WORK_DIR}}"

PREP_FLAGS=()
if [ "${{STORE_EDGES}}" = "1" ]; then
  PREP_FLAGS+=("--store-edges")
fi
if [ "${{REF_ENERGY}}" = "1" ]; then
  PREP_FLAGS+=("--ref-energy")
fi

if [ ! -f "${{PROCESSED_ROOT}}/2M/metadata.json" ] || \
   [ ! -f "${{PROCESSED_ROOT}}/val_id/metadata.json" ] || \
   [ ! -f "${{PROCESSED_ROOT}}/val_ood_ads/metadata.json" ] || \
   [ ! -f "${{PROCESSED_ROOT}}/val_ood_cat/metadata.json" ] || \
   [ ! -f "${{PROCESSED_ROOT}}/val_ood_both/metadata.json" ]; then
  python "${{PROJECT_ROOT}}/scripts/prepare_oc20_s2ef_for_elmlffs.py" \\
    --oc20-root "${{DATA_ROOT}}" \\
    --processed-root "${{PROCESSED_ROOT}}" \\
    --splits 2M val_id val_ood_ads val_ood_cat val_ood_both \\
    --cutoff "${{CUTOFF}}" \\
    --num-workers "${{DATA_PREP_WORKERS}}" \\
    "${{PREP_FLAGS[@]}}"
fi

export CUDA_VISIBLE_DEVICES="${{GPU_IDS}}"

python "${{PROJECT_ROOT}}/train_oc20_base_multigpu.py" \\
  --model-name "{model_name}" \\
  --dataset-backend oc20_lmdb \\
  --data-file "${{PROCESSED_ROOT}}/2M" \\
  --val-data-file "${{PROCESSED_ROOT}}/val_id" \\
  --extra-val-file "ood_ads=${{PROCESSED_ROOT}}/val_ood_ads" \\
  --extra-val-file "ood_cat=${{PROCESSED_ROOT}}/val_ood_cat" \\
  --extra-val-file "ood_both=${{PROCESSED_ROOT}}/val_ood_both" \\
  --cutoff "${{CUTOFF}}" \\
  --lr "{lr}" \\
  --min-lr "{min_lr}" \\
  --energy-weight "{energy_weight}" \\
  --force-weight "{force_weight}" \\
  --initial-batch-size "{initial_batch_size}" \\
  --max-batch-size "{max_batch_size}" \\
  --target-memory-fraction "${{TARGET_MEMORY_FRACTION}}" \\
  --memory-headroom-fraction "${{MEMORY_HEADROOM_FRACTION}}" \\
  --target-total-steps "${{TARGET_TOTAL_STEPS}}" \\
  --nproc-per-node "${{NPROC_PER_NODE}}" \\
  --master-port "${{MASTER_PORT}}" \\
  --num-workers "${{NUM_WORKERS}}" \\
  --output-dir "${{OUTPUT_ROOT}}/{model_name}" \\
  --save-name "{tag}.pth" \\
{model_kwarg_lines}
  2>&1 | tee "${{LOG_ROOT}}/{index}_{tag}.log"
"""


def make_model_kwarg_lines(model_kwargs: dict[str, object]) -> str:
    return "".join([f'  --model-kwarg "{key}={value}" \\\n' for key, value in model_kwargs.items()])


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SCRIPT_DIR / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle:
        writer = csv.DictWriter(
            manifest_handle,
            fieldnames=["index", "model_name", "tag", "initial_batch_size", "max_batch_size", "model_kwargs"],
        )
        writer.writeheader()
        for spec in MODEL_SPECS:
            script_path = SCRIPT_DIR / f"{spec['index']}.sh"
            script_text = SCRIPT_TEMPLATE.format(
                index=spec["index"],
                model_name=spec["model_name"],
                tag=spec["tag"],
                lr="3e-4",
                min_lr="1e-6",
                energy_weight="1.0",
                force_weight="50.0",
                initial_batch_size=spec["initial_batch_size"],
                max_batch_size=spec["max_batch_size"],
                model_kwarg_lines=make_model_kwarg_lines(spec["model_kwargs"]),
            )
            script_path.write_text(script_text, encoding="utf-8")
            script_path.chmod(0o755)
            writer.writerow(
                {
                    "index": spec["index"],
                    "model_name": spec["model_name"],
                    "tag": spec["tag"],
                    "initial_batch_size": spec["initial_batch_size"],
                    "max_batch_size": spec["max_batch_size"],
                    "model_kwargs": json_like(spec["model_kwargs"]),
                }
            )


def json_like(mapping: dict[str, object]) -> str:
    return ";".join(f"{key}={value}" for key, value in mapping.items())


if __name__ == "__main__":
    main()
