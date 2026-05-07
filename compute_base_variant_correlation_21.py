from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


DEFAULT_DELIVERY_ROOT = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a Pearson correlation matrix over force errors for the 21 base-model variants."
    )
    parser.add_argument("--delivery-root", default=str(DEFAULT_DELIVERY_ROOT))
    parser.add_argument("--data-file", help="Dataset path. Defaults to delivery-bundle test.extxyz.")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "base_variant_correlation_21_outputs"),
        help="Directory for CSV/PNG/JSON outputs.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_delivery_modules(delivery_root: Path):
    module_dir = delivery_root / "el-mlffs"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import train_torch_ensemble_diverse as ensemble  # type: ignore
    from torch_data import ExtXYZDataset  # type: ignore
    from torch_workflow import unique_atomic_numbers  # type: ignore

    return ensemble, ExtXYZDataset, unique_atomic_numbers


def existing_variant_layout() -> dict[str, list[str]]:
    # This matches the actual checkpoints present in the delivery bundle,
    # not the stale manifest names.
    return {
        "schnet": ["baseline", "compact", "expressive"],
        "painn": ["baseline", "compact", "expressive"],
        "dp": ["baseline", "compact", "tiny"],
        "nep": ["baseline", "compact", "expressive"],
        "mtp": ["baseline", "compact", "tiny"],
        "soap": ["baseline", "compact", "expressive"],
        "mace": ["baseline", "compact", "expressive"],
    }


def load_variant_specs(delivery_root: Path, ensemble) -> tuple[list[str], list[Path]]:
    labels: list[str] = []
    checkpoint_paths: list[Path] = []
    families = existing_variant_layout()
    for model_name, variants in families.items():
        for variant_name in variants:
            checkpoint_path = (
                delivery_root
                / "el-mlffs"
                / "checkpoints"
                / "base_model_variants"
                / model_name
                / variant_name
                / f"{model_name}_torch.pth"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            alias = f"{model_name}_{variant_name}"
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model_kwargs = dict(ckpt.get("config") or {}).get("model_kwargs") or {}
            ensemble.BASE_MODEL_CONFIGS[alias] = {
                "arch": model_name,
                "checkpoint": str(checkpoint_path),
                **model_kwargs,
            }
            labels.append(alias)
            checkpoint_paths.append(checkpoint_path)
    return labels, checkpoint_paths


def write_matrix_csv(matrix: np.ndarray, labels: list[str], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{value:.8f}" for value in row]])


def write_heatmap(matrix: np.ndarray, labels: list[str], png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.25,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Base Variant Force Error Correlation Matrix", fontsize=16, pad=16, weight="bold")
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    delivery_root = Path(args.delivery_root).resolve()
    if not delivery_root.exists():
        raise FileNotFoundError(f"Delivery root not found: {delivery_root}")

    ensemble, ExtXYZDataset, unique_atomic_numbers = import_delivery_modules(delivery_root)
    device = resolve_device(args.device)
    data_file = Path(args.data_file).resolve() if args.data_file else delivery_root / "el-mlffs" / "data" / "test.extxyz"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, checkpoint_paths = load_variant_specs(delivery_root, ensemble)

    dataset = ExtXYZDataset(str(data_file), cutoff=args.cutoff)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    all_z = unique_atomic_numbers(dataset)
    base_models = ensemble.build_base_models(all_z, args.cutoff, device, labels)
    base_models.freeze()
    base_models.eval()

    per_model_error_values: list[np.ndarray] = []
    total_structures = 0
    total_atoms = 0

    with torch.enable_grad():
        for batch in tqdm(loader, desc="Collecting base-model force errors", unit="batch"):
            batch = batch.to(device)
            outputs = base_models(batch, create_graph=False)
            base_forces = outputs["forces"].detach().cpu().numpy()
            true_forces = batch.forces.detach().cpu().numpy()
            true_forces_flat = true_forces.reshape(-1)
            flattened_predictions = np.transpose(base_forces, (1, 0, 2)).reshape(base_forces.shape[1], -1)
            flattened_errors = flattened_predictions - true_forces_flat[None, :]
            per_model_error_values.append(flattened_errors)
            total_structures += int(batch.num_graphs)
            total_atoms += int(batch.z.shape[0])

    if not per_model_error_values:
        raise RuntimeError("No data processed.")

    force_error_values = np.concatenate(per_model_error_values, axis=1)
    corr = np.corrcoef(force_error_values)

    csv_path = output_dir / "base_variant_force_error_pearson_matrix.csv"
    png_path = output_dir / "base_variant_force_error_pearson_matrix.png"
    json_path = output_dir / "summary.json"
    write_matrix_csv(corr, labels, csv_path)
    write_heatmap(corr, labels, png_path)

    pairs: list[tuple[float, str, str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((float(corr[i, j]), labels[i], labels[j]))
    pairs.sort()

    summary = {
        "delivery_root": str(delivery_root),
        "data_file": str(data_file),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_models": len(labels),
        "labels": labels,
        "checkpoints": [str(p) for p in checkpoint_paths],
        "num_structures": total_structures,
        "num_atoms_total": total_atoms,
        "num_force_components": int(force_error_values.shape[1]),
        "lowest_pairs": [{"corr": c, "a": a, "b": b} for c, a, b in pairs[:20]],
        "highest_pairs": [{"corr": c, "a": a, "b": b} for c, a, b in pairs[-20:][::-1]],
        "csv_path": str(csv_path),
        "png_path": str(png_path),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Processed structures: {total_structures}")
    print(f"Processed atoms: {total_atoms}")
    print(f"Force components per model: {force_error_values.shape[1]}")
    print(f"CSV saved to: {csv_path}")
    print(f"Heatmap saved to: {png_path}")
    print("Lowest-correlation pairs:")
    for corr_value, a, b in pairs[:15]:
        print(f"  {a:20s} {b:20s} {corr_value:.6f}")
    print("Highest-correlation pairs:")
    for corr_value, a, b in pairs[-15:][::-1]:
        print(f"  {a:20s} {b:20s} {corr_value:.6f}")


if __name__ == "__main__":
    main()
