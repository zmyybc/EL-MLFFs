from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from eval_torch_ensemble import EvalConfig, build_eval_loader, load_model, resolve_eval_config
from train_torch_ensemble import load_checkpoint_bundle
from torch_base_models import BASE_MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate Torch ensemble checkpoints and export CSV results.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-csv", default="output.csv")
    parser.add_argument("--data-file")
    parser.add_argument("--architecture", choices=["direct", "conservative"])
    parser.add_argument("--cutoff", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-strategy", choices=["random", "ood"], default="ood")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--base-models", nargs="+", choices=sorted(BASE_MODEL_REGISTRY.keys()))
    return parser.parse_args()


def list_checkpoint_paths(models_dir: str) -> list[str]:
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        os.path.join(models_dir, filename)
        for filename in os.listdir(models_dir)
        if filename.endswith((".pt", ".pth"))
    )


def compute_rank_correlation(base_forces: list[np.ndarray], pred_forces: list[np.ndarray], true_forces: list[np.ndarray]) -> tuple[float, float]:
    if not base_forces:
        return math.nan, math.nan

    base_np = np.concatenate(base_forces, axis=0)
    pred_np = np.concatenate(pred_forces, axis=0)
    true_np = np.concatenate(true_forces, axis=0)

    ddof = 1 if base_np.shape[1] > 1 else 0
    uncertainty = np.std(base_np, axis=1, ddof=ddof).reshape(-1)
    error = np.abs(pred_np - true_np).reshape(-1)
    finite_mask = np.isfinite(uncertainty) & np.isfinite(error)
    uncertainty = uncertainty[finite_mask]
    error = error[finite_mask]
    if uncertainty.size < 2 or np.allclose(uncertainty, uncertainty[0]):
        return math.nan, math.nan
    return stats.pearsonr(uncertainty, error)[0], stats.spearmanr(uncertainty, error)[0]


def evaluate_checkpoint(path: str, args: argparse.Namespace) -> dict[str, object]:
    checkpoint = load_checkpoint_bundle(path, map_location="cpu")
    config = EvalConfig(
        model_path=path,
        architecture=args.architecture,
        data_file=args.data_file,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        split_strategy=args.split_strategy,
        seed=args.seed,
        train_ratio=args.train_ratio,
        base_model_names=tuple(args.base_models) if args.base_models else None,
    )
    config = resolve_eval_config(config, checkpoint)

    loader, _ = build_eval_loader(config)
    model = load_model(config, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_energy_mae = 0.0
    total_force_mae = 0.0
    base_forces = []
    pred_forces = []
    true_forces = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        if config.architecture == "conservative":
            total_energy_mae += F.l1_loss(outputs["energy"].view(-1), batch.energy.view(-1)).item() * batch.num_graphs
        total_force_mae += F.l1_loss(outputs["forces"], batch.forces).item() * batch.num_graphs
        base_forces.append(outputs["base_forces"].detach().cpu().numpy())
        pred_forces.append(outputs["forces"].detach().cpu().numpy())
        true_forces.append(batch.forces.detach().cpu().numpy())

    num_samples = len(loader.dataset)
    pearson_r, spearman_r = compute_rank_correlation(base_forces, pred_forces, true_forces)
    return {
        "checkpoint": path,
        "architecture": config.architecture,
        "base_models": ",".join(config.base_model_names),
        "cutoff": config.cutoff,
        "split_strategy": config.split_strategy,
        "energy_mae": total_energy_mae / num_samples,
        "force_mae": total_force_mae / num_samples,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    checkpoint_paths = list_checkpoint_paths(args.models_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt or .pth checkpoints found in {args.models_dir}")

    results = []
    for checkpoint_path in checkpoint_paths:
        try:
            result = evaluate_checkpoint(checkpoint_path, args)
            print(
                f"{os.path.basename(checkpoint_path)} | "
                f"{result['architecture']} | Force MAE: {result['force_mae']:.6f}"
            )
        except Exception as exc:  # pragma: no cover - keeps batch scan going on mixed directories
            result = {
                "checkpoint": checkpoint_path,
                "architecture": "",
                "base_models": "",
                "cutoff": "",
                "split_strategy": "",
                "energy_mae": "",
                "force_mae": "",
                "pearson_r": "",
                "spearman_r": "",
                "status": f"skipped: {exc}",
            }
            print(f"Skipping {os.path.basename(checkpoint_path)}: {exc}")
        results.append(result)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "architecture",
                "base_models",
                "cutoff",
                "split_strategy",
                "energy_mae",
                "force_mae",
                "pearson_r",
                "spearman_r",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
