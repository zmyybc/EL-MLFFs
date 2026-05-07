from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

ROOT_DIR = Path(__file__).resolve().parent
MODULE_DIR = ROOT_DIR / "el-mlffs"

import sys

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from torch_base_models import BASE_MODEL_REGISTRY
from torch_data import ExtXYZDataset


BASE_MODELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "el-mlffs" / "checkpoints" / "base_models_a100_8gpu_50ksteps"


class LearnableWeightedAverage(nn.Module):
    """Learnable weighted average of base model force predictions."""

    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_models))

    def forward(self, forces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            forces: (num_models, num_atoms, 3)
        Returns:
            weighted_forces: (num_atoms, 3)
        """
        weights = torch.softmax(self.weights, dim=0)
        return (forces * weights.view(-1, 1, 1)).sum(dim=0)


def iter_model_combinations() -> list[tuple[str, ...]]:
    combinations: list[tuple[str, ...]] = []
    for size in range(1, len(BASE_MODELS) + 1):
        combinations.extend(itertools.combinations(BASE_MODELS, size))
    return combinations


def load_base_models(model_names: tuple[str, ...], checkpoint_dir: Path, cutoff: float, device: torch.device, all_z: list[int]):
    models: dict[str, torch.nn.Module] = {}
    for model_name in model_names:
        checkpoint_path = checkpoint_dir / f"{model_name}_torch.pth"
        model_cls = BASE_MODEL_REGISTRY[model_name]
        if model_name in {"dp", "nep", "mtp", "soap"}:
            model = model_cls(z_list=all_z, cutoff=cutoff).to(device)
        else:
            model = model_cls(cutoff=cutoff).to(device)
        if checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location=device)
            state_dict = payload.get("state_dict", payload)
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        models[model_name] = model
    return models


def collect_base_predictions(models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device):
    """Collect all base model predictions for the entire dataset."""
    all_forces: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_graph_indices: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        forces_list = []
        for model in models.values():
            with torch.no_grad():
                outputs = model(batch, compute_forces=True, create_graph=False)
            forces_list.append(outputs["forces"])
        # (num_models, num_atoms, 3)
        stacked = torch.stack(forces_list, dim=0)
        all_forces.append(stacked.cpu())
        all_targets.append(batch.forces.cpu())
        all_graph_indices.append(batch.batch.cpu() if hasattr(batch, "batch") else torch.zeros(batch.forces.size(0), dtype=torch.long))
    return torch.cat(all_forces, dim=1), torch.cat(all_targets, dim=0), torch.cat(all_graph_indices, dim=0)


def train_lwa(forces: torch.Tensor, targets: torch.Tensor, graph_indices: torch.Tensor, epochs: int = 500, lr: float = 0.1) -> LearnableWeightedAverage:
    """Train LWA on collected predictions."""
    num_models = forces.size(0)
    lwa = LearnableWeightedAverage(num_models)
    optimizer = torch.optim.Adam(lwa.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = lwa(forces)
        loss = torch.abs(pred - targets).mean()
        loss.backward()
        optimizer.step()

    return lwa


def evaluate_lwa(lwa: LearnableWeightedAverage, forces: torch.Tensor, targets: torch.Tensor, graph_indices: torch.Tensor) -> float:
    with torch.no_grad():
        pred = lwa(forces)
        mae = torch.abs(pred - targets).mean().item()
    return mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LWA (Learnable Weighted Average) baseline.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--train-data-file", default="el-mlffs/data/train.extxyz")
    parser.add_argument("--val-data-file", default="el-mlffs/data/test.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "reports" / "lwa_baseline_metrics.csv")
    parser.add_argument("--lwa-epochs", type=int, default=500)
    parser.add_argument("--lwa-lr", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_dataset = ExtXYZDataset(args.train_data_file, cutoff=args.cutoff)
    val_dataset = ExtXYZDataset(args.val_data_file, cutoff=args.cutoff)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_z = sorted({int(z) for data in train_dataset for z in data.z})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index, combo in enumerate(iter_model_combinations(), start=1):
        print(f"Evaluating LWA baseline {index}/127: {combo}")
        models = load_base_models(combo, args.checkpoint_dir, args.cutoff, device, all_z)

        # Collect predictions
        train_forces, train_targets, train_graph_idx = collect_base_predictions(models, train_loader, device)
        val_forces, val_targets, val_graph_idx = collect_base_predictions(models, val_loader, device)

        # Train LWA
        lwa = train_lwa(train_forces, train_targets, train_graph_idx, epochs=args.lwa_epochs, lr=args.lwa_lr)

        # Evaluate
        val_force_mae = evaluate_lwa(lwa, val_forces, val_targets, val_graph_idx)

        # Also compute energy LWA (same weights)
        with torch.no_grad():
            weights = torch.softmax(lwa.weights, dim=0).cpu()
        print(f"  Learned weights: {weights.numpy().round(4)}")

        rows.append({
            "task_index": index,
            "combo_tag": "_".join(combo),
            "num_models": len(combo),
            "base_model_names": " ".join(combo),
            "val_force_mae": val_force_mae,
        })

        del models, lwa
        torch.cuda.empty_cache()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_index", "combo_tag", "num_models", "base_model_names", "val_force_mae"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved LWA baseline metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
