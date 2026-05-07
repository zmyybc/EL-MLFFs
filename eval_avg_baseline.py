from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path

import torch
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


def evaluate_avg_forces(models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device) -> tuple[float, float]:
    total_force_mae = 0.0
    total_energy_mae = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        forces_list = []
        energies_list = []
        for model in models.values():
            outputs = model(batch, compute_forces=True, create_graph=False)
            energies_list.append(outputs["energy"].view(-1))
            forces_list.append(outputs["forces"])
        avg_forces = torch.stack(forces_list, dim=0).mean(dim=0)
        avg_energies = torch.stack(energies_list, dim=0).mean(dim=0)
        force_mae = torch.abs(avg_forces - batch.forces).mean().item()
        energy_mae = torch.abs(avg_energies - batch.energy.view(-1)).mean().item()
        total_force_mae += force_mae * batch.num_graphs
        total_energy_mae += energy_mae * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_energy_mae / total_graphs, total_force_mae / total_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AVG baseline for all ensemble combinations.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--val-data-file", default="el-mlffs/data/test.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "reports" / "avg_baseline_metrics.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = ExtXYZDataset(args.val_data_file, cutoff=args.cutoff)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Infer atomic numbers from dataset
    all_z = sorted({int(z) for data in dataset for z in data.z})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index, combo in enumerate(iter_model_combinations(), start=1):
        print(f"Evaluating AVG baseline {index}/127: {combo}")
        models = load_base_models(combo, args.checkpoint_dir, args.cutoff, device, all_z)
        val_energy_mae, val_force_mae = evaluate_avg_forces(models, loader, device)
        rows.append({
            "task_index": index,
            "combo_tag": "_".join(combo),
            "num_models": len(combo),
            "base_model_names": " ".join(combo),
            "val_energy_mae": val_energy_mae,
            "val_force_mae": val_force_mae,
        })
        del models
        torch.cuda.empty_cache()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_index", "combo_tag", "num_models", "base_model_names", "val_energy_mae", "val_force_mae"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved AVG baseline metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
