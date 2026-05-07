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


class MLPForcePredictor(nn.Module):
    """MLP meta-model: non-linear combination of base model predictions per atom."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.SiLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LNCBaseline(nn.Module):
    """Learnable Non-linear Combination (LNC) baseline."""

    def __init__(self, num_models: int, atom_emb_dim: int = 16, hidden_dims: tuple[int, ...] = (256, 256, 128)) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(100, atom_emb_dim)
        self.mlp = MLPForcePredictor(num_models * 3 + atom_emb_dim, hidden_dims)

    def forward(self, z: torch.Tensor, base_forces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (num_atoms,) atomic numbers
            base_forces: (num_models, num_atoms, 3)
        Returns:
            forces: (num_atoms, 3)
        """
        atom_emb = self.atom_embedding(z)
        # (num_models, num_atoms, 3) -> (num_atoms, num_models * 3)
        forces_flat = base_forces.permute(1, 0, 2).reshape(base_forces.size(1), -1)
        x = torch.cat([atom_emb, forces_flat], dim=-1)
        return self.mlp(x)


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


def train_lnc(
    base_models: dict[str, torch.nn.Module],
    lnc: LNCBaseline,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    force_weight: float = 50.0,
    grad_clip_norm: float = 10.0,
) -> LNCBaseline:
    optimizer = torch.optim.Adam(lnc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        lnc.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            forces_list = []
            for model in base_models.values():
                with torch.no_grad():
                    outputs = model(batch, compute_forces=True, create_graph=False)
                forces_list.append(outputs["forces"])
            base_forces = torch.stack(forces_list, dim=0)

            pred_forces = lnc(batch.z, base_forces)
            loss = torch.nn.functional.huber_loss(pred_forces, batch.forces, delta=1.0)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(lnc.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  LNC Epoch {epoch:03d} | Loss: {total_loss / max(total_graphs, 1):.6f}")

    return lnc


def evaluate_lnc(lnc: LNCBaseline, base_models: dict[str, torch.nn.Module], loader: DataLoader, device: torch.device) -> float:
    lnc.eval()
    total_force_mae = 0.0
    total_graphs = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            forces_list = []
            for model in base_models.values():
                outputs = model(batch, compute_forces=True, create_graph=False)
                forces_list.append(outputs["forces"])
            base_forces = torch.stack(forces_list, dim=0)
            pred_forces = lnc(batch.z, base_forces)
            force_mae = torch.abs(pred_forces - batch.forces).mean().item()
            total_force_mae += force_mae * batch.num_graphs
            total_graphs += batch.num_graphs
    return total_force_mae / total_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LNC (MLP meta-model) baseline.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--train-data-file", default="el-mlffs/data/train.extxyz")
    parser.add_argument("--val-data-file", default="el-mlffs/data/test.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "reports" / "lnc_baseline_metrics.csv")
    parser.add_argument("--lnc-epochs", type=int, default=100)
    parser.add_argument("--lnc-lr", type=float, default=1e-3)
    parser.add_argument("--lnc-hidden-dims", nargs="+", type=int, default=[256, 256, 128])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_dataset = ExtXYZDataset(args.train_data_file, cutoff=args.cutoff)
    val_dataset = ExtXYZDataset(args.val_data_file, cutoff=args.cutoff)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_z = sorted({int(z) for data in train_dataset for z in data.z})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index, combo in enumerate(iter_model_combinations(), start=1):
        print(f"Evaluating LNC baseline {index}/127: {combo}")
        base_models = load_base_models(combo, args.checkpoint_dir, args.cutoff, device, all_z)

        lnc = LNCBaseline(
            num_models=len(combo),
            hidden_dims=tuple(args.lnc_hidden_dims),
        ).to(device)

        lnc = train_lnc(
            base_models,
            lnc,
            train_loader,
            device,
            epochs=args.lnc_epochs,
            lr=args.lnc_lr,
        )

        val_force_mae = evaluate_lnc(lnc, base_models, val_loader, device)
        print(f"  Val Force MAE: {val_force_mae:.6f}")

        rows.append({
            "task_index": index,
            "combo_tag": "_".join(combo),
            "num_models": len(combo),
            "base_model_names": " ".join(combo),
            "val_force_mae": val_force_mae,
        })

        del base_models, lnc
        torch.cuda.empty_cache()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_index", "combo_tag", "num_models", "base_model_names", "val_force_mae"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved LNC baseline metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
