from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(ROOT_DIR, "el-mlffs")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from train_distributed import build_dataloader, cleanup_distributed, reduce_average, setup_distributed, unwrap_model, wrap_model
from torch_base_models import BASE_MODEL_REGISTRY
from torch_data import ExtXYZDataset, energy_to_forces, get_batch_index
from torch_ensemble_models import BaseModelStack
from torch_workflow import random_split_dataset, unique_atomic_numbers


class LSMMACEMetaEncoder(nn.Module):
    """Large MACE-style encoder for LSM-Conserv. ~15M parameters."""

    def __init__(
        self,
        scalar_input_dim: int,
        vector_input_channels: int,
        hidden_scalar_channels: int = 256,
        hidden_vector_channels: int = 128,
        num_layers: int = 6,
        num_basis: int = 64,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        try:
            from e3nn import o3
        except ImportError as exc:
            raise ImportError("LSMMACEMetaEncoder requires e3nn to be installed.") from exc

        self.o3 = o3
        self.hidden_scalar_channels = hidden_scalar_channels
        self.hidden_vector_channels = hidden_vector_channels
        self.cutoff = cutoff

        self.irreps_in = o3.Irreps(f"{scalar_input_dim}x0e + {vector_input_channels}x1o")
        self.irreps_hidden = o3.Irreps(f"{hidden_scalar_channels}x0e + {hidden_vector_channels}x1o")
        self.irreps_sh = o3.Irreps("1x0e + 1x1o + 1x2e")

        from torch_base_models import GaussianBasis, MACETensorInteraction, SmoothBumpEnvelope

        self.rbf = GaussianBasis(num_basis=num_basis, cutoff=cutoff)
        self.envelope = SmoothBumpEnvelope(cutoff=cutoff)
        self.interactions = nn.ModuleList(
            [
                MACETensorInteraction(
                    irreps_in=str(self.irreps_in if layer_idx == 0 else self.irreps_hidden),
                    irreps_out=str(self.irreps_hidden),
                    num_basis=num_basis,
                    irreps_sh=str(self.irreps_sh),
                )
                for layer_idx in range(num_layers)
            ]
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_scalar_channels

    def forward(self, batch, scalar_features, vector_features):
        from torch_data import compute_edge_geometry

        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)

        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        sh = self.o3.spherical_harmonics(self.irreps_sh, direction, normalize=True, normalization="component")
        edge_attr = self.rbf(edge_length) * self.envelope(edge_length).unsqueeze(-1)

        scalar_features = torch.nan_to_num(scalar_features)
        vector_features = torch.nan_to_num(vector_features)
        x = torch.cat([scalar_features, vector_features.reshape(vector_features.size(0), -1)], dim=-1)
        for interaction in self.interactions:
            x = interaction(x, batch.edge_index, edge_attr, sh)
            x = torch.nan_to_num(x)

        scalar_out = x[:, : self.hidden_scalar_channels]
        vector_out = x[:, self.hidden_scalar_channels :].reshape(-1, self.hidden_vector_channels, 3)
        return {
            "node_features": x,
            "scalar_features": scalar_out,
            "vector_features": vector_out,
        }


class LSMConservativeEnergyMixer(nn.Module):
    """LSM-Conserv: large single conservative model using MACE-style encoder.
    Input is atomic species + geometry (no base model predictions).
    ~15M parameters."""

    def __init__(
        self,
        atom_emb_dim: int = 64,
        hidden_scalar_channels: int = 256,
        hidden_vector_channels: int = 128,
        num_layers: int = 6,
        num_basis: int = 64,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(100, atom_emb_dim)
        self.encoder = LSMMACEMetaEncoder(
            scalar_input_dim=atom_emb_dim,
            vector_input_channels=1,
            hidden_scalar_channels=hidden_scalar_channels,
            hidden_vector_channels=hidden_vector_channels,
            num_layers=num_layers,
            num_basis=num_basis,
            cutoff=cutoff,
        )
        self.atomic_correction = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        if not batch.pos.requires_grad:
            batch.pos = batch.pos.clone().detach().requires_grad_(True)

        atom_emb = self.atom_embedding(batch.z)
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))

        # Dummy vector input (just position direction, no base forces)
        vector_features = torch.zeros(batch.pos.size(0), 1, 3, device=batch.pos.device)
        encoded = self.encoder(batch, atom_emb, vector_features)
        scalar_node_features = encoded["scalar_features"]
        graph_features = global_mean_pool(scalar_node_features, batch_index)

        mlp_energy = self.energy_mlp(graph_features)
        correction_energy = global_add_pool(self.atomic_correction(scalar_node_features), batch_index)
        total_energy = torch.nan_to_num(mlp_energy + correction_energy)

        return {
            "energy": total_energy,
            "forces": energy_to_forces(total_energy, batch.pos, create_graph=self.training),
        }


@dataclass
class TrainConfig:
    data_file: str = "el-mlffs/data/train.extxyz"
    val_data_file: str = "el-mlffs/data/test.extxyz"
    cutoff: float = 5.0
    batch_size: int = 4
    lr: float = 1e-3
    min_lr: float = 1e-6
    energy_weight: float = 1.0
    force_weight: float = 50.0
    save_path: str = "el-mlffs/checkpoints/lsm_conserv.pth"
    train_ratio: float = 0.9
    seed: int = 42
    num_workers: int = 0
    grad_clip_norm: float = 10.0
    huber_delta: float = 1.0
    target_total_steps: int = 50000
    hidden_scalar_channels: int = 256
    hidden_vector_channels: int = 128
    num_layers: int = 6
    num_basis: int = 64


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, config, context, scheduler=None, max_steps=None):
    model.train()
    total_loss = 0.0
    total_graphs = 0.0
    steps = 0
    for batch_idx, batch in enumerate(loader):
        if max_steps is not None and steps >= max_steps:
            break
        batch = batch.to(context.device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss_e = F.huber_loss(outputs["energy"].view(-1), batch.energy.view(-1), delta=config.huber_delta)
        loss_f = F.huber_loss(outputs["forces"], batch.forces, delta=config.huber_delta)
        loss = config.energy_weight * loss_e + config.force_weight * loss_f
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
        steps += 1
    return reduce_average(total_loss, total_graphs, context), steps


def evaluate(model, loader, context):
    model.eval()
    total_energy_mae = 0.0
    total_force_mae = 0.0
    total_graphs = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(context.device)
            outputs = model(batch)
            energy_mae = F.l1_loss(outputs["energy"].view(-1), batch.energy.view(-1)).item()
            force_mae = F.l1_loss(outputs["forces"], batch.forces).item()
            total_energy_mae += energy_mae * batch.num_graphs
            total_force_mae += force_mae * batch.num_graphs
            total_graphs += batch.num_graphs
    return reduce_average(total_energy_mae, total_graphs, context), reduce_average(total_force_mae, total_graphs, context)


def run_training(config: TrainConfig):
    context = setup_distributed()
    try:
        if context.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        dataset = ExtXYZDataset(config.data_file, cutoff=config.cutoff)
        if config.val_data_file:
            train_dataset = dataset
            val_dataset = ExtXYZDataset(config.val_data_file, cutoff=config.cutoff)
        else:
            train_dataset, val_dataset = random_split_dataset(dataset, train_ratio=config.train_ratio, seed=config.seed)

        train_loader, train_sampler = build_dataloader(
            train_dataset, batch_size=config.batch_size, shuffle=True, context=context, num_workers=config.num_workers
        )
        val_loader, _ = build_dataloader(
            val_dataset, batch_size=config.batch_size, shuffle=False, context=context, num_workers=config.num_workers
        )

        model = LSMConservativeEnergyMixer(
            hidden_scalar_channels=config.hidden_scalar_channels,
            hidden_vector_channels=config.hidden_vector_channels,
            num_layers=config.num_layers,
            num_basis=config.num_basis,
            cutoff=config.cutoff,
        ).to(context.device)

        print(f"LSM-Conserv parameters: {count_parameters(model):,}")
        model = wrap_model(model, context)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.target_total_steps, eta_min=config.min_lr
        )

        best_force_mae = float("inf")
        epoch = 0
        global_step = 0
        while global_step < config.target_total_steps:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            max_steps = config.target_total_steps - global_step
            train_loss, steps_this_epoch = train_one_epoch(
                model, train_loader, optimizer, config, context, scheduler=scheduler, max_steps=max_steps
            )
            global_step += steps_this_epoch
            val_energy_mae, val_force_mae = evaluate(model, val_loader, context)
            current_lr = optimizer.param_groups[0]["lr"]
            if context.is_main:
                print(
                    f"Epoch {epoch:03d} | Steps: {global_step}/{config.target_total_steps} | "
                    f"Train Loss: {train_loss:.6f} | Val Energy MAE: {val_energy_mae:.6f} | "
                    f"Val Force MAE: {val_force_mae:.6f} | LR: {current_lr:.8f}"
                )
            if context.is_main and val_force_mae < best_force_mae:
                best_force_mae = val_force_mae
                save_dir = os.path.dirname(config.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                payload = {
                    "state_dict": unwrap_model(model).state_dict(),
                    "metadata": {
                        "kind": "lsm_conserv",
                        "val_energy_mae": float(val_energy_mae),
                        "val_force_mae": float(val_force_mae),
                        "num_parameters": count_parameters(unwrap_model(model)),
                    },
                    "config": asdict(config),
                }
                torch.save(payload, config.save_path)
        return config.save_path
    finally:
        cleanup_distributed(context)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LSM-Conserv (large single conservative model).")
    parser.add_argument("--data-file", default="el-mlffs/data/train.extxyz")
    parser.add_argument("--val-data-file", default="el-mlffs/data/test.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--save-path", default="el-mlffs/checkpoints/lsm_conserv.pth")
    parser.add_argument("--target-total-steps", type=int, default=50000)
    parser.add_argument("--hidden-scalar-channels", type=int, default=256)
    parser.add_argument("--hidden-vector-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-basis", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    args = parser.parse_args()
    return TrainConfig(
        data_file=args.data_file,
        val_data_file=args.val_data_file,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        save_path=args.save_path,
        target_total_steps=args.target_total_steps,
        hidden_scalar_channels=args.hidden_scalar_channels,
        hidden_vector_channels=args.hidden_vector_channels,
        num_layers=args.num_layers,
        num_basis=args.num_basis,
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
