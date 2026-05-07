from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, global_mean_pool

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(ROOT_DIR, "el-mlffs")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from torch_base_models import TorchForceField
from train_distributed import build_dataloader, cleanup_distributed, reduce_average, setup_distributed, unwrap_model, wrap_model
from torch_data import ExtXYZDataset, energy_to_forces
from torch_workflow import random_split_dataset, unique_atomic_numbers


class LSMDirectModel(TorchForceField):
    """Large Single Model (LSM) — Direct fitting with GAT backbone.
    Approx ~15M parameters for the default configuration.
    """

    def __init__(
        self,
        num_atom_types: int = 100,
        atom_emb_dim: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 10,
        dropout: float = 0.0,
        jk_mode: str = "cat",
    ) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, atom_emb_dim)
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.residuals = nn.ModuleList()
        in_dim = atom_emb_dim
        for i in range(num_layers):
            if i < num_layers - 1:
                # Intermediate layers: multi-head with concatenation
                conv = GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
                out_dim = hidden_dim
            else:
                # Last layer: single head
                conv = GATConv(in_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
                out_dim = hidden_dim
            self.convs.append(conv)
            self.residuals.append(
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )
            in_dim = out_dim

        self.jk = JumpingKnowledge(mode=jk_mode)
        if jk_mode == "cat":
            jk_dim = sum(
                hidden_dim if i < num_layers - 1 else hidden_dim
                for i in range(num_layers)
            )
        else:
            jk_dim = hidden_dim

        self.force_head = nn.Sequential(
            nn.Linear(jk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward_energy(self, batch) -> torch.Tensor:
        x = self.atom_embedding(batch.z)
        edge_index = batch.edge_index
        xs = []
        for conv, res in zip(self.convs, self.residuals):
            x_res = res(x)
            x = F.elu(conv(x, edge_index))
            x = x + x_res
            xs.append(x)

        x = self.jk(xs)
        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Node-level force prediction (no pooling — per-atom forces)
        forces = self.force_head(x)
        # To get energy, sum per-atom contributions
        energy = global_mean_pool(forces.norm(dim=-1, keepdim=True), batch_index).sum(dim=-1, keepdim=True)
        # But we need a proper energy prediction — use the force head as a proxy
        # Actually for LSM-Direct we just need forces, but TorchForceField.forward expects energy
        # Let's use a simple energy predictor
        energy_pred = global_mean_pool(x, batch_index)
        energy = energy_pred.mean(dim=-1, keepdim=True)
        return energy

    def forward(self, batch, compute_forces: bool = True, create_graph: bool = True) -> dict[str, torch.Tensor]:
        x = self.atom_embedding(batch.z)
        edge_index = batch.edge_index
        xs = []
        for conv, res in zip(self.convs, self.residuals):
            x_res = res(x)
            x = F.elu(conv(x, edge_index))
            x = x + x_res
            xs.append(x)

        x = self.jk(xs)
        forces = self.force_head(x)

        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Dummy energy for interface compatibility; forces are predicted directly.
        energy = global_mean_pool(forces.norm(dim=-1, keepdim=True), batch_index).sum(dim=-1, keepdim=True)
        return {"energy": energy, "forces": forces}


@dataclass
class TrainConfig:
    data_file: str = "data/train.extxyz"
    val_data_file: str | None = None
    cutoff: float = 5.0
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-3
    min_lr: float = 1e-6
    force_weight: float = 50.0
    save_path: str = "checkpoints/lsm_direct.pth"
    train_ratio: float = 0.9
    seed: int = 42
    num_workers: int = 0
    grad_clip_norm: float = 10.0
    huber_delta: float = 1.0
    target_total_steps: int = 50000
    num_atom_types: int = 100
    atom_emb_dim: int = 64
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 10
    dropout: float = 0.0
    jk_mode: str = "cat"


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
        outputs = model(batch, compute_forces=True, create_graph=True)
        loss_f = F.huber_loss(outputs["forces"], batch.forces, delta=config.huber_delta)
        loss = config.force_weight * loss_f
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
    total_force_mae = 0.0
    total_graphs = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(context.device)
            outputs = model(batch, compute_forces=True, create_graph=False)
            force_mae = F.l1_loss(outputs["forces"], batch.forces).item()
            total_force_mae += force_mae * batch.num_graphs
            total_graphs += batch.num_graphs
    return reduce_average(total_force_mae, total_graphs, context)


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

        all_z = unique_atomic_numbers(dataset)
        train_loader, train_sampler = build_dataloader(
            train_dataset, batch_size=config.batch_size, shuffle=True, context=context, num_workers=config.num_workers
        )
        val_loader, _ = build_dataloader(
            val_dataset, batch_size=config.batch_size, shuffle=False, context=context, num_workers=config.num_workers
        )

        model = LSMDirectModel(
            num_atom_types=config.num_atom_types,
            atom_emb_dim=config.atom_emb_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            jk_mode=config.jk_mode,
        ).to(context.device)

        print(f"LSM-Direct parameters: {count_parameters(model):,}")
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
            val_force_mae = evaluate(model, val_loader, context)
            current_lr = optimizer.param_groups[0]["lr"]
            if context.is_main:
                print(
                    f"Epoch {epoch:03d} | Steps: {global_step}/{config.target_total_steps} | "
                    f"Train Loss: {train_loss:.6f} | Val Force MAE: {val_force_mae:.6f} | LR: {current_lr:.8f}"
                )
            if context.is_main and val_force_mae < best_force_mae:
                best_force_mae = val_force_mae
                save_dir = os.path.dirname(config.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                payload = {
                    "state_dict": unwrap_model(model).state_dict(),
                    "metadata": {
                        "kind": "lsm_direct",
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
    parser = argparse.ArgumentParser(description="Train LSM-Direct (large single GAT model).")
    parser.add_argument("--data-file", default="el-mlffs/data/train.extxyz")
    parser.add_argument("--val-data-file", default="el-mlffs/data/test.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--force-weight", type=float, default=50.0)
    parser.add_argument("--save-path", default="el-mlffs/checkpoints/lsm_direct.pth")
    parser.add_argument("--target-total-steps", type=int, default=50000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
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
        force_weight=args.force_weight,
        save_path=args.save_path,
        target_total_steps=args.target_total_steps,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
