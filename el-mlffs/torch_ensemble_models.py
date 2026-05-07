from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from torch_base_models import GaussianBasis, MACETensorInteraction, SmoothBumpEnvelope
from torch_data import compute_edge_geometry, energy_to_forces, get_batch_index


class BaseModelStack(nn.Module):
    def __init__(self, models: dict[str, nn.Module]) -> None:
        super().__init__()
        self.model_names = list(models.keys())
        self.models = nn.ModuleList(list(models.values()))

    @property
    def num_models(self) -> int:
        return len(self.models)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch, create_graph: bool) -> dict[str, torch.Tensor]:
        energies = []
        forces = []
        for model in self.models:
            outputs = model(batch, compute_forces=True, create_graph=create_graph)
            energies.append(outputs["energy"].view(-1))
            forces.append(outputs["forces"])
        return {
            "energies": torch.stack(energies, dim=-1),
            "forces": torch.stack(forces, dim=1),
        }


class MACEMetaEncoder(nn.Module):
    def __init__(
        self,
        scalar_input_dim: int,
        vector_input_channels: int,
        hidden_scalar_channels: int = 64,
        hidden_vector_channels: int = 32,
        num_layers: int = 3,
        num_basis: int = 32,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        try:
            from e3nn import o3
        except ImportError as exc:
            raise ImportError("MACEMetaEncoder requires e3nn to be installed.") from exc

        self.o3 = o3
        self.scalar_input_dim = scalar_input_dim
        self.vector_input_channels = vector_input_channels
        self.hidden_scalar_channels = hidden_scalar_channels
        self.hidden_vector_channels = hidden_vector_channels
        self.cutoff = cutoff

        self.irreps_in = o3.Irreps(f"{scalar_input_dim}x0e + {vector_input_channels}x1o")
        self.irreps_hidden = o3.Irreps(f"{hidden_scalar_channels}x0e + {hidden_vector_channels}x1o")
        self.irreps_sh = o3.Irreps("1x0e + 1x1o + 1x2e")

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

    def forward(
        self,
        batch,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)

        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        sh = self.o3.spherical_harmonics(self.irreps_sh, direction, normalize=True, normalization="component")
        edge_gate = self.envelope(edge_length)
        edge_attr = self.rbf(edge_length) * edge_gate.unsqueeze(-1)

        scalar_features = torch.nan_to_num(scalar_features)
        vector_features = torch.nan_to_num(vector_features)
        x = torch.cat([scalar_features, vector_features.reshape(vector_features.size(0), -1)], dim=-1)
        for interaction in self.interactions:
            x = interaction(x, batch.edge_index, edge_attr, sh, edge_gate)
            x = torch.nan_to_num(x)

        scalar_out = x[:, : self.hidden_scalar_channels]
        vector_out = x[:, self.hidden_scalar_channels :].reshape(-1, self.hidden_vector_channels, 3)
        return {
            "node_features": x,
            "scalar_features": scalar_out,
            "vector_features": vector_out,
        }


class DirectForceFittingEnsemble(nn.Module):
    def __init__(
        self,
        base_models: BaseModelStack,
        atom_emb_dim: int = 16,
        hidden_scalar_channels: int = 64,
        hidden_vector_channels: int = 32,
        num_layers: int = 3,
        num_basis: int = 32,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        try:
            from e3nn import o3
        except ImportError as exc:
            raise ImportError("DirectForceFittingEnsemble requires e3nn to be installed.") from exc

        self.base_models = base_models
        self.o3 = o3
        self.atom_embedding = nn.Embedding(100, atom_emb_dim)
        scalar_input_dim = atom_emb_dim + base_models.num_models + 2
        vector_input_channels = base_models.num_models + 1
        self.encoder = MACEMetaEncoder(
            scalar_input_dim=scalar_input_dim,
            vector_input_channels=vector_input_channels,
            hidden_scalar_channels=hidden_scalar_channels,
            hidden_vector_channels=hidden_vector_channels,
            num_layers=num_layers,
            num_basis=num_basis,
            cutoff=cutoff,
        )
        self.force_head = o3.Linear(str(self.encoder.irreps_hidden), "1x1o")

    def forward(self, batch, base_predictions: Optional[dict[str, torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        if base_predictions is None:
            base_predictions = self.base_models(batch, create_graph=False)

        base_forces = base_predictions["forces"]
        mean_force = base_forces.mean(dim=1)
        force_norm = base_forces.norm(dim=-1)
        force_norm_mean = force_norm.mean(dim=1, keepdim=True)
        force_norm_std = force_norm.std(dim=1, unbiased=False, keepdim=True)

        scalar_features = torch.cat(
            [
                self.atom_embedding(batch.z),
                force_norm,
                force_norm_mean,
                force_norm_std,
            ],
            dim=-1,
        )
        vector_features = torch.cat([base_forces, mean_force.unsqueeze(1)], dim=1)
        encoded = self.encoder(batch, scalar_features, vector_features)
        node_repr = encoded["node_features"]
        return {
            "forces": self.force_head(node_repr),
            "base_energies": base_predictions["energies"],
            "base_forces": base_forces,
        }


class ConservativeEnergyMixer(nn.Module):
    def __init__(
        self,
        base_models: BaseModelStack,
        atom_emb_dim: int = 32,
        hidden_scalar_channels: int = 64,
        hidden_vector_channels: int = 32,
        num_layers: int = 3,
        num_basis: int = 32,
        cutoff: float = 5.0,
        differentiate_force_features: bool = False,
    ) -> None:
        super().__init__()
        self.base_models = base_models
        self.differentiate_force_features = differentiate_force_features
        self.atom_embedding = nn.Embedding(100, atom_emb_dim)
        scalar_input_dim = atom_emb_dim + base_models.num_models + 2
        vector_input_channels = base_models.num_models + 1
        self.encoder = MACEMetaEncoder(
            scalar_input_dim=scalar_input_dim,
            vector_input_channels=vector_input_channels,
            hidden_scalar_channels=hidden_scalar_channels,
            hidden_vector_channels=hidden_vector_channels,
            num_layers=num_layers,
            num_basis=num_basis,
            cutoff=cutoff,
        )
        self.atomic_correction = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.graph_gate = nn.Sequential(
            nn.Linear(self.encoder.output_dim + base_models.num_models, 128),
            nn.SiLU(),
            nn.Linear(128, base_models.num_models),
        )

    def forward(self, batch, base_predictions: Optional[dict[str, torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        if not batch.pos.requires_grad:
            batch.pos = batch.pos.clone().detach().requires_grad_(True)

        if base_predictions is None:
            base_predictions = self.base_models(batch, create_graph=True)

        base_energies = torch.nan_to_num(base_predictions["energies"])
        base_forces = torch.nan_to_num(base_predictions["forces"])
        if not self.differentiate_force_features:
            # Fast path used for training/standard evaluation. For strict energy
            # conservation tests, enable differentiate_force_features so forces
            # include the full VJP through force-valued base-model features.
            base_forces = base_forces.detach()
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))

        mean_force = base_forces.mean(dim=1)
        force_norm = base_forces.norm(dim=-1)
        force_norm_mean = force_norm.mean(dim=1, keepdim=True)
        force_norm_std = force_norm.std(dim=1, unbiased=False, keepdim=True)

        x = torch.cat(
            [
                self.atom_embedding(batch.z),
                force_norm,
                force_norm_mean,
                force_norm_std,
            ],
            dim=-1,
        )
        vector_features = torch.cat([base_forces, mean_force.unsqueeze(1)], dim=1)
        encoded = self.encoder(batch, x, vector_features)
        scalar_node_features = encoded["scalar_features"]
        graph_features = global_mean_pool(scalar_node_features, batch_index)
        gate_logits = self.graph_gate(torch.cat([graph_features, base_energies], dim=-1))
        gate_weights = torch.softmax(gate_logits, dim=-1)

        mixed_energy = torch.sum(gate_weights * base_energies, dim=-1, keepdim=True)
        correction_energy = global_add_pool(self.atomic_correction(scalar_node_features), batch_index)
        total_energy = torch.nan_to_num(mixed_energy + correction_energy)

        return {
            "energy": total_energy,
            "forces": energy_to_forces(total_energy, batch.pos, create_graph=self.training),
            "gate_weights": gate_weights,
            "base_energies": base_energies,
            "base_forces": base_forces,
        }
