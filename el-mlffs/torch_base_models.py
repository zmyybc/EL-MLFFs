from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet, global_add_pool

from torch_data import compute_edge_geometry, energy_to_forces, get_batch_index


class TorchForceField(nn.Module):
    def forward_energy(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch, compute_forces: bool = True, create_graph: bool = True) -> dict[str, torch.Tensor]:
        if compute_forces and not batch.pos.requires_grad:
            batch.pos = batch.pos.clone().detach().requires_grad_(True)

        energy = self.forward_energy(batch)
        outputs = {"energy": energy}
        if compute_forces:
            outputs["forces"] = energy_to_forces(energy, batch.pos, create_graph=create_graph)
        return outputs


class CosineEnvelope(nn.Module):
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        scaled = distances * (torch.pi / self.cutoff)
        envelope = 0.5 * (torch.cos(scaled) + 1.0)
        return torch.where(distances < self.cutoff, envelope, torch.zeros_like(distances))


class SmoothBumpEnvelope(nn.Module):
    """
    Compact-support C-infinity envelope.
    All derivatives vanish at the cutoff.
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        scaled = distances / self.cutoff
        mask = scaled < 1.0
        out = torch.zeros_like(distances)
        scaled_masked = scaled[mask]
        denom = 1.0 - scaled_masked.pow(2)
        out[mask] = torch.exp(1.0 - 1.0 / denom)
        return out


class GaussianBasis(nn.Module):
    def __init__(self, num_basis: int = 32, cutoff: float = 5.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0]).pow(2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).pow(2))


class TorchSchNet(TorchForceField, SchNet):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout="add",
        )
        self.cutoff = cutoff
        self.envelope = SmoothBumpEnvelope(cutoff)

    def forward_energy(self, batch) -> torch.Tensor:
        h = self.embedding(batch.z)
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)

        edge_attr = self.distance_expansion(edge_length)
        edge_attr = edge_attr * self.envelope(edge_length).unsqueeze(-1)

        for interaction in self.interactions:
            h = h + interaction(h, batch.edge_index, edge_length, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        return global_add_pool(h, batch_index)


class SafePaiNNMessage(nn.Module):
    def __init__(self, hidden_channels: int, num_basis: int) -> None:
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.filter_layer = nn.Linear(num_basis, hidden_channels * 3)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        direction = edge_vec / edge_length.unsqueeze(-1)

        weights = self.scalar_mlp(scalar[col]) * self.filter_layer(edge_attr)
        weights = weights * edge_gate.unsqueeze(-1)
        w1, w2, w3 = torch.chunk(weights, 3, dim=-1)

        delta_scalar = torch.zeros_like(scalar)
        delta_vector = torch.zeros_like(vector)

        delta_scalar.index_add_(0, row, w1)
        delta_vector.index_add_(0, row, vector[col] * w2.unsqueeze(1) + w3.unsqueeze(1) * direction.unsqueeze(-1))
        return delta_scalar, delta_vector


class SafePaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.u = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_vec = self.u(vector)
        v_vec = self.v(vector)
        v_norm = torch.sqrt(torch.sum(v_vec.pow(2), dim=1) + 1e-8)
        a1, a2, a3 = torch.chunk(self.mlp(torch.cat([scalar, v_norm], dim=-1)), 3, dim=-1)
        delta_scalar = a1 + a2 * torch.sum(u_vec * v_vec, dim=1)
        delta_vector = a3.unsqueeze(1) * u_vec
        return delta_scalar, delta_vector


class TorchPaiNN(TorchForceField):
    def __init__(self, hidden_channels: int = 128, num_layers: int = 3, num_basis: int = 32, cutoff: float = 5.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_channels)
        self.rbf = GaussianBasis(num_basis=num_basis, cutoff=cutoff)
        self.envelope = SmoothBumpEnvelope(cutoff)
        self.message_blocks = nn.ModuleList([SafePaiNNMessage(hidden_channels, num_basis) for _ in range(num_layers)])
        self.update_blocks = nn.ModuleList([SafePaiNNUpdate(hidden_channels) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward_energy(self, batch) -> torch.Tensor:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        scalar = self.embedding(batch.z)
        vector = torch.zeros(scalar.size(0), 3, scalar.size(1), device=scalar.device, dtype=batch.pos.dtype)
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)

        edge_gate = self.envelope(edge_length)
        edge_attr = self.rbf(edge_length) * edge_gate.unsqueeze(-1)
        for message_block, update_block in zip(self.message_blocks, self.update_blocks):
            delta_scalar, delta_vector = message_block(scalar, vector, batch.edge_index, edge_length, edge_vec, edge_attr, edge_gate)
            scalar, vector = scalar + delta_scalar, vector + delta_vector
            delta_scalar, delta_vector = update_block(scalar, vector)
            scalar, vector = scalar + delta_scalar, vector + delta_vector

        return global_add_pool(self.readout(scalar), batch_index)


class DPDescriptor(nn.Module):
    """
    DeepPot-style descriptor:
    1. edge embedding from smoothed inverse distance and neighbor type
    2. environment matrix aggregation
    3. invariant contraction between full channels and axis channels
    """

    def __init__(
        self,
        num_types: int,
        r_cs: float = 4.0,
        r_c: float = 5.0,
        embed_dim: int = 64,
        axis_dim: int = 16,
        type_emb_dim: int = 16,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
    ) -> None:
        super().__init__()
        self.r_cs = r_cs
        self.r_c = r_c
        self.embed_dim = embed_dim
        self.axis_dim = axis_dim
        self.envelope = SmoothBumpEnvelope(r_c)
        self.type_embedding = nn.Embedding(num_types, type_emb_dim)

        layers: list[nn.Module] = []
        input_dim = 1 + type_emb_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, embed_dim))
        self.embedding_net = nn.Sequential(*layers)
        self.output_dim = embed_dim * axis_dim

    def get_srij(self, edge_length: torch.Tensor) -> torch.Tensor:
        inv_r = edge_length.clamp_min(1e-8).reciprocal()
        cutoff = self.envelope(edge_length)
        return inv_r * cutoff

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_vec: torch.Tensor,
        num_atoms: int,
        types: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        s_rij = self.get_srij(edge_length)
        neighbor_type_emb = self.type_embedding(types[col])
        g_rij = self.embedding_net(torch.cat([s_rij.unsqueeze(-1), neighbor_type_emb], dim=-1))

        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        r_tilde = torch.cat([s_rij.unsqueeze(-1), s_rij.unsqueeze(-1) * direction], dim=-1)
        edge_feature = torch.einsum("em,ed->emd", g_rij, r_tilde)

        env_mat = torch.zeros(num_atoms, self.embed_dim, 4, device=edge_index.device, dtype=edge_feature.dtype)
        env_mat.view(num_atoms, -1).index_add_(0, row, edge_feature.reshape(edge_feature.size(0), -1))

        axis_mat = env_mat[:, : self.axis_dim, :]
        invariant = torch.einsum("nmd,nad->nma", env_mat, axis_mat)
        return invariant.reshape(num_atoms, -1)


class TorchDP(TorchForceField):
    def __init__(
        self,
        z_list: list[int],
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        descriptor_embed_dim: int = 64,
        descriptor_axis_dim: int = 16,
        descriptor_type_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.z_to_idx = {z: idx for idx, z in enumerate(sorted(set(z_list)))}
        self.descriptor = DPDescriptor(
            num_types=len(self.z_to_idx),
            r_cs=cutoff * 0.8,
            r_c=cutoff,
            embed_dim=descriptor_embed_dim,
            axis_dim=descriptor_axis_dim,
            type_emb_dim=descriptor_type_emb_dim,
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.descriptor.output_dim, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_channels // 2, 1),
                )
                for _ in self.z_to_idx
            ]
        )

    def forward_energy(self, batch) -> torch.Tensor:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)
        type_idx = torch.tensor([self.z_to_idx[int(z.item())] for z in batch.z], device=batch.z.device)
        descriptor = self.descriptor(batch.edge_index, edge_length, edge_vec, batch.z.size(0), type_idx)

        atom_energy = torch.zeros(batch.z.size(0), 1, device=batch.z.device)
        for type_id, mlp in enumerate(self.mlps):
            mask = type_idx == type_id
            if mask.any():
                atom_energy[mask] = mlp(descriptor[mask])
        return global_add_pool(atom_energy, batch_index)


def chebyshev_polynomials(x: torch.Tensor, n_max: int) -> torch.Tensor:
    polynomials = [torch.ones_like(x), x]
    for _ in range(2, n_max + 1):
        polynomials.append(2 * x * polynomials[-1] - polynomials[-2])
    return torch.stack(polynomials, dim=-1)


def flatten_upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    tri = torch.triu_indices(matrix.size(-2), matrix.size(-1), device=matrix.device)
    return matrix[..., tri[0], tri[1]]


class NEPDescriptor(nn.Module):
    def __init__(
        self,
        num_types: int,
        n_max_radial: int = 10,
        n_max_angular: int = 6,
        basis_radial: int = 10,
        basis_angular: int = 8,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.envelope = SmoothBumpEnvelope(cutoff)
        self.n_max_radial = n_max_radial
        self.n_max_angular = n_max_angular
        self.c_radial = nn.Parameter(torch.randn(num_types, num_types, n_max_radial + 1, basis_radial + 1) * 0.01)
        self.c_angular = nn.Parameter(torch.randn(num_types, num_types, n_max_angular + 1, basis_angular + 1) * 0.01)
        self.descriptor_dim = (n_max_radial + 1) + 2 * (n_max_angular + 1)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_vec: torch.Tensor,
        num_atoms: int,
        types: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        type_i, type_j = types[row], types[col]
        scaled = 2.0 * (edge_length / self.cutoff) - 1.0
        cutoff = self.envelope(edge_length).unsqueeze(-1)

        radial_basis = chebyshev_polynomials(scaled, self.c_radial.size(-1) - 1) * cutoff
        angular_basis = chebyshev_polynomials(scaled, self.c_angular.size(-1) - 1) * cutoff

        radial_features = torch.einsum("eb,enb->en", radial_basis, self.c_radial[type_i, type_j])
        angular_features = torch.einsum("eb,enb->en", angular_basis, self.c_angular[type_i, type_j])

        radial_descriptor = torch.zeros(num_atoms, self.n_max_radial + 1, device=edge_index.device, dtype=radial_features.dtype)
        radial_descriptor.index_add_(0, row, radial_features)

        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        angular_vector = torch.zeros(num_atoms, self.n_max_angular + 1, 3, device=edge_index.device, dtype=angular_features.dtype)
        for axis in range(3):
            angular_vector[:, :, axis].index_add_(0, row, angular_features * direction[:, axis].unsqueeze(-1))

        identity = torch.eye(3, device=edge_index.device, dtype=angular_features.dtype)
        quadrupole = 1.5 * torch.einsum("ea,eb->eab", direction, direction) - 0.5 * identity.unsqueeze(0)
        angular_tensor = torch.zeros(num_atoms, self.n_max_angular + 1, 3, 3, device=edge_index.device, dtype=angular_features.dtype)
        for axis_i in range(3):
            for axis_j in range(3):
                angular_tensor[:, :, axis_i, axis_j].index_add_(
                    0,
                    row,
                    angular_features * quadrupole[:, axis_i, axis_j].unsqueeze(-1),
                )

        l1_invariants = torch.sum(angular_vector.pow(2), dim=-1)
        l2_invariants = torch.sum(angular_tensor.pow(2), dim=(-1, -2))
        return torch.cat([radial_descriptor, l1_invariants, l2_invariants], dim=-1)


class TorchNEP(TorchForceField):
    def __init__(
        self,
        z_list: list[int],
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        n_max_radial: int = 10,
        n_max_angular: int = 6,
        basis_radial: int = 10,
        basis_angular: int = 8,
    ) -> None:
        super().__init__()
        self.z_to_idx = {z: idx for idx, z in enumerate(sorted(set(z_list)))}
        self.descriptor = NEPDescriptor(
            num_types=len(self.z_to_idx),
            n_max_radial=n_max_radial,
            n_max_angular=n_max_angular,
            basis_radial=basis_radial,
            basis_angular=basis_angular,
            cutoff=cutoff,
        )
        descriptor_dim = self.descriptor.descriptor_dim
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(descriptor_dim, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, 1),
                )
                for _ in self.z_to_idx
            ]
        )

    def forward_energy(self, batch) -> torch.Tensor:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)
        types = torch.tensor([self.z_to_idx[int(z.item())] for z in batch.z], device=batch.z.device)
        descriptor = self.descriptor(batch.edge_index, edge_length, edge_vec, batch.z.size(0), types)

        atom_energy = torch.zeros(batch.z.size(0), 1, device=batch.z.device)
        for type_id, mlp in enumerate(self.mlps):
            mask = types == type_id
            if mask.any():
                atom_energy[mask] = mlp(descriptor[mask])
        return global_add_pool(atom_energy, batch_index)


class MTPDescriptor(nn.Module):
    """
    Moment Tensor Potential style descriptor built from radial basis moments.
    """

    def __init__(
        self,
        num_types: int,
        num_basis: int = 16,
        cutoff: float = 5.0,
        type_emb_dim: int = 8,
        radial_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.rbf = GaussianBasis(num_basis=num_basis, cutoff=cutoff)
        self.envelope = SmoothBumpEnvelope(cutoff)
        self.type_embedding = nn.Embedding(num_types, type_emb_dim)
        self.radial_proj = nn.Sequential(
            nn.Linear(num_basis + type_emb_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, num_basis),
        )
        self.output_dim = num_basis * 4

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_vec: torch.Tensor,
        num_atoms: int,
        types: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        edge_gate = self.envelope(edge_length)
        edge_basis = self.rbf(edge_length) * edge_gate.unsqueeze(-1)
        radial = self.radial_proj(torch.cat([edge_basis, self.type_embedding(types[col])], dim=-1))
        radial = radial * edge_gate.unsqueeze(-1)

        moment_0 = torch.zeros(num_atoms, self.num_basis, device=edge_index.device, dtype=radial.dtype)
        moment_0.index_add_(0, row, radial)

        moment_1_contrib = radial.unsqueeze(-1) * direction.unsqueeze(1)
        moment_1 = torch.zeros(num_atoms, self.num_basis, 3, device=edge_index.device, dtype=radial.dtype)
        moment_1.view(num_atoms, -1).index_add_(0, row, moment_1_contrib.reshape(moment_1_contrib.size(0), -1))

        outer = torch.einsum("ea,eb->eab", direction, direction)
        moment_2_contrib = radial.unsqueeze(-1).unsqueeze(-1) * outer.unsqueeze(1)
        moment_2 = torch.zeros(num_atoms, self.num_basis, 3, 3, device=edge_index.device, dtype=radial.dtype)
        moment_2.view(num_atoms, -1).index_add_(0, row, moment_2_contrib.reshape(moment_2_contrib.size(0), -1))

        return torch.cat(
            [
                moment_0,
                torch.sum(moment_1.pow(2), dim=-1),
                torch.einsum("nbii->nb", moment_2),
                torch.sum(moment_2.pow(2), dim=(-1, -2)),
            ],
            dim=-1,
        )


class TorchMTP(TorchForceField):
    def __init__(
        self,
        z_list: list[int],
        cutoff: float = 5.0,
        hidden_channels: int = 128,
        descriptor_num_basis: int = 16,
        descriptor_type_emb_dim: int = 8,
        descriptor_radial_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.z_to_idx = {z: idx for idx, z in enumerate(sorted(set(z_list)))}
        self.descriptor = MTPDescriptor(
            num_types=len(self.z_to_idx),
            num_basis=descriptor_num_basis,
            cutoff=cutoff,
            type_emb_dim=descriptor_type_emb_dim,
            radial_hidden_dim=descriptor_radial_hidden_dim,
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.descriptor.output_dim, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, 1),
                )
                for _ in self.z_to_idx
            ]
        )

    def forward_energy(self, batch) -> torch.Tensor:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)
        types = torch.tensor([self.z_to_idx[int(z.item())] for z in batch.z], device=batch.z.device)
        descriptor = self.descriptor(batch.edge_index, edge_length, edge_vec, batch.z.size(0), types)

        atom_energy = torch.zeros(batch.z.size(0), 1, device=batch.z.device)
        for type_id, mlp in enumerate(self.mlps):
            mask = types == type_id
            if mask.any():
                atom_energy[mask] = mlp(descriptor[mask])
        return global_add_pool(atom_energy, batch_index)


class SOAPDescriptor(nn.Module):
    """
    SOAP-style power spectrum descriptor with type-resolved spherical expansion.
    """

    def __init__(self, num_types: int, num_radial: int = 8, cutoff: float = 5.0) -> None:
        super().__init__()
        try:
            from e3nn import o3
        except ImportError as exc:
            raise ImportError("TorchSOAP requires e3nn to be installed.") from exc

        self.o3 = o3
        self.num_types = num_types
        self.num_radial = num_radial
        self.rbf = GaussianBasis(num_basis=num_radial, cutoff=cutoff)
        self.envelope = SmoothBumpEnvelope(cutoff)
        self.sh_irreps = o3.Irreps("1x0e + 1x1o + 1x2e")
        self.l_slices = ((0, 1), (1, 4), (4, 9))
        self.output_dim = num_types * len(self.l_slices) * (num_radial * (num_radial + 1) // 2)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_vec: torch.Tensor,
        num_atoms: int,
        types: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index
        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        sh = self.o3.spherical_harmonics(
            self.sh_irreps,
            direction,
            normalize=True,
            normalization="component",
        )
        radial = self.rbf(edge_length) * self.envelope(edge_length).unsqueeze(-1)

        coeffs = torch.zeros(
            num_atoms,
            self.num_types,
            self.num_radial,
            self.sh_irreps.dim,
            device=edge_index.device,
            dtype=radial.dtype,
        )

        for type_id in range(self.num_types):
            mask = types[col] == type_id
            if not mask.any():
                continue
            contrib = radial[mask].unsqueeze(-1) * sh[mask].unsqueeze(1)
            coeffs[:, type_id].view(num_atoms, -1).index_add_(0, row[mask], contrib.reshape(contrib.size(0), -1))

        descriptor_parts: list[torch.Tensor] = []
        for type_id in range(self.num_types):
            type_coeffs = coeffs[:, type_id]
            for start, end in self.l_slices:
                block = type_coeffs[:, :, start:end]
                gram = torch.einsum("nrm,nkm->nrk", block, block)
                descriptor_parts.append(flatten_upper_triangle(gram))
        return torch.cat(descriptor_parts, dim=-1)


class TorchSOAP(TorchForceField):
    def __init__(self, z_list: list[int], cutoff: float = 5.0, hidden_channels: int = 128, num_radial: int = 8) -> None:
        super().__init__()
        self.z_to_idx = {z: idx for idx, z in enumerate(sorted(set(z_list)))}
        self.descriptor = SOAPDescriptor(num_types=len(self.z_to_idx), num_radial=num_radial, cutoff=cutoff)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.descriptor.output_dim, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, 1),
                )
                for _ in self.z_to_idx
            ]
        )

    def forward_energy(self, batch) -> torch.Tensor:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)
        types = torch.tensor([self.z_to_idx[int(z.item())] for z in batch.z], device=batch.z.device)
        descriptor = self.descriptor(batch.edge_index, edge_length, edge_vec, batch.z.size(0), types)

        atom_energy = torch.zeros(batch.z.size(0), 1, device=batch.z.device)
        for type_id, mlp in enumerate(self.mlps):
            mask = types == type_id
            if mask.any():
                atom_energy[mask] = mlp(descriptor[mask])
        return global_add_pool(atom_energy, batch_index)


class MACETensorInteraction(nn.Module):
    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        num_basis: int,
        irreps_sh: str = "1x0e + 1x1o + 1x2e",
    ) -> None:
        super().__init__()
        try:
            from e3nn import o3
        except ImportError as exc:
            raise ImportError("TorchMACE requires e3nn to be installed.") from exc

        self.o3 = o3
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            internal_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_basis, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )
        self.self_interaction = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        sh: torch.Tensor,
        edge_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        weights = self.radial_mlp(edge_attr)
        messages = self.tp(node_feat[col], sh, weights)
        if edge_gate is not None:
            messages = messages * edge_gate.unsqueeze(-1)
        out = torch.zeros(node_feat.size(0), self.irreps_out.dim, device=node_feat.device, dtype=node_feat.dtype)
        out.index_add_(0, row, messages)
        return out + self.self_interaction(node_feat)


class TorchMACE(TorchForceField):
    def __init__(
        self,
        num_layers: int = 4,
        num_basis: int = 32,
        cutoff: float = 5.0,
        hidden_scalar_dim: int = 128,
        hidden_vector_dim: int = 64,
    ) -> None:
        super().__init__()
        try:
            from e3nn import o3
        except ImportError as exc:
            raise ImportError("TorchMACE requires e3nn to be installed.") from exc

        self.o3 = o3
        self.embedding = nn.Embedding(100, hidden_scalar_dim)
        self.rbf = GaussianBasis(num_basis=num_basis, cutoff=cutoff)
        self.envelope = SmoothBumpEnvelope(cutoff=cutoff)
        self.hidden_scalar_dim = hidden_scalar_dim
        self.hidden_irreps = o3.Irreps(f"{hidden_scalar_dim}x0e + {hidden_vector_dim}x1o")
        self.sh_irreps = o3.Irreps("1x0e + 1x1o")
        self.interactions = nn.ModuleList(
            [
                MACETensorInteraction(
                    f"{hidden_scalar_dim}x0e" if layer_idx == 0 else str(self.hidden_irreps),
                    str(self.hidden_irreps),
                    num_basis,
                    irreps_sh=str(self.sh_irreps),
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.scalar_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_scalar_dim, self.hidden_scalar_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_scalar_dim, self.hidden_scalar_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.hidden_scalar_dim),
            nn.Linear(self.hidden_scalar_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward_energy(self, batch) -> torch.Tensor:
        batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))
        edge_vec, edge_length = compute_edge_geometry(batch.pos, batch.edge_index, batch.shifts, batch.cell, batch_index)
        direction = edge_vec / edge_length.clamp_min(1e-8).unsqueeze(-1)
        sh = self.o3.spherical_harmonics(self.sh_irreps, direction, normalize=False)
        edge_gate = self.envelope(edge_length)
        edge_attr = self.rbf(edge_length) * edge_gate.unsqueeze(-1)

        x = self.embedding(batch.z)
        for interaction, scalar_mixer in zip(self.interactions, self.scalar_mixers):
            x = interaction(x, batch.edge_index, edge_attr, sh, edge_gate)
            scalar_part = x[:, : self.hidden_scalar_dim]
            x = torch.cat([scalar_part + scalar_mixer(scalar_part), x[:, self.hidden_scalar_dim :]], dim=-1)
        scalars = x[:, : self.hidden_scalar_dim]
        return global_add_pool(self.readout(scalars), batch_index)


BASE_MODEL_REGISTRY = {
    "dp": TorchDP,
    "nep": TorchNEP,
    "mtp": TorchMTP,
    "soap": TorchSOAP,
    "painn": TorchPaiNN,
    "schnet": TorchSchNet,
    "mace": TorchMACE,
}
