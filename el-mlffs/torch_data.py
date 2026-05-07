import bisect
import fcntl
import glob
import os
import pickle
from typing import Optional

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data, Dataset

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


def get_batch_index(z: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    if batch is None:
        return torch.zeros(z.size(0), dtype=torch.long, device=z.device)
    return batch


def compute_edge_geometry(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    cell: torch.Tensor,
    batch: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    batch_index = get_batch_index(pos, batch)
    edge_batch = batch_index[row]
    edge_cell = cell[edge_batch]
    physical_shift = torch.bmm(shifts.unsqueeze(1), edge_cell).squeeze(1)
    edge_vec = pos[row] - pos[col] - physical_shift
    edge_length = torch.sqrt(torch.sum(edge_vec.pow(2), dim=-1) + 1e-12)
    return edge_vec, edge_length


def energy_to_forces(
    energy: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool = True,
    retain_graph: Optional[bool] = None,
) -> torch.Tensor:
    if retain_graph is None:
        retain_graph = create_graph
    grad_outputs = torch.ones_like(energy)
    gradients = torch.autograd.grad(
        outputs=energy,
        inputs=pos,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]
    return -gradients


class ExtXYZDataset(Dataset):
    def __init__(
        self,
        extxyz_file: str,
        cutoff: float = 5.0,
        energy_key: str = "energy",
        force_key: str = "force",
        cache_tag: str = "torch_graphs_v2",
    ) -> None:
        super().__init__()
        self.extxyz_file = extxyz_file
        self.cutoff = cutoff
        self.energy_key = energy_key
        self.force_key = force_key
        self.cache_file = f"{extxyz_file}_{cache_tag}_{cutoff:.2f}.pt"

        if os.path.exists(self.cache_file):
            self.data_list = torch.load(self.cache_file)
        else:
            self.data_list = self._load_from_extxyz()
            torch.save(self.data_list, self.cache_file)

    def _resolve_energy(self, atoms) -> float:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass

        if self.energy_key in atoms.info:
            return float(atoms.info[self.energy_key])

        raise KeyError(
            f"Could not find energy for frame in {self.extxyz_file}. "
            f"Expected calculator energy or info['{self.energy_key}']."
        )

    def _resolve_forces(self, atoms) -> np.ndarray:
        candidate_keys = [self.force_key]
        if self.force_key == "force":
            candidate_keys.append("forces")
        elif self.force_key == "forces":
            candidate_keys.append("force")

        for key in candidate_keys:
            if key in atoms.arrays:
                return np.asarray(atoms.arrays[key], dtype=np.float32)

        raise KeyError(
            f"Could not find force array for frame in {self.extxyz_file}. "
            f"Tried keys {candidate_keys}; available arrays are {sorted(atoms.arrays.keys())}."
        )

    def _load_from_extxyz(self) -> list[Data]:
        atoms_list = ase.io.read(self.extxyz_file, index=":", format="extxyz")
        data_list: list[Data] = []

        for atoms in atoms_list:
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
            energy = torch.tensor([self._resolve_energy(atoms)], dtype=torch.float)
            forces = torch.tensor(self._resolve_forces(atoms), dtype=torch.float)

            virial_val = atoms.info.get("virial")
            if virial_val is None:
                virial = torch.zeros(3, 3, dtype=torch.float)
            else:
                virial = torch.tensor(np.asarray(virial_val).reshape(3, 3), dtype=torch.float)

            i, j, shifts = neighbor_list("ijS", atoms, self.cutoff)
            edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
            shifts_tensor = torch.tensor(shifts, dtype=torch.float)
            cell_tensor = torch.tensor(atoms.cell.array, dtype=torch.float).unsqueeze(0)

            data_list.append(
                Data(
                    z=z,
                    pos=pos,
                    energy=energy,
                    forces=forces,
                    virial=virial,
                    edge_index=edge_index,
                    shifts=shifts_tensor,
                    cell=cell_tensor,
                )
            )

        return data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def _as_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _resolve_sample_attr(sample, names: list[str]):
    if isinstance(sample, dict):
        for name in names:
            if name in sample:
                return sample[name]
    else:
        for name in names:
            if hasattr(sample, name):
                return getattr(sample, name)
    return None


def _compute_edges_from_structure(
    z: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    atoms = Atoms(
        numbers=z.detach().cpu().numpy(),
        positions=pos.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=np.asarray(pbc, dtype=bool),
    )
    i, j, shifts = neighbor_list("ijS", atoms, cutoff)
    edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
    shifts_tensor = torch.tensor(shifts, dtype=torch.float32)
    return edge_index, shifts_tensor


def _convert_oc20_sample_to_data(
    sample,
    cutoff: float,
    require_precomputed_edges: bool = False,
) -> Data:
    z = _as_tensor(_resolve_sample_attr(sample, ["atomic_numbers", "z"]), dtype=torch.long)
    pos = _as_tensor(_resolve_sample_attr(sample, ["pos", "positions"]), dtype=torch.float32)

    energy_val = _resolve_sample_attr(sample, ["energy", "y"])
    if energy_val is None:
        raise KeyError("OC20 sample is missing energy/y.")
    energy = _as_tensor(energy_val, dtype=torch.float32).view(-1)
    if energy.numel() != 1:
        energy = energy[:1]

    forces_val = _resolve_sample_attr(sample, ["forces", "force"])
    if forces_val is None:
        raise KeyError("OC20 sample is missing forces/force.")
    forces = _as_tensor(forces_val, dtype=torch.float32)

    cell_val = _resolve_sample_attr(sample, ["cell"])
    if cell_val is None:
        raise KeyError("OC20 sample is missing cell.")
    cell = _as_tensor(cell_val, dtype=torch.float32)
    if cell.ndim == 3:
        cell = cell[0]
    if cell.ndim != 2 or cell.shape != (3, 3):
        cell = cell.reshape(3, 3)

    pbc_val = _resolve_sample_attr(sample, ["pbc"])
    if pbc_val is None:
        pbc = np.array([True, True, True], dtype=bool)
    else:
        pbc = np.asarray(_as_tensor(pbc_val).detach().cpu().numpy(), dtype=bool).reshape(-1)
        if pbc.size == 1:
            pbc = np.repeat(pbc, 3)

    edge_index_val = _resolve_sample_attr(sample, ["edge_index"])
    shifts_val = _resolve_sample_attr(sample, ["shifts", "cell_offsets"])
    if edge_index_val is not None and shifts_val is not None:
        edge_index = _as_tensor(edge_index_val, dtype=torch.long)
        shifts = _as_tensor(shifts_val, dtype=torch.float32)
    else:
        if require_precomputed_edges:
            raise KeyError("OC20 sample is missing precomputed edges/cell offsets.")
        edge_index, shifts = _compute_edges_from_structure(z, pos, cell, pbc, cutoff)

    tags_val = _resolve_sample_attr(sample, ["tags"])
    tags = None if tags_val is None else _as_tensor(tags_val, dtype=torch.long)
    fixed_val = _resolve_sample_attr(sample, ["fixed"])
    fixed = None if fixed_val is None else _as_tensor(fixed_val, dtype=torch.long)
    sid_val = _resolve_sample_attr(sample, ["sid"])
    fid_val = _resolve_sample_attr(sample, ["fid"])

    payload = {
        "z": z,
        "pos": pos,
        "energy": energy,
        "forces": forces,
        "virial": torch.zeros(3, 3, dtype=torch.float32),
        "edge_index": edge_index,
        "shifts": shifts,
        "cell": cell.unsqueeze(0),
        "pbc": torch.as_tensor(pbc, dtype=torch.bool),
    }
    if tags is not None:
        payload["tags"] = tags
    if fixed is not None:
        payload["fixed"] = fixed
    if sid_val is not None:
        payload["sid"] = int(sid_val)
    if fid_val is not None:
        payload["fid"] = int(fid_val)
    return Data(**payload)


class OC20LmdbDataset(Dataset):
    def __init__(
        self,
        lmdb_dir: str,
        cutoff: float = 5.0,
        require_precomputed_edges: bool = False,
    ) -> None:
        super().__init__()
        self.lmdb_dir = lmdb_dir
        self.cutoff = cutoff
        self.require_precomputed_edges = require_precomputed_edges
        self.db_paths = sorted(glob.glob(os.path.join(lmdb_dir, "*.lmdb")))
        if not self.db_paths:
            raise FileNotFoundError(f"No LMDB shards found under {lmdb_dir}")

        self.db_lengths = [self._read_length(path) for path in self.db_paths]
        self.cumulative_lengths = np.cumsum(self.db_lengths).tolist()
        self._envs = None

        metadata_path = os.path.join(lmdb_dir, "metadata.json")
        self.atomic_numbers = None
        if os.path.exists(metadata_path):
            try:
                import json

                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                atomic_numbers = metadata.get("atomic_numbers")
                if atomic_numbers:
                    self.atomic_numbers = sorted({int(z) for z in atomic_numbers})
            except Exception:
                self.atomic_numbers = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_envs"] = None
        return state

    def _read_length(self, path: str) -> int:
        import lmdb

        env = lmdb.open(
            path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        try:
            with env.begin() as txn:
                return int(pickle.loads(txn.get(b"length")))
        finally:
            env.close()

    def _ensure_envs(self):
        if self._envs is not None:
            return
        import lmdb

        self._envs = [
            lmdb.open(
                path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )
            for path in self.db_paths
        ]

    def len(self) -> int:
        return self.cumulative_lengths[-1]

    def get(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        self._ensure_envs()
        db_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        prev_total = 0 if db_idx == 0 else self.cumulative_lengths[db_idx - 1]
        local_idx = idx - prev_total
        with self._envs[db_idx].begin() as txn:
            payload = txn.get(str(local_idx).encode("ascii"))
        if payload is None:
            raise KeyError(f"Missing index {local_idx} in {self.db_paths[db_idx]}")
        sample = pickle.loads(payload)
        return _convert_oc20_sample_to_data(
            sample,
            cutoff=self.cutoff,
            require_precomputed_edges=self.require_precomputed_edges,
        )


def _convert_peptide_dft_sample_to_data(
    sample,
    cutoff: float,
    energy_field: str = "ae",
    energy_scale: float = HARTREE_TO_EV,
    force_scale: float = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    cell_size: float = 80.0,
) -> Data:
    z = _as_tensor(_resolve_sample_attr(sample, ["z", "atomic_numbers"]), dtype=torch.long)
    pos = _as_tensor(_resolve_sample_attr(sample, ["pos", "positions"]), dtype=torch.float32)

    energy_val = _resolve_sample_attr(sample, [energy_field])
    if energy_val is None:
        raise KeyError(f"Peptide DFT sample is missing requested energy field: {energy_field}")
    energy = _as_tensor(energy_val, dtype=torch.float32).view(-1)
    if energy.numel() != 1:
        energy = energy[:1]
    energy = energy * float(energy_scale)

    forces_val = _resolve_sample_attr(sample, ["forces", "force"])
    if forces_val is None:
        raise KeyError("Peptide DFT sample is missing forces/force.")
    forces = _as_tensor(forces_val, dtype=torch.float32) * float(force_scale)

    cell = torch.eye(3, dtype=torch.float32) * float(cell_size)
    pbc = np.array([False, False, False], dtype=bool)
    # Peptide DFT records are isolated molecules. Direct pairwise construction is
    # much faster than ASE neighbor_list and gives zero periodic shifts.
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist2 = torch.sum(diff * diff, dim=-1)
    cutoff2 = float(cutoff) * float(cutoff)
    edge_mask = (dist2 <= cutoff2) & (dist2 > 1e-12)
    edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous().to(dtype=torch.long)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32)

    payload = {
        "z": z,
        "pos": pos,
        "energy": energy,
        "forces": forces,
        "virial": torch.zeros(3, 3, dtype=torch.float32),
        "edge_index": edge_index,
        "shifts": shifts,
        "cell": cell.unsqueeze(0),
        "pbc": torch.as_tensor(pbc, dtype=torch.bool),
    }
    sample_id = _resolve_sample_attr(sample, ["sample_id", "sid"])
    if sample_id is not None:
        payload["sample_id"] = str(sample_id)
    source = _resolve_sample_attr(sample, ["source"])
    if source is not None:
        payload["source"] = str(source)
    return Data(**payload)


class PeptideDftLmdbDataset(Dataset):
    """Single-file PyG LMDB for GPU4PySCF peptide DFT records.

    The source LMDB stores positions in Angstrom, atomization energy in Hartree
    by default, and forces in Hartree/Bohr. The training code expects eV and
    eV/Angstrom, so conversion is done at load time.
    """

    def __init__(
        self,
        lmdb_file: str,
        cutoff: float = 5.0,
        energy_field: str = "ae",
        energy_scale: float = HARTREE_TO_EV,
        force_scale: float = HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        cell_size: float = 80.0,
        max_force_norm: float | None = None,
        cache_dir: str | None = None,
        cache_tag: str = "torch_graphs_peptide_v1",
    ) -> None:
        super().__init__()
        self.lmdb_file = lmdb_file
        self.cutoff = cutoff
        self.energy_field = energy_field
        self.energy_scale = energy_scale
        self.force_scale = force_scale
        self.cell_size = cell_size
        self.max_force_norm = max_force_norm
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag
        self._env = None
        self._length = self._read_length()
        self.filtered_indices = self._build_filtered_indices() if self.max_force_norm is not None else None
        self.atomic_numbers = [1, 6, 7, 8, 16]
        self.cache_file = self._resolve_cache_file()
        self.data_list = self._load_or_build_cache()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def _open_env(self):
        import lmdb

        return lmdb.open(
            self.lmdb_file,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

    def _read_length(self) -> int:
        env = self._open_env()
        try:
            with env.begin() as txn:
                payload = txn.get(b"length")
                if payload is None:
                    raise KeyError(f"Missing length key in {self.lmdb_file}")
                return int(pickle.loads(payload))
        finally:
            env.close()

    def _ensure_env(self):
        if self._env is None:
            self._env = self._open_env()

    def len(self) -> int:
        return len(self.data_list)

    def _read_raw_sample(self, idx: int):
        self._ensure_env()
        with self._env.begin() as txn:
            payload = txn.get(str(idx).encode("ascii"))
        if payload is None:
            raise KeyError(f"Missing index {idx} in {self.lmdb_file}")
        return pickle.loads(payload)

    def _build_filtered_indices(self) -> np.ndarray:
        threshold = float(self.max_force_norm)
        valid_indices: list[int] = []
        for idx in range(self._length):
            sample = self._read_raw_sample(idx)
            forces_val = _resolve_sample_attr(sample, ["forces", "force"])
            if forces_val is None:
                raise KeyError("Peptide DFT sample is missing forces/force.")
            forces = _as_tensor(forces_val, dtype=torch.float32) * float(self.force_scale)
            max_norm = torch.linalg.vector_norm(forces, dim=1).max().item()
            if max_norm <= threshold:
                valid_indices.append(idx)
        return np.asarray(valid_indices, dtype=np.int64)

    def get(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        return self.data_list[idx]

    def _resolve_cache_file(self) -> str:
        cache_dir = self.cache_dir or os.path.dirname(os.path.abspath(self.lmdb_file))
        os.makedirs(cache_dir, exist_ok=True)
        basename = os.path.basename(self.lmdb_file)
        force_tag = "noforcefilter" if self.max_force_norm is None else f"maxF{float(self.max_force_norm):.3f}"
        tag = (
            f"{basename}_{self.cache_tag}_cutoff{self.cutoff:.2f}_"
            f"energy{self.energy_field}_cell{float(self.cell_size):.1f}_{force_tag}.pt"
        )
        return os.path.join(cache_dir, tag)

    def _load_or_build_cache(self) -> list[Data]:
        if os.path.exists(self.cache_file):
            return torch.load(self.cache_file)

        lock_path = self.cache_file + ".lock"
        with open(lock_path, "w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if os.path.exists(self.cache_file):
                return torch.load(self.cache_file)

            data_list: list[Data] = []
            raw_indices = (
                self.filtered_indices.tolist()
                if self.filtered_indices is not None
                else list(range(self._length))
            )
            for raw_idx in raw_indices:
                sample = self._read_raw_sample(int(raw_idx))
                data_list.append(
                    _convert_peptide_dft_sample_to_data(
                        sample,
                        cutoff=self.cutoff,
                        energy_field=self.energy_field,
                        energy_scale=self.energy_scale,
                        force_scale=self.force_scale,
                        cell_size=self.cell_size,
                    )
                )

            tmp_path = self.cache_file + ".tmp"
            torch.save(data_list, tmp_path)
            os.replace(tmp_path, self.cache_file)
            return data_list

    def _get_uncached(self, idx: int) -> Data:
        raw_idx = int(self.filtered_indices[idx]) if self.filtered_indices is not None else idx
        sample = self._read_raw_sample(raw_idx)
        return _convert_peptide_dft_sample_to_data(
            sample,
            cutoff=self.cutoff,
            energy_field=self.energy_field,
            energy_scale=self.energy_scale,
            force_scale=self.force_scale,
            cell_size=self.cell_size,
        )


def build_dataset(
    data_path: str,
    cutoff: float = 5.0,
    dataset_backend: str = "extxyz",
    dataset_kwargs: Optional[dict[str, int | float | bool | str]] = None,
):
    dataset_kwargs = dict(dataset_kwargs or {})
    if dataset_backend == "extxyz":
        return ExtXYZDataset(data_path, cutoff=cutoff, **dataset_kwargs)
    if dataset_backend == "oc20_lmdb":
        return OC20LmdbDataset(data_path, cutoff=cutoff, **dataset_kwargs)
    if dataset_backend == "peptide_dft_lmdb":
        return PeptideDftLmdbDataset(data_path, cutoff=cutoff, **dataset_kwargs)
    raise ValueError(f"Unsupported dataset backend: {dataset_backend}")
