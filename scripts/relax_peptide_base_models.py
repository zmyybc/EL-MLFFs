#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from ase.optimize import FIRE
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "el-mlffs"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from torch_base_models import BASE_MODEL_REGISTRY  # noqa: E402

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_SCALE = HARTREE_TO_EV / BOHR_TO_ANGSTROM
DEFAULT_MODELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")


def build_nonperiodic_batch(atoms: Atoms, cutoff: float, device: torch.device) -> Data:
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist2 = torch.sum(diff * diff, dim=-1)
    edge_mask = (dist2 <= cutoff * cutoff) & (dist2 > 1e-12)
    edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous().to(dtype=torch.long)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 80.0
    batch = torch.zeros(z.numel(), dtype=torch.long, device=device)
    return Data(z=z, pos=pos, edge_index=edge_index, shifts=shifts, cell=cell, batch=batch)


class BaseModelCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model: torch.nn.Module, cutoff: float, device: torch.device, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.cutoff = cutoff
        self.device = device

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        batch = build_nonperiodic_batch(self.atoms, self.cutoff, self.device)
        outputs = self.model(batch, compute_forces=True, create_graph=False)
        self.results["energy"] = float(outputs["energy"].view(-1)[0].detach().cpu().item())
        self.results["forces"] = outputs["forces"].detach().cpu().numpy()


def load_model(model_name: str, checkpoint_dir: Path, cutoff: float, device: torch.device) -> torch.nn.Module:
    checkpoint_path = checkpoint_dir / f"{model_name}_torch.pth"
    payload = torch.load(checkpoint_path, map_location=device)
    metadata = dict(payload.get("metadata") or {})
    config = dict(payload.get("config") or {})
    model_kwargs = dict(config.get("model_kwargs") or {})
    all_z = sorted(int(z) for z in metadata.get("atomic_numbers", [1, 6, 7, 8, 16]))
    model_cls = BASE_MODEL_REGISTRY[model_name]
    if model_name in {"dp", "nep", "mtp", "soap"}:
        model = model_cls(z_list=all_z, cutoff=cutoff, **model_kwargs).to(device)
    else:
        model = model_cls(cutoff=cutoff, **model_kwargs).to(device)
    model.load_state_dict(payload.get("state_dict", payload), strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def iter_lmdb_samples(lmdb_path: Path):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False, readahead=False, meminit=False)
    try:
        with env.begin() as txn:
            length = int(pickle.loads(txn.get(b"length")))
            for idx in range(length):
                sample = pickle.loads(txn.get(str(idx).encode("ascii")))
                yield idx, sample
    finally:
        env.close()


def load_raw_metadata(results_root: Path) -> dict[str, dict]:
    metadata = {}
    for path in results_root.glob("shard_*/results/*.json"):
        payload = json.loads(path.read_text())
        metadata[payload.get("id")] = {
            "run_type": payload.get("run_type"),
            "has_optimized_xyz": bool(payload.get("optimized_xyz")),
        }
    return metadata


def sample_force_stats(sample) -> tuple[float, float]:
    forces = sample.forces.detach().cpu().numpy().astype(np.float64) * FORCE_SCALE
    norms = np.linalg.norm(forces, axis=1)
    return float(norms.max()), float(norms.mean())


def atoms_from_sample(sample) -> Atoms:
    return Atoms(
        numbers=sample.z.detach().cpu().numpy().astype(int),
        positions=sample.pos.detach().cpu().numpy().astype(float),
        pbc=False,
        cell=[80.0, 80.0, 80.0],
    )


def min_distance(atoms: Atoms) -> float:
    pos = atoms.get_positions()
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[dist < 1e-12] = np.inf
    return float(dist.min())


def choose_default_samples(lmdb_path: Path, results_root: Path, max_force_norm: float) -> list[tuple[str, int, object]]:
    metadata = load_raw_metadata(results_root)
    candidates: dict[str, list[tuple[float, int, object]]] = {"opt_low": [], "short_opt_high": [], "sp_high": []}
    for idx, sample in iter_lmdb_samples(lmdb_path):
        sample_id = str(sample.sample_id)
        meta = metadata.get(sample_id, {})
        max_force, _ = sample_force_stats(sample)
        if max_force > max_force_norm:
            continue
        run_type = meta.get("run_type")
        if run_type == "opt":
            candidates["opt_low"].append((max_force, idx, sample))
        elif run_type == "short_opt":
            candidates["short_opt_high"].append((max_force, idx, sample))
        elif run_type == "sp":
            candidates["sp_high"].append((max_force, idx, sample))

    selected = []
    selected.append(("opt_low", *min(candidates["opt_low"], key=lambda item: item[0])[1:]))
    selected.append(("short_opt_high", *max(candidates["short_opt_high"], key=lambda item: item[0])[1:]))
    selected.append(("sp_high", *max(candidates["sp_high"], key=lambda item: item[0])[1:]))
    return selected


def relax_one(
    model_name: str,
    model: torch.nn.Module,
    sample_label: str,
    sample_idx: int,
    sample,
    cutoff: float,
    device: torch.device,
    output_dir: Path,
    fmax: float,
    steps: int,
) -> dict[str, float | int | str | bool]:
    atoms = atoms_from_sample(sample)
    atoms.calc = BaseModelCalculator(model=model, cutoff=cutoff, device=device)
    initial_positions = atoms.get_positions().copy()
    initial_energy = float(atoms.get_potential_energy())
    initial_forces = atoms.get_forces()
    initial_fmax = float(np.linalg.norm(initial_forces, axis=1).max())
    initial_mean_force = float(np.linalg.norm(initial_forces, axis=1).mean())
    initial_min_distance = min_distance(atoms)

    run_dir = output_dir / model_name / f"{sample_label}_{sample.sample_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "relax.traj"
    logfile_path = run_dir / "relax.log"
    optimizer = FIRE(atoms, trajectory=str(trajectory_path), logfile=str(logfile_path))
    converged = bool(optimizer.run(fmax=fmax, steps=steps))

    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    final_fmax = float(np.linalg.norm(final_forces, axis=1).max())
    final_mean_force = float(np.linalg.norm(final_forces, axis=1).mean())
    displacement_rms = float(np.sqrt(np.mean(np.sum((atoms.get_positions() - initial_positions) ** 2, axis=1))))
    final_min_distance = min_distance(atoms)
    write(str(run_dir / "final.extxyz"), atoms, format="extxyz")

    dft_max_force, dft_mean_force = sample_force_stats(sample)
    return {
        "model": model_name,
        "sample_label": sample_label,
        "sample_idx": sample_idx,
        "sample_id": str(sample.sample_id),
        "natoms": len(atoms),
        "dft_initial_fmax_eV_A": dft_max_force,
        "dft_initial_mean_force_eV_A": dft_mean_force,
        "model_initial_energy_eV": initial_energy,
        "model_final_energy_eV": final_energy,
        "model_delta_energy_eV": final_energy - initial_energy,
        "model_initial_fmax_eV_A": initial_fmax,
        "model_final_fmax_eV_A": final_fmax,
        "model_initial_mean_force_eV_A": initial_mean_force,
        "model_final_mean_force_eV_A": final_mean_force,
        "initial_min_distance_A": initial_min_distance,
        "final_min_distance_A": final_min_distance,
        "displacement_rms_A": displacement_rms,
        "converged": converged,
        "steps": optimizer.nsteps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb-path", type=Path, default=Path("/mnt/bn/bangchen1/HORM3/data/peptide_dft_wb97x_5k.lmdb"))
    parser.add_argument("--results-root", type=Path, default=Path("/mnt/bn/changsu-data3/ybc/gvgvpgvg_full_5000_ff_diverse_gpu4pyscf_qc/results_40"))
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "el-mlffs/checkpoints/peptide_dft_wb97x_5k_base_models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/peptide_base_relax_smoke")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-force-norm", type=float, default=100.0)
    parser.add_argument("--fmax", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = choose_default_samples(args.lmdb_path, args.results_root, args.max_force_norm)
    rows = []
    for model_name in args.models:
        model = load_model(model_name, args.checkpoint_dir, args.cutoff, device)
        for sample_label, sample_idx, sample in selected_samples:
            print(f"relax model={model_name} sample={sample_label}:{sample.sample_id}", flush=True)
            rows.append(
                relax_one(
                    model_name=model_name,
                    model=model,
                    sample_label=sample_label,
                    sample_idx=sample_idx,
                    sample=sample,
                    cutoff=args.cutoff,
                    device=device,
                    output_dir=args.output_dir,
                    fmax=args.fmax,
                    steps=args.steps,
                )
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.output_dir / "relax_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
