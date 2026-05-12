#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation, force_temperature
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "el-mlffs"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import train_torch_ensemble as ensemble  # noqa: E402
from relax_peptide_base_models import FORCE_SCALE, load_raw_metadata, min_distance, sample_force_stats  # noqa: E402


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


def select_sample(lmdb_path: Path, results_root: Path, sample_id: str | None):
    if sample_id:
        for idx, sample in iter_lmdb_samples(lmdb_path):
            if str(sample.sample_id) == sample_id:
                return idx, "user", sample
        raise KeyError(f"Sample id not found: {sample_id}")

    metadata = load_raw_metadata(results_root)
    best = None
    for idx, sample in iter_lmdb_samples(lmdb_path):
        meta = metadata.get(str(sample.sample_id), {})
        if meta.get("run_type") != "opt":
            continue
        max_force, _ = sample_force_stats(sample)
        if best is None or max_force < best[0]:
            best = (max_force, idx, sample)
    if best is None:
        raise RuntimeError("Could not find an opt sample.")
    return best[1], "opt_low", best[2]


def atoms_from_sample(sample) -> Atoms:
    return Atoms(
        numbers=sample.z.detach().cpu().numpy().astype(int),
        positions=sample.pos.detach().cpu().numpy().astype(float),
        pbc=False,
        cell=[80.0, 80.0, 80.0],
    )


def build_nonperiodic_batch(
    atoms: Atoms,
    cutoff: float,
    device: torch.device,
    fixed_edge_index: torch.Tensor | None = None,
) -> Data:
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    if fixed_edge_index is None:
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist2 = torch.sum(diff * diff, dim=-1)
        edge_mask = (dist2 <= cutoff * cutoff) & (dist2 > 1e-12)
        edge_index = edge_mask.nonzero(as_tuple=False).t().contiguous().to(dtype=torch.long)
    else:
        edge_index = fixed_edge_index
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 80.0
    batch = torch.zeros(z.numel(), dtype=torch.long, device=device)
    return Data(z=z, pos=pos, edge_index=edge_index, shifts=shifts, cell=cell, batch=batch)


class ConservativeMetaCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model: torch.nn.Module, cutoff: float, device: torch.device, fixed_neighbor_graph: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.cutoff = cutoff
        self.device = device
        self.fixed_neighbor_graph = fixed_neighbor_graph
        self.fixed_edge_index: torch.Tensor | None = None

    def set_fixed_graph(self, atoms: Atoms) -> None:
        batch = build_nonperiodic_batch(atoms, self.cutoff, self.device)
        self.fixed_edge_index = batch.edge_index

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        if self.fixed_neighbor_graph and self.fixed_edge_index is None:
            self.set_fixed_graph(self.atoms)
        batch = build_nonperiodic_batch(
            self.atoms,
            self.cutoff,
            self.device,
            fixed_edge_index=self.fixed_edge_index if self.fixed_neighbor_graph else None,
        )
        outputs = self.model(batch)
        self.results["energy"] = float(outputs["energy"].view(-1)[0].detach().cpu().item())
        self.results["forces"] = outputs["forces"].detach().cpu().numpy()


def load_meta_model(model_path: Path, base_model_dir: Path, base_checkpoint_template: str, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    print(f"[load] meta checkpoint: {model_path}", flush=True)
    checkpoint = ensemble.load_checkpoint_bundle(str(model_path), map_location=device)
    metadata = dict(checkpoint.get("metadata") or {})
    config_dict = dict(checkpoint.get("config") or {})
    base_model_names = ensemble.normalize_base_model_names(
        metadata.get("base_model_names") or config_dict.get("base_model_names") or ensemble.DEFAULT_BASE_MODEL_NAMES
    )
    for model_name in base_model_names:
        checkpoint_path = base_model_dir / base_checkpoint_template.format(model=model_name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing base-model checkpoint: {checkpoint_path}")
        ensemble.BASE_MODEL_CONFIGS[model_name] = {"checkpoint": str(checkpoint_path)}
        print(f"[load] base checkpoint: {checkpoint_path}", flush=True)

    all_z = sorted(int(z) for z in metadata.get("atomic_numbers", [1, 6, 7, 8, 16]))
    train_config = ensemble.TrainConfig(
        architecture="conservative",
        cutoff=float(metadata.get("cutoff", config_dict.get("cutoff", 5.0))),
        freeze_base_models=True,
        base_model_names=tuple(base_model_names),
    )
    model = ensemble.build_model(train_config, all_z, device)
    print("[load] ensemble model built; loading state dict", flush=True)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device), metadata, config_dict


def max_force(atoms: Atoms) -> float:
    forces = atoms.get_forces()
    return float(np.linalg.norm(forces, axis=1).max())


def write_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lmdb-path", type=Path, default=ROOT / "el-mlffs/data/peptide/peptide_dft_wb97x_5k.lmdb")
    parser.add_argument("--results-root", type=Path, required=True, help="Directory containing peptide DFT reference results")
    parser.add_argument("--base-model-dir", type=Path, default=ROOT / "el-mlffs/checkpoints/peptide_dft_wb97x_5k_base_models")
    parser.add_argument("--base-checkpoint-template", default="{model}_torch.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/peptide_meta_nve_smoke")
    parser.add_argument("--sample-id")
    parser.add_argument("--initial-structure", type=Path, help="Extxyz/traj file to start from, skipping LMDB selection and relaxation")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--temperature-k", type=float, default=150.0)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--traj-interval", type=int, default=200)
    parser.add_argument("--relax-steps", type=int, default=2000)
    parser.add_argument("--relax-fmax", type=float, default=0.02)
    parser.add_argument("--stop-drift-mev-atom", type=float, default=50.0)
    parser.add_argument("--stop-min-distance", type=float, default=0.45)
    parser.add_argument("--fixed-neighbor-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, metadata, _ = load_meta_model(args.model_path, args.base_model_dir, args.base_checkpoint_template, device)
    cutoff_val = args.cutoff if args.cutoff is not None else float(metadata.get("cutoff", 5.0))

    if args.initial_structure:
        atoms = ase.io.read(str(args.initial_structure))
        sample_label = "hotstart"
        sample_id = Path(args.initial_structure).stem
        run_dir = args.output_dir / f"{sample_label}_{sample_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        atoms.calc = ConservativeMetaCalculator(model=model, cutoff=cutoff_val, device=device, fixed_neighbor_graph=args.fixed_neighbor_graph)
        initial_dft_fmax = 0.0
        initial_dft_mean_force = 0.0
        relaxed_energy = float(atoms.get_potential_energy())
        relaxed_fmax = max_force(atoms)
        relaxed_min_dist = min_distance(atoms)
        relax_converged = True
        print(f"[hotstart] loaded {args.initial_structure} | atoms={len(atoms)} | E={relaxed_energy:.3f}", flush=True)
    else:
        sample_idx, sample_label, sample = select_sample(args.lmdb_path, args.results_root, args.sample_id)
        print(f"[sample] label={sample_label} idx={sample_idx} sample_id={sample.sample_id}", flush=True)
        run_dir = args.output_dir / f"{sample_label}_{sample.sample_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        atoms = atoms_from_sample(sample)
        atoms.calc = ConservativeMetaCalculator(model=model, cutoff=cutoff_val, device=device, fixed_neighbor_graph=args.fixed_neighbor_graph)

        initial_dft_fmax, initial_dft_mean_force = sample_force_stats(sample)
        print("[stage] pre-relax evaluation", flush=True)
        pre_relax_energy = float(atoms.get_potential_energy())
        pre_relax_fmax = max_force(atoms)
        pre_relax_min_dist = min_distance(atoms)

        opt_log = run_dir / "relax.log"
        opt_traj = run_dir / "relax.traj"
        print(f"[stage] relax start | fmax={args.relax_fmax} | steps={args.relax_steps}", flush=True)
        optimizer = FIRE(atoms, logfile=str(opt_log), trajectory=str(opt_traj))
        relax_converged = bool(optimizer.run(fmax=args.relax_fmax, steps=args.relax_steps))
        relaxed_energy = float(atoms.get_potential_energy())
        relaxed_fmax = max_force(atoms)
        relaxed_min_dist = min_distance(atoms)
        write(str(run_dir / "relaxed.extxyz"), atoms, format="extxyz")
        if args.fixed_neighbor_graph:
            atoms.calc.set_fixed_graph(atoms)

    rng = np.random.default_rng(args.seed)
    print(f"[stage] nve start | T={args.temperature_k}K | dt={args.timestep_fs}fs | steps={args.steps}", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)
    force_temperature(atoms, temperature=args.temperature_k, unit="K")

    dyn = VelocityVerlet(atoms, args.timestep_fs * units.fs)
    traj = Trajectory(str(run_dir / "nve.traj"), "w", atoms)
    rows = []
    initial_total = None
    stopped_reason = ""

    def record(step: int) -> bool:
        nonlocal initial_total, stopped_reason
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        etot = epot + ekin
        if initial_total is None:
            initial_total = etot
        drift_mev_atom = 1000.0 * (etot - initial_total) / len(atoms)
        abs_drift_mev_atom = abs(drift_mev_atom)
        fmax = max_force(atoms)
        mind = min_distance(atoms)
        row = {
            "step": step,
            "time_fs": step * args.timestep_fs,
            "epot_eV": epot,
            "ekin_eV": ekin,
            "etot_eV": etot,
            "drift_mev_atom": drift_mev_atom,
            "abs_drift_mev_atom": abs_drift_mev_atom,
            "temperature_K": float(atoms.get_temperature()),
            "fmax_eV_A": fmax,
            "min_distance_A": mind,
        }
        rows.append(row)
        write_rows(rows, run_dir / "nve_metrics.csv")
        print(
            f"Step {step:6d} | Time {row['time_fs']:8.2f} fs | "
            f"E_pot {epot:11.6f} eV | E_kin {ekin:10.6f} eV | E_tot {etot:11.6f} eV | "
            f"Drift {drift_mev_atom:9.4f} meV/atom | "
            f"|Drift| {abs_drift_mev_atom:9.4f} meV/atom | "
            f"T {row['temperature_K']:8.2f} K | "
            f"Fmax {fmax:8.4f} eV/A | dmin {mind:7.4f} A",
            flush=True,
        )
        if step % args.traj_interval == 0:
            traj.write(atoms)
        if abs_drift_mev_atom > args.stop_drift_mev_atom:
            stopped_reason = f"drift>{args.stop_drift_mev_atom}"
            return False
        if mind < args.stop_min_distance:
            stopped_reason = f"min_distance<{args.stop_min_distance}"
            return False
        return True

    ok = record(0)
    step = 0
    while ok and step < args.steps:
        dyn.run(args.log_interval)
        step += args.log_interval
        ok = record(step)
    traj.close()
    write_rows(rows, run_dir / "nve_metrics.csv")

    summary = {
        "model_path": str(args.model_path),
        "sample_idx": sample_idx,
        "sample_label": sample_label,
        "sample_id": str(sample.sample_id),
        "natoms": len(atoms),
        "dft_initial_fmax_eV_A": initial_dft_fmax,
        "dft_initial_mean_force_eV_A": initial_dft_mean_force,
        "pre_relax_energy_eV": pre_relax_energy,
        "pre_relax_fmax_eV_A": pre_relax_fmax,
        "pre_relax_min_distance_A": pre_relax_min_dist,
        "relax_converged": relax_converged,
        "relax_steps": optimizer.nsteps,
        "relaxed_energy_eV": relaxed_energy,
        "relaxed_fmax_eV_A": relaxed_fmax,
        "relaxed_min_distance_A": relaxed_min_dist,
        "temperature_k": args.temperature_k,
        "timestep_fs": args.timestep_fs,
        "requested_steps": args.steps,
        "completed_steps": step,
        "completed_time_fs": step * args.timestep_fs,
        "stopped_reason": stopped_reason,
        "final_abs_drift_mev_atom": rows[-1]["abs_drift_mev_atom"],
        "max_abs_drift_mev_atom": max(row["abs_drift_mev_atom"] for row in rows),
        "final_temperature_K": rows[-1]["temperature_K"],
        "min_distance_min_A": min(row["min_distance_A"] for row in rows),
        "fixed_neighbor_graph": args.fixed_neighbor_graph,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
