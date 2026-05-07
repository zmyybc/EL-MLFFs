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
from ase import Atoms, units
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation, force_temperature
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "el-mlffs"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from relax_peptide_base_models import BaseModelCalculator, load_model, min_distance, sample_force_stats  # noqa: E402


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
    parser.add_argument("--model", required=True)
    parser.add_argument("--lmdb-path", type=Path, default=Path("/mnt/bn/bangchen1/HORM3/data/peptide_dft_wb97x_5k.lmdb"))
    parser.add_argument("--results-root", type=Path, default=Path("/mnt/bn/changsu-data3/ybc/gvgvpgvg_full_5000_ff_diverse_gpu4pyscf_qc/results_40"))
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "el-mlffs/checkpoints/peptide_dft_wb97x_5k_base_models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/peptide_base_nve_smoke")
    parser.add_argument("--sample-id")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--temperature-k", type=float, default=150.0)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--traj-interval", type=int, default=200)
    parser.add_argument("--relax-steps", type=int, default=300)
    parser.add_argument("--relax-fmax", type=float, default=0.2)
    parser.add_argument("--stop-drift-mev-atom", type=float, default=50.0)
    parser.add_argument("--stop-min-distance", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_idx, sample_label, sample = select_sample(args.lmdb_path, args.results_root, args.sample_id)
    atoms = atoms_from_sample(sample)
    model = load_model(args.model, args.checkpoint_dir, args.cutoff, device)
    atoms.calc = BaseModelCalculator(model=model, cutoff=args.cutoff, device=device)

    initial_dft_fmax, initial_dft_mean_force = sample_force_stats(sample)
    pre_relax_energy = float(atoms.get_potential_energy())
    pre_relax_fmax = max_force(atoms)
    pre_relax_min_dist = min_distance(atoms)

    opt_log = run_dir / "relax.log"
    opt_traj = run_dir / "relax.traj"
    optimizer = FIRE(atoms, logfile=str(opt_log), trajectory=str(opt_traj))
    relax_converged = bool(optimizer.run(fmax=args.relax_fmax, steps=args.relax_steps))
    relaxed_energy = float(atoms.get_potential_energy())
    relaxed_fmax = max_force(atoms)
    relaxed_min_dist = min_distance(atoms)
    write(str(run_dir / "relaxed.extxyz"), atoms, format="extxyz")

    rng = np.random.default_rng(args.seed)
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
        "model": args.model,
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
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
