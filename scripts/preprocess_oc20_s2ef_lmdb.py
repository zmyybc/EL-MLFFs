#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import lzma
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import ase.io
import lmdb
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess extracted OC20 S2EF split into EL-MLFFs LMDB shards.")
    parser.add_argument("--split-root", required=True, help="Path under extracted/<split> containing the tar contents.")
    parser.add_argument("--output-dir", required=True, help="Directory to write LMDB shards and metadata.")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ref-energy", action="store_true", help="Subtract the reference energy column from labels.")
    parser.add_argument("--store-edges", action="store_true", help="Precompute and store edge_index/shifts in LMDB.")
    parser.add_argument("--skip-uncompress", action="store_true", help="Assume plain .extxyz/.txt files already exist.")
    parser.add_argument("--keep-uncompressed", action="store_true", help="Keep uncompressed files on disk after preprocessing.")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def find_compressed_dir(split_root: Path) -> Path:
    candidates = []
    for root, _, files in os.walk(split_root):
        if any(name.endswith(".extxyz.xz") for name in files):
            candidates.append(Path(root))
    if not candidates:
        raise FileNotFoundError(f"No compressed OC20 trajectories found under {split_root}")
    candidates.sort()
    return candidates[0]


def uncompress_one(task: tuple[str, str]) -> None:
    src, dst = task
    if os.path.exists(dst):
        return
    with lzma.open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(fin.read())


def resolve_plain_dir(args: argparse.Namespace) -> Path:
    split_root = Path(args.split_root)
    if args.skip_uncompress:
        return split_root

    compressed_dir = find_compressed_dir(split_root)
    plain_dir = Path(args.output_dir) / "uncompressed"
    plain_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str]] = []
    for src in sorted(glob.glob(str(compressed_dir / "*.txt.xz"))) + sorted(glob.glob(str(compressed_dir / "*.extxyz.xz"))):
        src_path = Path(src)
        tasks.append((str(src_path), str(plain_dir / src_path.name[:-3])))

    with mp.Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap(uncompress_one, tasks),
                total=len(tasks),
                desc="Uncompressing OC20 split",
            )
        )
    return plain_dir


def resolve_energy(atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        pass
    for key in ("energy", "y"):
        if key in atoms.info:
            return float(atoms.info[key])
    raise KeyError("Missing energy in OC20 frame.")


def resolve_forces(atoms) -> np.ndarray:
    for key in ("forces", "force"):
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=np.float32)
    try:
        return np.asarray(atoms.get_forces(), dtype=np.float32)
    except Exception as exc:
        raise KeyError("Missing forces in OC20 frame.") from exc


def parse_frame_log(line: str) -> tuple[int, int, float | None]:
    parts = [part.strip() for part in line.split(",")]
    sid = int(parts[0].split("random")[-1])
    fid = int(parts[1].split("frame")[-1])
    ref_energy = float(parts[2]) if len(parts) > 2 else None
    return sid, fid, ref_energy


def compute_edges(atoms, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    i, j, shifts = neighbor_list("ijS", atoms, cutoff)
    edge_index = np.vstack([i, j]).astype(np.int64, copy=False)
    shifts = np.asarray(shifts, dtype=np.float32)
    return edge_index, shifts


def write_worker_shard(task) -> tuple[int, list[int]]:
    txt_files, plain_dir, output_dir, shard_idx, args = task
    db_path = os.path.join(output_dir, f"data.{shard_idx:04d}.lmdb")
    env = lmdb.open(
        db_path,
        map_size=1099511627776,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    count = 0
    atomic_numbers: set[int] = set()
    try:
        for txt_path in txt_files:
            txt_base = os.path.splitext(os.path.basename(txt_path))[0]
            extxyz_path = os.path.join(plain_dir, f"{txt_base}.extxyz")
            with open(txt_path, encoding="utf-8") as handle:
                logs = handle.read().splitlines()
            frames = ase.io.iread(extxyz_path, index=":", format="extxyz")
            for frame, line in zip(frames, logs):
                sid, fid, ref_energy = parse_frame_log(line)
                energy = resolve_energy(frame)
                if args.ref_energy and ref_energy is not None:
                    energy -= ref_energy
                payload = {
                    "atomic_numbers": np.asarray(frame.get_atomic_numbers(), dtype=np.int64),
                    "pos": np.asarray(frame.get_positions(), dtype=np.float32),
                    "cell": np.asarray(frame.cell.array, dtype=np.float32),
                    "pbc": np.asarray(frame.pbc, dtype=bool),
                    "energy": np.asarray([energy], dtype=np.float32),
                    "forces": resolve_forces(frame),
                    "tags": np.asarray(frame.get_tags(), dtype=np.int64),
                    "sid": sid,
                    "fid": fid,
                }
                if args.store_edges:
                    edge_index, shifts = compute_edges(frame, args.cutoff)
                    payload["edge_index"] = edge_index
                    payload["shifts"] = shifts

                for z in payload["atomic_numbers"]:
                    atomic_numbers.add(int(z))

                with env.begin(write=True) as txn:
                    txn.put(str(count).encode("ascii"), pickle.dumps(payload, protocol=-1))
                count += 1

        with env.begin(write=True) as txn:
            txn.put(b"length", pickle.dumps(count, protocol=-1))
        env.sync()
        return count, sorted(atomic_numbers)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plain_dir = resolve_plain_dir(args)

    txt_files = sorted(glob.glob(str(Path(plain_dir) / "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No plain .txt trajectory logs found under {plain_dir}")
    if args.max_trajectories is not None:
        txt_files = txt_files[: args.max_trajectories]

    num_workers = max(1, min(args.num_workers, len(txt_files)))
    chunks = [chunk.tolist() for chunk in np.array_split(np.array(txt_files, dtype=object), num_workers) if len(chunk) > 0]
    tasks = [
        (chunk, str(plain_dir), str(output_dir), shard_idx, args)
        for shard_idx, chunk in enumerate(chunks)
    ]

    total_samples = 0
    atomic_numbers: set[int] = set()
    with mp.Pool(num_workers) as pool:
        for shard_count, shard_z in tqdm(
            pool.imap_unordered(write_worker_shard, tasks),
            total=len(tasks),
            desc="Writing OC20 LMDB shards",
        ):
            total_samples += shard_count
            atomic_numbers.update(shard_z)

    metadata = {
        "format": "el_mlffs_oc20_lmdb_v1",
        "split_root": str(Path(args.split_root).resolve()),
        "plain_dir": str(Path(plain_dir).resolve()),
        "cutoff": args.cutoff,
        "num_workers": num_workers,
        "ref_energy": bool(args.ref_energy),
        "store_edges": bool(args.store_edges),
        "num_samples": total_samples,
        "atomic_numbers": sorted(atomic_numbers),
        "num_shards": len(tasks),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    if not args.keep_uncompressed and not args.skip_uncompress:
        for path in Path(plain_dir).glob("*"):
            path.unlink()
        Path(plain_dir).rmdir()

    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
