#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SPLITS = ["2M", "val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare downloaded OC20 S2EF splits for EL-MLFFs training.")
    parser.add_argument("--oc20-root", type=Path, required=True, help="Root containing extracted/<split> directories.")
    parser.add_argument("--processed-root", type=Path, default=None, help="Output root for processed LMDB splits.")
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=DEFAULT_SPLITS)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ref-energy", action="store_true")
    parser.add_argument("--store-edges", action="store_true")
    parser.add_argument("--keep-uncompressed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root or (args.oc20_root / "processed_lmdb")
    processed_root.mkdir(parents=True, exist_ok=True)

    preprocess_script = Path(__file__).with_name("preprocess_oc20_s2ef_lmdb.py")
    for split in args.splits:
        split_root = args.oc20_root / "extracted" / split
        output_dir = processed_root / split
        cmd = [
            sys.executable,
            str(preprocess_script),
            "--split-root",
            str(split_root),
            "--output-dir",
            str(output_dir),
            "--cutoff",
            str(args.cutoff),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.ref_energy:
            cmd.append("--ref-energy")
        if args.store_edges:
            cmd.append("--store-edges")
        if args.keep_uncompressed:
            cmd.append("--keep-uncompressed")
        print("[prepare]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
