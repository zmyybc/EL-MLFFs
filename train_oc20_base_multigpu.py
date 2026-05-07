#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(ROOT_DIR, "el-mlffs")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from train_one_base_model import autotune_batch_size, compute_schedule, parse_key_value_items, parse_named_paths, resolve_num_train_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune a local batch size and launch multi-GPU OC20 training.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-backend", choices=["extxyz", "oc20_lmdb"], default="oc20_lmdb")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--val-data-file", required=True)
    parser.add_argument("--extra-val-file", action="append", default=[], help="Extra validation sets as name=path.")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--initial-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--target-memory-fraction", type=float, default=0.78)
    parser.add_argument("--memory-headroom-fraction", type=float, default=0.05)
    parser.add_argument("--target-total-steps", type=int, default=20000)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--master-port", type=int, default=29531)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--dataset-kwarg", action="append", default=[], help="Extra dataset kwargs as key=value.")
    parser.add_argument("--model-kwarg", action="append", default=[], help="Extra model kwargs as key=value.")
    args = parser.parse_args()
    args.dataset_kwargs = parse_key_value_items(args.dataset_kwarg)
    args.model_kwargs = parse_key_value_items(args.model_kwarg)
    args.extra_val_files = parse_named_paths(args.extra_val_file)
    return args


def build_torchrun_command(args: argparse.Namespace, local_batch_size: int, epochs: int) -> list[str]:
    save_name = args.save_name or f"{args.model_name}_torch.pth"
    save_path = os.path.join(args.output_dir, save_name)
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port),
        os.path.join(MODULE_DIR, "train_torch_base.py"),
        "--model-name",
        args.model_name,
        "--dataset-backend",
        args.dataset_backend,
        "--data-file",
        args.data_file,
        "--val-data-file",
        args.val_data_file,
        "--cutoff",
        str(args.cutoff),
        "--batch-size",
        str(local_batch_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--energy-weight",
        str(args.energy_weight),
        "--force-weight",
        str(args.force_weight),
        "--save-path",
        save_path,
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
    ]
    for name, path in args.extra_val_files.items():
        cmd.extend(["--extra-val-file", f"{name}={path}"])
    for key, value in args.dataset_kwargs.items():
        cmd.extend(["--dataset-kwarg", f"{key}={value}"])
    for key, value in args.model_kwargs.items():
        cmd.extend(["--model-kwarg", f"{key}={value}"])
    return cmd


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    local_batch_size = autotune_batch_size(args)
    num_train_samples = resolve_num_train_samples(args)
    steps_per_epoch, epochs = compute_schedule(
        num_train_samples=num_train_samples,
        local_batch_size=local_batch_size,
        target_total_steps=args.target_total_steps,
        world_size=args.nproc_per_node,
    )
    total_steps = steps_per_epoch * epochs
    print(
        f"[launcher] model={args.model_name} local_batch_size={local_batch_size} "
        f"world_size={args.nproc_per_node} steps_per_epoch={steps_per_epoch} "
        f"epochs={epochs} total_steps={total_steps}",
        flush=True,
    )
    cmd = build_torchrun_command(args, local_batch_size=local_batch_size, epochs=epochs)
    print("[launcher] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
