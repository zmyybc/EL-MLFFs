from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(ROOT_DIR, "el-mlffs")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from train_torch_base import BaseTrainConfig, build_model, parse_key_value_items, parse_named_paths, run_training  # noqa: E402
from torch_data import build_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune batch size and train one base model to a target total step count.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-backend", choices=["extxyz", "oc20_lmdb", "peptide_dft_lmdb"], default="extxyz")
    parser.add_argument("--data-file", default=os.path.join("el-mlffs", "data", "train.extxyz"))
    parser.add_argument("--val-data-file", default=os.path.join("el-mlffs", "data", "test.extxyz"))
    parser.add_argument("--extra-val-file", action="append", default=[], help="Extra validation sets as name=path.")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=50.0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--initial-batch-size", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--target-memory-fraction", type=float, default=0.85)
    parser.add_argument("--memory-headroom-fraction", type=float, default=0.05)
    parser.add_argument("--target-total-steps", type=int, default=50000)
    parser.add_argument("--world-size", type=int, default=1, help="Planned number of distributed workers for schedule calculation.")
    parser.add_argument("--autotune-only", action="store_true", help="Only print the tuned local batch size and exit.")
    parser.add_argument("--output-dir", default=os.path.join("el-mlffs", "checkpoints", "base_models_a100_8gpu_50ksteps"))
    parser.add_argument("--dataset-kwarg", action="append", default=[], help="Extra dataset kwargs as key=value.")
    parser.add_argument("--model-kwarg", action="append", default=[], help="Extra model constructor kwargs as key=value.")
    parser.add_argument("--resume-from", help="Load initial model weights from an existing checkpoint.")
    args = parser.parse_args()
    args.extra_val_files = parse_named_paths(args.extra_val_file)
    args.dataset_kwargs = parse_key_value_items(args.dataset_kwarg)
    args.model_kwargs = parse_key_value_items(args.model_kwarg)
    return args


def is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def probe_batch_size(
    dataset,
    config: BaseTrainConfig,
    batch_size: int,
    device: torch.device,
) -> tuple[bool, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader)).to(device)
    model = build_model(config, dataset, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    torch.cuda.reset_peak_memory_stats(device)
    try:
        outputs = model(batch, compute_forces=True, create_graph=True)
        loss_e = F.mse_loss(outputs["energy"].view(-1), batch.energy.view(-1))
        loss_f = F.mse_loss(outputs["forces"], batch.forces)
        loss = config.energy_weight * loss_e + config.force_weight * loss_f
        loss.backward()
        optimizer.step()
        peak_fraction = torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
        return True, peak_fraction
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        return False, math.inf
    finally:
        del loader, batch, model, optimizer
        torch.cuda.empty_cache()


def autotune_batch_size(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for autotuning on the target multi-GPU machine.")

    device = torch.device("cuda")
    dataset = build_dataset(
        args.data_file,
        cutoff=args.cutoff,
        dataset_backend=args.dataset_backend,
        dataset_kwargs=args.dataset_kwargs,
    )
    probe_config = BaseTrainConfig(
        model_name=args.model_name,
        dataset_backend=args.dataset_backend,
        data_file=args.data_file,
        val_data_file=args.val_data_file,
        extra_val_files=dict(args.extra_val_files),
        cutoff=args.cutoff,
        batch_size=args.initial_batch_size,
        epochs=1,
        lr=args.lr,
        min_lr=args.min_lr,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=0,
        dataset_kwargs=dict(args.dataset_kwargs),
        model_kwargs=dict(args.model_kwargs),
        resume_from=args.resume_from,
    )

    effective_target_fraction = max(0.05, min(0.98, args.target_memory_fraction - args.memory_headroom_fraction))
    print(
        f"[autotune] model={args.model_name} target_memory_fraction={args.target_memory_fraction:.3f} "
        f"effective_target_fraction={effective_target_fraction:.3f}",
        flush=True,
    )

    low = 0
    high = None
    batch_size = max(1, args.initial_batch_size)
    best_ok = 1
    best_under_target = 0

    while batch_size <= args.max_batch_size:
        ok, peak_fraction = probe_batch_size(dataset, probe_config, batch_size, device)
        if not ok:
            high = batch_size
            break
        print(
            f"[autotune] model={args.model_name} batch_size={batch_size} "
            f"peak_memory_fraction={peak_fraction:.3f}",
            flush=True,
        )
        best_ok = batch_size
        if peak_fraction <= effective_target_fraction:
            best_under_target = batch_size
            low = batch_size
            batch_size *= 2
            continue
        high = batch_size
        break

    if high is None:
        return best_under_target if best_under_target > 0 else best_ok

    left = low + 1
    right = high - 1
    while left <= right:
        mid = (left + right) // 2
        ok, peak_fraction = probe_batch_size(dataset, probe_config, mid, device)
        if ok:
            print(
                f"[autotune] model={args.model_name} batch_size={mid} "
                f"peak_memory_fraction={peak_fraction:.3f}",
                flush=True,
            )
            best_ok = mid
            if peak_fraction <= effective_target_fraction:
                best_under_target = max(best_under_target, mid)
                left = mid + 1
            else:
                right = mid - 1
        else:
            right = mid - 1

    if best_under_target > 0:
        return best_under_target
    if best_ok > 1:
        fallback = max(1, best_ok // 2)
        print(
            f"[autotune] no candidate stayed under the effective target; fallback batch_size={fallback}",
            flush=True,
        )
        return fallback
    return best_ok


def resolve_num_train_samples(args: argparse.Namespace) -> int:
    dataset = build_dataset(
        args.data_file,
        cutoff=args.cutoff,
        dataset_backend=args.dataset_backend,
        dataset_kwargs=args.dataset_kwargs,
    )
    if args.val_data_file:
        return len(dataset)
    return max(1, int(len(dataset) * args.train_ratio))


def compute_schedule(num_train_samples: int, local_batch_size: int, target_total_steps: int, world_size: int) -> tuple[int, int]:
    global_batch = max(1, local_batch_size * max(1, world_size))
    steps_per_epoch = max(1, math.ceil(num_train_samples / global_batch))
    epochs = max(1, math.ceil(target_total_steps / steps_per_epoch))
    return steps_per_epoch, epochs


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_kwargs:
        print(f"Model kwargs for {args.model_name}: {args.model_kwargs}", flush=True)
    if args.dataset_kwargs:
        print(f"Dataset kwargs: {args.dataset_kwargs}", flush=True)

    tuned_batch_size = autotune_batch_size(args)
    print(f"Selected batch size for {args.model_name}: {tuned_batch_size}", flush=True)
    if args.autotune_only:
        print(tuned_batch_size, flush=True)
        return

    num_train_samples = resolve_num_train_samples(args)
    steps_per_epoch, epochs = compute_schedule(
        num_train_samples=num_train_samples,
        local_batch_size=tuned_batch_size,
        target_total_steps=args.target_total_steps,
        world_size=args.world_size,
    )
    total_steps = epochs * steps_per_epoch
    print(
        f"Training schedule for {args.model_name}: "
        f"train_samples={num_train_samples} | world_size={args.world_size} | "
        f"steps_per_epoch={steps_per_epoch} | epochs={epochs} | total_steps={total_steps}",
        flush=True,
    )

    config = BaseTrainConfig(
        model_name=args.model_name,
        dataset_backend=args.dataset_backend,
        data_file=args.data_file,
        val_data_file=args.val_data_file,
        extra_val_files=dict(args.extra_val_files),
        cutoff=args.cutoff,
        batch_size=tuned_batch_size,
        epochs=epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        save_path=os.path.join(args.output_dir, f"{args.model_name}_torch.pth"),
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        dataset_kwargs=dict(args.dataset_kwargs),
        model_kwargs=dict(args.model_kwargs),
    )
    run_training(config)


if __name__ == "__main__":
    main()
