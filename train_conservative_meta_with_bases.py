from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.join(ROOT_DIR, "el-mlffs")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import train_torch_ensemble as ensemble  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conservative meta model using existing base-model checkpoints.")
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=["dp", "nep", "mtp", "soap", "painn", "schnet", "mace"],
    )
    parser.add_argument(
        "--base-model-dir",
        default=os.path.join(ROOT_DIR, "el-mlffs", "checkpoints", "base_models_a100_8gpu_50ksteps"),
    )
    parser.add_argument("--base-checkpoint-template", default="{model}_torch.pth")
    parser.add_argument("--data-file", default=os.path.join("data", "train.extxyz"))
    parser.add_argument("--val-data-file", default=os.path.join("data", "test.extxyz"))
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=50.0)
    parser.add_argument("--save-path", default=os.path.join("checkpoints", "meta_models", "conservative_meta_current_bases.pth"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-strategy", choices=["random", "ood"], default="random")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--auto-batch-size", action="store_true")
    parser.add_argument("--batch-size-probe-start", type=int, default=4)
    parser.add_argument("--batch-size-probe-step", type=int, default=1)
    parser.add_argument("--batch-size-probe-max", type=int, default=64)
    parser.add_argument("--memory-target-ratio", type=float, default=0.8)
    parser.add_argument("--target-total-steps", type=int, default=20000)
    parser.add_argument("--train-base-models", action="store_true")
    parser.add_argument("--dataset-backend", choices=["extxyz", "oc20_lmdb", "peptide_dft_lmdb"], default="extxyz")
    parser.add_argument("--dataset-kwarg", action="append", default=[], help="Extra dataset kwargs as key=value.")
    parser.add_argument(
        "--differentiate-force-features",
        action="store_true",
        help="Keep base-force feature gradients connected for strict conservative training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model_names = ensemble.normalize_base_model_names(args.base_models)
    checkpoint_template = args.base_checkpoint_template
    if "{model}" not in checkpoint_template:
        raise ValueError(
            "--base-checkpoint-template must contain the literal placeholder '{model}', "
            f"got: {checkpoint_template!r}"
        )

    for model_name in base_model_names:
        checkpoint_path = os.path.join(
            args.base_model_dir,
            checkpoint_template.replace("{model}", model_name),
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing base-model checkpoint: {checkpoint_path}")
        ensemble.BASE_MODEL_CONFIGS[model_name] = {"checkpoint": checkpoint_path}

    dataset_kwargs: dict[str, int | float | bool | str] = {}
    for item in args.dataset_kwarg:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-kwarg '{item}'. Expected key=value.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        lower_value = raw_value.lower()
        if lower_value in {"true", "false"}:
            value: int | float | bool | str = lower_value == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        dataset_kwargs[key] = value

    config = ensemble.TrainConfig(
        data_file=args.data_file,
        val_data_file=args.val_data_file,
        architecture="conservative",
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        freeze_base_models=not args.train_base_models,
        save_path=args.save_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_strategy=args.split_strategy,
        base_model_names=tuple(base_model_names),
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
        huber_delta=args.huber_delta,
        auto_batch_size=args.auto_batch_size,
        batch_size_probe_start=args.batch_size_probe_start,
        batch_size_probe_step=args.batch_size_probe_step,
        batch_size_probe_max=args.batch_size_probe_max,
        memory_target_ratio=args.memory_target_ratio,
        target_total_steps=args.target_total_steps,
        dataset_backend=args.dataset_backend,
        dataset_kwargs=dataset_kwargs,
        differentiate_force_features=args.differentiate_force_features,
    )
    ensemble.run_training(config)


if __name__ == "__main__":
    main()
