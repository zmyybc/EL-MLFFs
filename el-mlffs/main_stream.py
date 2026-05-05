from __future__ import annotations

import argparse
import os

from train_torch_ensemble import DEFAULT_BASE_MODEL_NAMES, TrainConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility entrypoint for random-split ensemble training.")
    parser.add_argument("--architecture", choices=["direct", "conservative"], default="direct")
    parser.add_argument("--data-file", default="data/train.extxyz")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=50.0)
    parser.add_argument("--save-path", default="models/main_stream_torch.pth")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-base-models", action="store_true")
    parser.add_argument("--base-models", nargs="+", default=list(DEFAULT_BASE_MODEL_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    config = TrainConfig(
        data_file=args.data_file,
        architecture=args.architecture,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        freeze_base_models=not args.train_base_models,
        save_path=args.save_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_strategy="random",
        base_model_names=tuple(args.base_models),
        num_workers=args.num_workers,
    )
    run_training(config)


if __name__ == "__main__":
    main()
