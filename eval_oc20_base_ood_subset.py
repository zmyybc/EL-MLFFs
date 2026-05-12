from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OC20 base-model checkpoints on sampled OOD subsets."
    )
    parser.add_argument(
        "--delivery-root",
        type=Path,
        default=Path("./"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dp", "nep", "mtp", "soap", "painn", "schnet", "mace"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["ood_ads", "ood_cat", "ood_both"],
        choices=["val_id", "ood_ads", "ood_cat", "ood_both"],
    )
    parser.add_argument("--samples-per-split", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        required=True, help="Path to OC20 base OOD metrics TSV",
    )
    parser.add_argument(
        "--save-force-dir",
        type=Path,
        required=True, help="Path to OC20 base OOD force cache",
    )
    return parser.parse_args()


def split_dir_name(split_name: str) -> str:
    return {
        "val_id": "val_id",
        "ood_ads": "val_ood_ads",
        "ood_cat": "val_ood_cat",
        "ood_both": "val_ood_both",
    }[split_name]


def stable_split_seed(base_seed: int, split_name: str) -> int:
    digest = hashlib.sha1(split_name.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def sample_subset(dataset, sample_count: int, seed: int):
    if sample_count <= 0 or sample_count >= len(dataset):
        indices = list(range(len(dataset)))
        return dataset, len(dataset), indices
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    # Keep shard-local access as sequential as possible for oc20_pyg.
    indices.sort()
    return Subset(dataset, indices), sample_count, indices


def evaluate_and_collect_forces(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float, dict[str, torch.Tensor]]:
    model.eval()
    total_energy_mae = 0.0
    total_force_mae = 0.0
    total_graphs = 0

    pred_forces_chunks = []
    target_forces_chunks = []
    natoms_chunks = []

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch, compute_forces=True, create_graph=False)

        pred_forces = outputs["forces"].detach().cpu()
        target_forces = batch.forces.detach().cpu()
        natoms = torch.bincount(batch.batch.detach().cpu(), minlength=batch.num_graphs)

        pred_forces_chunks.append(pred_forces)
        target_forces_chunks.append(target_forces)
        natoms_chunks.append(natoms)

        total_energy_mae += F.l1_loss(outputs["energy"].view(-1), batch.energy.view(-1)).item() * batch.num_graphs
        total_force_mae += F.l1_loss(outputs["forces"], batch.forces).item() * batch.num_graphs
        total_graphs += batch.num_graphs

    packed = {
        "pred_forces": torch.cat(pred_forces_chunks, dim=0),
        "target_forces": torch.cat(target_forces_chunks, dim=0),
        "natoms": torch.cat(natoms_chunks, dim=0),
    }
    return total_energy_mae / total_graphs, total_force_mae / total_graphs, packed


def main() -> None:
    args = parse_args()
    delivery_root = args.delivery_root.resolve()
    module_dir = delivery_root / "el-mlffs"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    from torch_base_models import BASE_MODEL_REGISTRY
    from torch_data import build_dataset
    from train_distributed import DistributedContext, build_dataloader

    device = torch.device(args.device)
    context = DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )

    ckpt_root = delivery_root / "el-mlffs" / "checkpoints" / "oc20_base_models"
    data_root = delivery_root / "data" / "oc20" / "processed_pyg"
    output_path = args.output.resolve()
    force_dir = args.save_force_dir.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    force_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for model_name in args.models:
        ckpt_path = ckpt_root / model_name / f"{model_name}_compact.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu")
        config = payload["config"]
        metadata = payload["metadata"]
        model_kwargs = dict(config.get("model_kwargs") or {})
        dataset_kwargs = dict(config.get("dataset_kwargs") or {})

        model_cls = BASE_MODEL_REGISTRY[model_name]
        if model_name in {"dp", "nep", "mtp", "soap"}:
            model = model_cls(
                z_list=metadata["atomic_numbers"],
                cutoff=config["cutoff"],
                **model_kwargs,
            ).to(device)
        else:
            model = model_cls(
                cutoff=config["cutoff"],
                **model_kwargs,
            ).to(device)

        model.load_state_dict(payload["state_dict"], strict=False)
        model.eval()

        for split_name in args.splits:
            split_path = data_root / split_dir_name(split_name)
            dataset = build_dataset(
                str(split_path),
                cutoff=config["cutoff"],
                dataset_backend=config["dataset_backend"],
                dataset_kwargs=dataset_kwargs,
            )
            sampled_dataset, sample_count, sample_indices = sample_subset(
                dataset,
                args.samples_per_split,
                stable_split_seed(args.seed, split_name),
            )
            loader, _ = build_dataloader(
                sampled_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                context=context,
                num_workers=args.num_workers,
            )
            energy_mae, force_mae, packed_forces = evaluate_and_collect_forces(model, loader, device)
            force_path = force_dir / f"{model_name}_{split_name}_forces.pt"
            torch.save(
                {
                    "model": model_name,
                    "split": split_name,
                    "sample_count": sample_count,
                    "sample_indices": sample_indices,
                    "pred_forces": packed_forces["pred_forces"],
                    "target_forces": packed_forces["target_forces"],
                    "natoms": packed_forces["natoms"],
                    "checkpoint": str(ckpt_path),
                },
                force_path,
            )
            row = {
                "model": model_name,
                "split": split_name,
                "sample_count": sample_count,
                "energy_mae": energy_mae,
                "force_mae": force_mae,
                "checkpoint": str(ckpt_path),
                "force_cache": str(force_path),
            }
            rows.append(row)
            print(
                f"{model_name}\t{split_name}\tsamples={sample_count}\t"
                f"energy_mae={energy_mae:.6f}\tforce_mae={force_mae:.6f}\t"
                f"saved={force_path}",
                flush=True,
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "split", "sample_count", "energy_mae", "force_mae", "checkpoint", "force_cache"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved metrics to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
