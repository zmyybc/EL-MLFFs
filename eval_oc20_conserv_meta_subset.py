from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OC20 conservative meta checkpoint on the same sampled OOD subsets."
    )
    parser.add_argument(
        "--delivery-root",
        type=Path,
        default=Path("./"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "./"
            "el-mlffs/checkpoints/oc20_meta_models/oc20_conserv_7model.pth"
        ),
    )
    parser.add_argument(
        "--sample-cache-dir",
        type=Path,
        default=Path(
            "./oc20_ood_eval_runs"
        ),
        help="Directory containing base-model force cache files with sample_indices.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["ood_ads", "ood_cat", "ood_both"],
        choices=["val_id", "ood_ads", "ood_cat", "ood_both"],
    )
    parser.add_argument(
        "--sample-indices-file",
        action="append",
        default=[],
        help="Optional override as split=path_to_pt_or_txt. If provided, use these indices instead of reading from base force cache.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix appended to output/cache filenames, e.g. half1 or gpu0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "./"
            "oc20_ood_eval_runs/oc20_conserv_meta_ood_metrics_100k.tsv"
        ),
    )
    parser.add_argument(
        "--save-force-dir",
        type=Path,
        default=Path(
            "./"
            "oc20_ood_eval_runs/conserv_force_cache"
        ),
    )
    return parser.parse_args()


def split_dir_name(split_name: str) -> str:
    return {
        "val_id": "val_id",
        "ood_ads": "val_ood_ads",
        "ood_cat": "val_ood_cat",
        "ood_both": "val_ood_both",
    }[split_name]


def load_sample_indices(sample_cache_dir: Path, split_name: str) -> list[int]:
    candidates = sorted(sample_cache_dir.glob(f"force_cache_gpu*/*_{split_name}_forces.pt"))
    if not candidates:
        candidates = sorted(sample_cache_dir.glob(f"*_{split_name}_forces.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any cached base-model force file for split={split_name} under {sample_cache_dir}"
        )
    payload = torch.load(candidates[0], map_location="cpu")
    indices = payload.get("sample_indices")
    if indices is None:
        raise KeyError(f"sample_indices missing from {candidates[0]}")
    return list(indices)


def parse_sample_indices_overrides(items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --sample-indices-file '{item}'. Expected split=path.")
        split_name, raw_path = item.split("=", 1)
        split_name = split_name.strip()
        raw_path = raw_path.strip()
        if not split_name or not raw_path:
            raise ValueError(f"Invalid --sample-indices-file '{item}'. Expected split=path.")
        parsed[split_name] = Path(raw_path).resolve()
    return parsed


def load_indices_from_file(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing sample indices file: {path}")
    if path.suffix == ".txt":
        return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "sample_indices" in payload:
        return list(payload["sample_indices"])
    if isinstance(payload, (list, tuple)):
        return [int(x) for x in payload]
    if torch.is_tensor(payload):
        return [int(x) for x in payload.tolist()]
    raise ValueError(f"Unsupported sample indices payload in {path}")


def build_model_from_checkpoint(checkpoint_path: Path, delivery_root: Path, device: torch.device):
    module_dir = delivery_root / "el-mlffs"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    import train_torch_ensemble as ensemble

    payload = torch.load(checkpoint_path, map_location="cpu")
    config_dict = dict(payload.get("config") or {})
    metadata = dict(payload.get("metadata") or {})
    base_model_names = ensemble.normalize_base_model_names(
        config_dict.get("base_model_names") or metadata.get("base_model_names")
    )

    base_model_dir = delivery_root / "el-mlffs" / "checkpoints" / "oc20_base_models"
    for model_name in base_model_names:
        base_ckpt = base_model_dir / model_name / f"{model_name}_compact.pth"
        base_payload = torch.load(base_ckpt, map_location="cpu")
        model_kwargs = dict(base_payload.get("config") or {}).get("model_kwargs") or {}
        ensemble.BASE_MODEL_CONFIGS[model_name] = {"checkpoint": str(base_ckpt), **model_kwargs}

    config = ensemble.TrainConfig(
        data_file=config_dict["data_file"],
        val_data_file=config_dict.get("val_data_file"),
        architecture=config_dict["architecture"],
        cutoff=config_dict["cutoff"],
        batch_size=config_dict["batch_size"],
        epochs=config_dict["epochs"],
        lr=config_dict["lr"],
        min_lr=config_dict["min_lr"],
        energy_weight=config_dict["energy_weight"],
        force_weight=config_dict["force_weight"],
        freeze_base_models=config_dict["freeze_base_models"],
        save_path=config_dict["save_path"],
        train_ratio=config_dict["train_ratio"],
        seed=config_dict["seed"],
        split_strategy=config_dict["split_strategy"],
        base_model_names=tuple(base_model_names),
        num_workers=config_dict["num_workers"],
        grad_clip_norm=config_dict["grad_clip_norm"],
        huber_delta=config_dict["huber_delta"],
        auto_batch_size=config_dict["auto_batch_size"],
        batch_size_probe_start=config_dict["batch_size_probe_start"],
        batch_size_probe_step=config_dict["batch_size_probe_step"],
        batch_size_probe_max=config_dict["batch_size_probe_max"],
        memory_target_ratio=config_dict["memory_target_ratio"],
        target_total_steps=config_dict["target_total_steps"],
        dataset_backend=config_dict["dataset_backend"],
        dataset_kwargs=config_dict.get("dataset_kwargs"),
    )
    all_z = metadata["atomic_numbers"]
    model = ensemble.build_model(config, all_z, device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, config


def evaluate_and_collect_forces(model: torch.nn.Module, loader, device: torch.device) -> tuple[float, float, dict[str, torch.Tensor]]:
    model.eval()
    total_energy_mae = 0.0
    total_force_mae = 0.0
    total_graphs = 0

    pred_forces_chunks = []
    target_forces_chunks = []
    natoms_chunks = []

    progress = tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False)
    for batch in progress:
        batch = batch.to(device)
        outputs = model(batch)

        pred_forces = outputs["forces"].detach().cpu()
        target_forces = batch.forces.detach().cpu()
        natoms = torch.bincount(batch.batch.detach().cpu(), minlength=batch.num_graphs)

        pred_forces_chunks.append(pred_forces)
        target_forces_chunks.append(target_forces)
        natoms_chunks.append(natoms)

        total_energy_mae += F.l1_loss(outputs["energy"].view(-1), batch.energy.view(-1)).item() * batch.num_graphs
        total_force_mae += F.l1_loss(outputs["forces"], batch.forces).item() * batch.num_graphs
        total_graphs += batch.num_graphs

        progress.set_postfix(
            force=f"{(total_force_mae / max(total_graphs,1)):.4f}",
            energy=f"{(total_energy_mae / max(total_graphs,1)):.4f}",
            graphs=int(total_graphs),
        )
    progress.close()

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

    from torch_data import build_dataset
    from train_distributed import DistributedContext, build_dataloader

    device = torch.device(args.device)
    context = DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, device=device)

    output_path = args.output.resolve()
    force_dir = args.save_force_dir.resolve()
    sample_indices_overrides = parse_sample_indices_overrides(args.sample_indices_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    force_dir.mkdir(parents=True, exist_ok=True)

    tag_suffix = f"_{args.tag}" if args.tag else ""

    model, config = build_model_from_checkpoint(args.checkpoint.resolve(), delivery_root, device)
    data_root = delivery_root / "data" / "oc20" / "processed_pyg"

    rows: list[dict[str, object]] = []

    for split_name in args.splits:
        if split_name in sample_indices_overrides:
            sample_indices = load_indices_from_file(sample_indices_overrides[split_name])
        else:
            sample_indices = load_sample_indices(args.sample_cache_dir.resolve(), split_name)
        split_path = data_root / split_dir_name(split_name)
        dataset = build_dataset(
            str(split_path),
            cutoff=config.cutoff,
            dataset_backend=config.dataset_backend,
            dataset_kwargs=dict(config.dataset_kwargs or {}),
        )
        sampled_dataset = Subset(dataset, sample_indices)
        loader, _ = build_dataloader(
            sampled_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            context=context,
            num_workers=args.num_workers,
        )
        print(
            f"Evaluating split={split_name} | samples={len(sample_indices)} | batch_size={args.batch_size}",
            flush=True,
        )
        energy_mae, force_mae, packed_forces = evaluate_and_collect_forces(model, loader, device)
        force_path = force_dir / f"oc20_conserv_7model_{split_name}{tag_suffix}_forces.pt"
        torch.save(
            {
                "model": "oc20_conserv_7model",
                "split": split_name,
                "tag": args.tag,
                "sample_count": len(sample_indices),
                "sample_indices": sample_indices,
                "pred_forces": packed_forces["pred_forces"],
                "target_forces": packed_forces["target_forces"],
                "natoms": packed_forces["natoms"],
                "checkpoint": str(args.checkpoint.resolve()),
            },
            force_path,
        )
        rows.append(
            {
                "model": "oc20_conserv_7model",
                "split": split_name,
                "sample_count": len(sample_indices),
                "energy_mae": energy_mae,
                "force_mae": force_mae,
                "checkpoint": str(args.checkpoint.resolve()),
                "force_cache": str(force_path),
            }
        )
        print(
            f"oc20_conserv_7model\t{split_name}\tsamples={len(sample_indices)}\t"
            f"energy_mae={energy_mae:.6f}\tforce_mae={force_mae:.6f}\tsaved={force_path}",
            flush=True,
        )

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
