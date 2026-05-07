from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradient-based base-model contribution analysis for the conservative ensemble."
    )
    parser.add_argument(
        "--delivery-root",
        type=Path,
        default=Path("/mnt/bn/changsu-data3/ybc/repos/EL-MLFFs_ybc_delivery_bundle"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/bn/changsu-data3/ybc/repos/EL-MLFFs_ybc_delivery_bundle/"
            "el-mlffs/checkpoints/meta_models/conservative_combo/"
            "127_dp_nep_mtp_soap_painn_schnet_mace.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/bn/bangchen/EL-MLFFs/reports/conservative_model_contribution"),
    )
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--max-structures", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-vectors", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--connected", action="store_true", default=False, help="Analyze connected (2nd-order) gradient path instead of detached")
    parser.add_argument("--base-model-dir", type=Path, default=None, help="Override base model checkpoint directory (auto-inferred from checkpoint path if omitted)")
    parser.add_argument("--architecture", type=str, default=None, choices=["direct", "conservative"], help="Override architecture (auto-detected from checkpoint if omitted)")
    return parser.parse_args()


def setup_imports(delivery_root: Path) -> None:
    module_dir = delivery_root / "el-mlffs"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))


def load_model(args: argparse.Namespace, device: torch.device):
    import train_torch_ensemble as ensemble

    payload = torch.load(args.checkpoint, map_location="cpu")
    config_dict = dict(payload["config"])
    metadata = dict(payload["metadata"])
    base_model_names = ensemble.normalize_base_model_names(config_dict.get("base_model_names"))
    if args.base_model_dir is not None:
        base_model_dir = args.base_model_dir
    else:
        ckpt_str = str(args.checkpoint)
        if "peptide_dft_wb97x_5k" in ckpt_str:
            base_model_dir = args.checkpoint.parent.parent / "peptide_dft_wb97x_5k_base_models"
        else:
            base_model_dir = args.delivery_root / "el-mlffs" / "checkpoints" / "base_models_a100_8gpu_50ksteps"
        if not base_model_dir.exists():
            base_model_dir = args.delivery_root / "el-mlffs" / "checkpoints" / "base_models_a100_8gpu_50ksteps"

    for model_name in base_model_names:
        base_ckpt = base_model_dir / f"{model_name}_torch.pth"
        base_payload = torch.load(base_ckpt, map_location="cpu")
        model_kwargs = dict(base_payload.get("config") or {}).get("model_kwargs") or {}
        ensemble.BASE_MODEL_CONFIGS[model_name] = {"checkpoint": str(base_ckpt), **model_kwargs}

    train_config = ensemble.TrainConfig(
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
        num_workers=0,
        grad_clip_norm=config_dict.get("grad_clip_norm", 10.0),
        huber_delta=config_dict.get("huber_delta", 1.0),
        auto_batch_size=False,
        target_total_steps=config_dict.get("target_total_steps", 20000),
        dataset_backend=config_dict.get("dataset_backend", "extxyz"),
        dataset_kwargs=config_dict.get("dataset_kwargs") or {},
        meta_hidden_scalar_channels=config_dict.get("meta_hidden_scalar_channels", 64),
        meta_hidden_vector_channels=config_dict.get("meta_hidden_vector_channels", 32),
        meta_num_layers=config_dict.get("meta_num_layers", 3),
        meta_num_basis=config_dict.get("meta_num_basis", 32),
        meta_use_residual=config_dict.get("meta_use_residual", False),
        use_zbl=config_dict.get("use_zbl", False),
        zbl_inner=config_dict.get("zbl_inner", 0.60),
        zbl_outer=config_dict.get("zbl_outer", 0.95),
        differentiate_force_features=config_dict.get("differentiate_force_features", False),
    )
    model = ensemble.build_model(train_config, metadata["atomic_numbers"], device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, train_config, tuple(base_model_names)


def build_loader(args: argparse.Namespace, config, device: torch.device):
    from train_distributed import DistributedContext, build_dataloader

    data_file = args.data_file
    if data_file is None:
        if config.data_file and Path(config.data_file).exists():
            data_file = Path(config.data_file)
        else:
            data_file = args.delivery_root / "el-mlffs" / config.val_data_file

    dataset_backend = getattr(config, "dataset_backend", "extxyz")
    dataset_kwargs = getattr(config, "dataset_kwargs", None) or {}

    if dataset_backend == "peptide_dft_lmdb":
        import importlib.util
        import sys
        main_repo_torch_data = Path("/mnt/bn/bangchen/EL-MLFFs/el-mlffs/torch_data.py")
        spec = importlib.util.spec_from_file_location("main_torch_data", str(main_repo_torch_data))
        torch_data_mod = importlib.util.module_from_spec(spec)
        sys.modules["main_torch_data"] = torch_data_mod
        spec.loader.exec_module(torch_data_mod)
        dataset = torch_data_mod.build_dataset(str(data_file), cutoff=config.cutoff, dataset_backend=dataset_backend, dataset_kwargs=dataset_kwargs)
    else:
        from torch_data import build_dataset
        dataset = build_dataset(str(data_file), cutoff=config.cutoff, dataset_backend=dataset_backend, dataset_kwargs=dataset_kwargs)

    sample_count = min(args.max_structures, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    indices.sort()
    subset = Subset(dataset, indices)
    context = DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, device=device)
    loader, _ = build_dataloader(subset, batch_size=args.batch_size, shuffle=False, context=context, num_workers=0)
    return loader, indices


def _forward_with_force_grad_common(model, batch, base_predictions, create_graph: bool, retain_graph: bool):
    from torch_data import energy_to_forces, get_batch_index, global_add_pool, global_mean_pool

    if not batch.pos.requires_grad:
        batch.pos = batch.pos.clone().detach().requires_grad_(True)

    base_energies = torch.nan_to_num(base_predictions["energies"]).detach()
    if not getattr(model, "differentiate_force_features", False):
        base_forces = torch.nan_to_num(base_predictions["forces"]).detach().requires_grad_(True)
    else:
        base_forces = torch.nan_to_num(base_predictions["forces"])
    batch_index = get_batch_index(batch.z, getattr(batch, "batch", None))

    mean_force = base_forces.mean(dim=1)
    force_norm = base_forces.norm(dim=-1)
    force_norm_mean = force_norm.mean(dim=1, keepdim=True)
    force_norm_std = force_norm.std(dim=1, unbiased=False, keepdim=True)

    scalar_features = torch.cat(
        [
            model.atom_embedding(batch.z),
            force_norm,
            force_norm_mean,
            force_norm_std,
        ],
        dim=-1,
    )
    vector_features = torch.cat([base_forces, mean_force.unsqueeze(1)], dim=1)
    encoded = model.encoder(batch, scalar_features, vector_features)
    scalar_node_features = encoded["scalar_features"]
    graph_features = global_mean_pool(scalar_node_features, batch_index)
    gate_logits = model.graph_gate(torch.cat([graph_features, base_energies], dim=-1))
    gate_weights = torch.softmax(gate_logits, dim=-1)

    mixed_energy = torch.sum(gate_weights * base_energies, dim=-1, keepdim=True)
    correction_energy = global_add_pool(model.atomic_correction(scalar_node_features), batch_index)
    total_energy = torch.nan_to_num(mixed_energy + correction_energy)
    forces = energy_to_forces(total_energy, batch.pos, create_graph=create_graph, retain_graph=retain_graph)
    return forces, base_forces, gate_weights


def conservative_forward_with_force_grad(model, batch, base_predictions):
    return _forward_with_force_grad_common(model, batch, base_predictions, create_graph=True, retain_graph=True)


def direct_forward_with_force_grad(model, batch, base_predictions):
    if not getattr(model, "differentiate_force_features", False):
        base_forces = torch.nan_to_num(base_predictions["forces"]).detach().requires_grad_(True)
    else:
        base_forces = torch.nan_to_num(base_predictions["forces"])

    mean_force = base_forces.mean(dim=1)
    force_norm = base_forces.norm(dim=-1)
    force_norm_mean = force_norm.mean(dim=1, keepdim=True)
    force_norm_std = force_norm.std(dim=1, unbiased=False, keepdim=True)

    scalar_features = torch.cat(
        [
            model.atom_embedding(batch.z),
            force_norm,
            force_norm_mean,
            force_norm_std,
        ],
        dim=-1,
    )
    vector_features = torch.cat([base_forces, mean_force.unsqueeze(1)], dim=1)
    encoded = model.encoder(batch, scalar_features, vector_features)
    node_repr = encoded["node_features"]
    forces = model.force_head(node_repr)
    return forces, base_forces, None


def accumulate_sensitivity(
    model,
    loader,
    base_model_names: tuple[str, ...],
    device: torch.device,
    probe_vectors: int,
    seed: int,
    architecture: str,
):
    sensitivity_by_z: dict[int, torch.Tensor] = {}
    counts_by_z: dict[int, int] = {}
    gate_sum = torch.zeros(len(base_model_names), dtype=torch.float64)
    gate_count = 0
    total_atoms = 0
    has_gate_weights = True

    generator = torch.Generator(device=device).manual_seed(seed)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        batch.pos = batch.pos.clone().detach().requires_grad_(True)
        base_predictions = model.base_models(batch, create_graph=True)
        if architecture == "direct":
            forces, base_forces, gate_weights = direct_forward_with_force_grad(model, batch, base_predictions)
        else:
            forces, base_forces, gate_weights = conservative_forward_with_force_grad(model, batch, base_predictions)
        atom_sensitivity = torch.zeros(base_forces.shape[:2], device=device, dtype=torch.float64)
        num_probes = max(1, probe_vectors)
        for probe_idx in range(num_probes):
            random_probe = torch.randint(
                low=0,
                high=2,
                size=forces.shape,
                generator=generator,
                device=device,
                dtype=torch.int8,
            ).to(forces.dtype)
            random_probe = random_probe.mul_(2).sub_(1)
            grads = torch.autograd.grad(
                outputs=forces,
                inputs=base_forces,
                grad_outputs=random_probe,
                retain_graph=probe_idx < num_probes - 1,
                create_graph=False,
                allow_unused=False,
            )[0]
            atom_sensitivity += grads.detach().abs().mean(dim=-1).to(torch.float64)
        atom_sensitivity = (atom_sensitivity / num_probes).cpu()
        z_cpu = batch.z.detach().cpu().to(torch.long)

        for z_value in torch.unique(z_cpu).tolist():
            mask = z_cpu == int(z_value)
            sensitivity_by_z.setdefault(int(z_value), torch.zeros(len(base_model_names), dtype=torch.float64))
            counts_by_z.setdefault(int(z_value), 0)
            sensitivity_by_z[int(z_value)] += atom_sensitivity[mask].sum(dim=0)
            counts_by_z[int(z_value)] += int(mask.sum().item())

        if gate_weights is not None:
            gate_sum += gate_weights.detach().cpu().to(torch.float64).sum(dim=0)
            gate_count += int(gate_weights.size(0))
        else:
            has_gate_weights = False
        total_atoms += int(z_cpu.numel())
        if batch_idx % 25 == 0:
            print(f"processed_batches={batch_idx} | atoms={total_atoms}", flush=True)

    all_sensitivity = sum(sensitivity_by_z.values(), torch.zeros(len(base_model_names), dtype=torch.float64))
    mean_gate = gate_sum / max(gate_count, 1) if has_gate_weights else None
    return sensitivity_by_z, counts_by_z, all_sensitivity, mean_gate


def symbol_for_z(z_value: int) -> str:
    try:
        from ase.data import chemical_symbols

        return chemical_symbols[z_value]
    except Exception:
        return f"Z{z_value}"


def normalized_rows(
    sensitivity_by_z: dict[int, torch.Tensor],
    counts_by_z: dict[int, int],
    all_sensitivity: torch.Tensor,
    base_model_names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    group_items = [(symbol_for_z(z), sensitivity, counts_by_z[z]) for z, sensitivity in sorted(sensitivity_by_z.items())]
    group_items.append(("All", all_sensitivity, sum(counts_by_z.values())))

    for group_name, sensitivity, atom_count in group_items:
        total = float(sensitivity.sum().item())
        for model_name, value in zip(base_model_names, sensitivity.tolist()):
            rows.append(
                {
                    "group": group_name,
                    "atom_count": atom_count,
                    "base_model": model_name,
                    "raw_sensitivity": value,
                    "normalized_contribution": 0.0 if total == 0.0 else value / total,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_contributions(output_dir: Path, rows: list[dict[str, object]], base_model_names: tuple[str, ...], dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    groups = []
    for row in rows:
        group = str(row["group"])
        if group not in groups:
            groups.append(group)
    if "All" in groups:
        groups = [group for group in groups if group != "All"] + ["All"]

    matrix = np.zeros((len(groups), len(base_model_names)), dtype=float)
    for i, group in enumerate(groups):
        group_rows = [row for row in rows if row["group"] == group]
        by_model = {row["base_model"]: float(row["normalized_contribution"]) for row in group_rows}
        matrix[i] = [by_model[name] for name in base_model_names]

    fig, ax = plt.subplots(figsize=(5.8, 2.8), dpi=dpi)
    x = np.arange(len(groups), dtype=float)
    width = 0.11
    offsets = (np.arange(len(base_model_names)) - (len(base_model_names) - 1) / 2.0) * width
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_model_names)))
    for model_idx, model_name in enumerate(base_model_names):
        ax.bar(x + offsets[model_idx], matrix[:, model_idx], width=width, color=colors[model_idx], label=model_name)

    ax.axhline(1.0 / len(base_model_names), color="#444444", linestyle="--", linewidth=1.0, label="equal")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Normalized contribution")
    ax.set_ylim(0.0, max(0.22, matrix.max() * 1.20))
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.55, linestyle=":")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.23))
    fig.tight_layout(pad=0.45)
    fig.savefig(output_dir / "paper_model_contribution_by_atom_type.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "paper_model_contribution_by_atom_type.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.1, 2.25), dpi=dpi)
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.20, matrix.max()))
    ax.set_xticks(np.arange(len(base_model_names)))
    ax.set_xticklabels(base_model_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Contribution", fontsize=7)
    fig.tight_layout(pad=0.45)
    fig.savefig(output_dir / "paper_model_contribution_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "paper_model_contribution_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delivery_root = args.delivery_root.resolve()
    setup_imports(delivery_root)
    device = torch.device(args.device)
    model, config, base_model_names = load_model(args, device)
    architecture = args.architecture or config.architecture
    loader, indices = build_loader(args, config, device)
    sensitivity_by_z, counts_by_z, all_sensitivity, mean_gate = accumulate_sensitivity(
        model,
        loader,
        base_model_names,
        device,
        args.probe_vectors,
        args.seed,
        architecture,
    )
    rows = normalized_rows(sensitivity_by_z, counts_by_z, all_sensitivity, base_model_names)
    write_csv(args.output_dir / "model_contribution_by_atom_type.csv", rows)
    torch.save(
        {
            "sample_indices": indices,
            "base_model_names": base_model_names,
            "sensitivity_by_z": sensitivity_by_z,
            "counts_by_z": counts_by_z,
            "all_sensitivity": all_sensitivity,
            "mean_gate_weights": mean_gate,
            "probe_vectors": args.probe_vectors,
        },
        args.output_dir / "model_contribution_raw.pt",
    )
    if mean_gate is not None:
        gate_rows = [
            {"base_model": model_name, "mean_gate_weight": float(value)}
            for model_name, value in zip(base_model_names, mean_gate.tolist())
        ]
        write_csv(args.output_dir / "mean_gate_weights.csv", gate_rows)
    plot_contributions(args.output_dir, rows, base_model_names, args.dpi)
    print(f"Saved model contribution analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
