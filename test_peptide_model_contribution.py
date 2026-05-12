"""
Base-model contribution analysis following plot.py logic.
Reuses the proven gradient computation from analyze_conservative_model_contribution.py.
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path(
        "el-mlffs/checkpoints/"
        "el-mlffs/checkpoints/meta_models/conservative_combo/"
        "127_dp_nep_mtp_soap_painn_schnet_mace.pth"
    ))
    p.add_argument("--delivery-root", type=Path, default=Path(
        "el-mlffs/checkpoints"
    ))
    p.add_argument("--data-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path(
        "reports/model_contribution_plotpy_style"
    ))
    p.add_argument("--max-structures", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


# Old data: H, C, O, Cu
TYPE_MAP = {1: "H", 6: "C", 8: "O", 29: "Cu"}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Import from the existing working script
    sys.path.insert(0, str(args.delivery_root / "el-mlffs"))
    from analyze_conservative_model_contribution import (
        load_model,
        build_loader,
        accumulate_sensitivity,
        normalized_rows,
        symbol_for_z,
        write_csv,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    inner_args = argparse.Namespace(
        delivery_root=args.delivery_root,
        checkpoint=args.checkpoint,
        data_file=args.data_file,
        max_structures=args.max_structures,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model, config, base_model_names = load_model(inner_args, device)
    loader, indices = build_loader(inner_args, config, device)

    # Use 1 probe vector with random signs (same principle as plot.py's ones_like,
    # but using the proven accumulate_sensitivity function)
    sensitivity_by_z, counts_by_z, all_sensitivity, mean_gate = accumulate_sensitivity(
        model, loader, base_model_names, device,
        probe_vectors=4,
        seed=args.seed,
    )

    # === plot.py style processing ===
    rows = normalized_rows(sensitivity_by_z, counts_by_z, all_sensitivity, base_model_names)

    # Pivot: rows → (group, base_model) → DataFrame
    records = {}
    for row in rows:
        key = row["group"]
        if key not in records:
            records[key] = {}
        records[key][row["base_model"]] = row["normalized_contribution"]

    df_by_type = pd.DataFrame.from_dict(records, orient="index")
    df_by_type = df_by_type[list(base_model_names)]  # enforce column order

    # Sort: atom types alphabetically, then 'All' last
    atom_labels = sorted([x for x in df_by_type.index if x != "All"])
    new_order = atom_labels + ["All"]
    df_by_type = df_by_type.reindex(new_order)

    print("\n=== Normalized contribution (plot.py style) ===")
    print(df_by_type)

    # Save CSV
    df_by_type.to_csv(args.output_dir / "model_contribution_by_atom_type.csv")

    # Gate weights
    gate_df = pd.DataFrame({
        "base_model": list(base_model_names),
        "mean_gate_weight": mean_gate.numpy(),
    })
    gate_df.to_csv(args.output_dir / "mean_gate_weights.csv", index=False)
    print("\n=== Mean gate weights ===")
    print(gate_df.to_string(index=False))

    # === plot.py style bar chart ===
    sns.set_theme(style="whitegrid", context="talk")
    df_plot = df_by_type.reset_index()
    df_plot = df_plot.melt(
        id_vars="index",
        var_name="Input Model",
        value_name="Normalized Contribution",
    ).rename(columns={"index": "Group"})

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.barplot(
        data=df_plot, x="Group", y="Normalized Contribution",
        hue="Input Model", palette="viridis", ax=ax,
    )

    plt.title("Normalized Contribution of Base Models (plot.py style)", fontsize=22, pad=20)
    plt.ylabel("Contribution Ratio", fontsize=16)
    plt.xlabel("Atom Group", fontsize=16)

    if "All" in new_order:
        idx = new_order.index("All")
        plt.axvline(x=idx - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="Base Models", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(args.output_dir / "model_contribution_plotpy_style.png", dpi=args.dpi)
    print(f"\nSaved: {args.output_dir / 'model_contribution_plotpy_style.png'}")
    plt.close()

    # Heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=args.dpi)
    matrix = df_by_type.values
    im = ax2.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=matrix.max() * 1.2)
    ax2.set_xticks(range(len(base_model_names)))
    ax2.set_xticklabels(list(base_model_names), rotation=35, ha="right")
    ax2.set_yticks(range(len(new_order)))
    ax2.set_yticklabels(new_order)
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    ax2.set_title("Base Model Contribution Heatmap", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(args.output_dir / "model_contribution_heatmap_plotpy_style.png", dpi=args.dpi)
    print(f"Saved: {args.output_dir / 'model_contribution_heatmap_plotpy_style.png'}")
    plt.close()


if __name__ == "__main__":
    main()
