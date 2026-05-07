from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ensemble size vs validation accuracy using updated combo metrics."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("/mnt/bn/bangchen/EL-MLFFs/reports/conservative_combo_metrics/metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/bn/bangchen/EL-MLFFs/reports/conservative_combo_metrics"),
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def grouped_values(df: pd.DataFrame, metric: str) -> list[np.ndarray]:
    values = []
    for size in range(1, 8):
        group = df.loc[df["num_models"] == size, metric].astype(float).to_numpy()
        values.append(group)
    return values


def plot_panel(ax, df: pd.DataFrame, metric: str, ylabel: str, title: str, mean_color: str) -> None:
    positions = np.arange(1, 8, dtype=float)
    values = grouped_values(df, metric)
    means = np.array([v.mean() for v in values], dtype=float)
    best = np.array([v.min() for v in values], dtype=float)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1f2937", "linewidth": 1.1},
        boxprops={"facecolor": "#D8E8F7", "edgecolor": "#24476B", "linewidth": 1.0},
        whiskerprops={"color": "#24476B", "linewidth": 0.95},
        capprops={"color": "#24476B", "linewidth": 0.95},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.85)

    rng = np.random.default_rng(42)
    for size, group in zip(positions, values):
        jitter = rng.normal(loc=0.0, scale=0.035, size=group.size)
        ax.scatter(
            np.full(group.size, size) + jitter,
            group,
            s=15,
            color="#4B759D",
            alpha=0.38,
            linewidths=0,
            zorder=2,
        )

    ax.plot(
        positions,
        means,
        color=mean_color,
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        label="Mean",
        zorder=4,
    )
    ax.plot(
        positions,
        best,
        color="#C1121F",
        linewidth=1.5,
        linestyle="--",
        marker="D",
        markersize=3.8,
        label="Best",
        zorder=5,
    )

    ax.set_title(title, pad=5)
    ax.set_xlabel("Number of base models in ensemble")
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xlim(0.5, 7.5)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.65, linestyle=":", alpha=0.95)
    ax.legend(frameon=False, loc="upper right", handlelength=2.1)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), dpi=args.dpi)
    plot_panel(
        axes[0],
        df,
        "val_force_mae",
        "Validation Force MAE",
        "Ensemble Size vs Force Accuracy",
        "#17365D",
    )
    plot_panel(
        axes[1],
        df,
        "val_energy_mae",
        "Validation Energy MAE",
        "Ensemble Size vs Energy Accuracy",
        "#2A9D8F",
    )
    fig.suptitle("Relationship Between Ensemble Size and Validation Accuracy", fontsize=12, y=1.03)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(output_dir / "paper_ensemble_size_accuracy_reference_style.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "paper_ensemble_size_accuracy_reference_style.png", dpi=args.dpi, bbox_inches="tight")

    fig2, ax = plt.subplots(figsize=(3.45, 2.55), dpi=args.dpi)
    plot_panel(
        ax,
        df,
        "val_force_mae",
        "Validation Force MAE",
        "",
        "#17365D",
    )
    ax.set_title("")
    fig2.tight_layout(pad=0.35)
    fig2.savefig(output_dir / "paper_ensemble_size_force_reference_style.pdf", bbox_inches="tight")
    fig2.savefig(output_dir / "paper_ensemble_size_force_reference_style.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved reference-style ensemble-size figures to: {output_dir}")


if __name__ == "__main__":
    main()
