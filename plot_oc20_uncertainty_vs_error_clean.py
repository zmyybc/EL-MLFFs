from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


SPLITS = ("val_id", "ood_ads", "ood_cat", "ood_both")
COLORS = {
    "val_id": "#4C78A8",
    "ood_ads": "#0072B2",
    "ood_cat": "#D55E00",
    "ood_both": "#009E73",
}
LABELS = {
    "val_id": "ID",
    "ood_ads": "Adsorbate OOD",
    "ood_cat": "Catalyst OOD",
    "ood_both": "Both OOD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clear OC20 uncertainty-vs-error figures.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/mnt/bn/changsu-data3/ybc/repos/EL-MLFFs_ybc_delivery_bundle/"
            "oc20_ood_eval_runs/uncertainty_vs_conserv_error"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins", type=int, default=20)
    return parser.parse_args()


def load_split(input_dir: Path, split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(input_dir / f"per_structure_{split_name}.pt", map_location="cpu")
    x = payload["component_mean_std"].to(torch.float64)
    y = payload["conserv_force_mae"].to(torch.float64)
    return x, y


def quantile_table(x: torch.Tensor, y: torch.Tensor, bins: int) -> list[dict[str, float]]:
    order = torch.argsort(x)
    rows = []
    n = x.numel()
    for i in range(bins):
        start = round(i * n / bins)
        end = round((i + 1) * n / bins)
        idx = order[start:end]
        xb = x[idx]
        yb = y[idx]
        rows.append(
            {
                "bin": i + 1,
                "count": int(idx.numel()),
                "uncertainty_mean": float(xb.mean()),
                "uncertainty_median": float(xb.median()),
                "error_mean": float(yb.mean()),
                "error_median": float(yb.median()),
                "error_q25": float(torch.quantile(yb, 0.25)),
                "error_q75": float(torch.quantile(yb, 0.75)),
            }
        )
    return rows


def write_quantile_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def annotate_panel(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_split_figure(input_dir: Path, split_name: str, bins: int, dpi: int) -> None:
    plt = setup_matplotlib()
    x, y = load_split(input_dir, split_name)
    rows = quantile_table(x, y, bins)
    write_quantile_csv(input_dir / f"trend_{split_name}.csv", rows)

    x_limit = float(torch.quantile(x, 0.99))
    y_limit = float(torch.quantile(y, 0.99))
    mask = (x <= x_limit) & (y <= y_limit)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=dpi)
    hb = ax.hexbin(
        x[mask].numpy(),
        y[mask].numpy(),
        gridsize=46,
        mincnt=1,
        cmap="Blues",
        linewidths=0,
        bins="log",
    )
    colorbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("log count", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    trend_x = [row["uncertainty_mean"] for row in rows]
    trend_y = [row["error_mean"] for row in rows]
    trend_q25 = [row["error_q25"] for row in rows]
    trend_q75 = [row["error_q75"] for row in rows]
    ax.plot(trend_x, trend_y, color="#111827", linewidth=2.2, marker="o", markersize=3.8, label="Decile mean")
    ax.fill_between(trend_x, trend_q25, trend_q75, color="#111827", alpha=0.14, label="IQR")

    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_xlabel("Base-model force std")
    ax.set_ylabel("Conservative force MAE")
    ax.set_title(f"{LABELS[split_name]}")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(input_dir / f"{split_name}_uncertainty_vs_error_clean.png")
    fig.savefig(input_dir / f"{split_name}_uncertainty_vs_error_clean.pdf")
    plt.close(fig)


def draw_paper_combined_trend(input_dir: Path, bins: int, dpi: int) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.45, 2.65), dpi=dpi)

    for split_name in SPLITS:
        x, y = load_split(input_dir, split_name)
        rows = quantile_table(x, y, bins)
        trend_x = [row["uncertainty_mean"] for row in rows]
        trend_y = [row["error_mean"] for row in rows]
        trend_q25 = [row["error_q25"] for row in rows]
        trend_q75 = [row["error_q75"] for row in rows]
        color = COLORS[split_name]
        ax.plot(
            trend_x,
            trend_y,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=2.8,
            markeredgewidth=0,
            label=LABELS[split_name],
        )
        ax.fill_between(trend_x, trend_q25, trend_q75, color=color, alpha=0.13, linewidth=0)

    ax.set_xlabel("Base-model force standard deviation")
    ax.set_ylabel("Conservative force MAE")
    ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.85)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderpad=0.1)
    fig.tight_layout(pad=0.35)
    fig.savefig(input_dir / "paper_uncertainty_error_trend.pdf", bbox_inches="tight")
    fig.savefig(input_dir / "paper_uncertainty_error_trend.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_paper_density_panels(input_dir: Path, bins: int, dpi: int) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(8.9, 2.25), dpi=dpi, sharex=False, sharey=False)
    panel_labels = ("a", "b", "c", "d")

    for ax, split_name, panel_label in zip(axes, SPLITS, panel_labels):
        x, y = load_split(input_dir, split_name)
        rows = quantile_table(x, y, bins)
        x_limit = float(torch.quantile(x, 0.99))
        y_limit = float(torch.quantile(y, 0.99))
        mask = (x <= x_limit) & (y <= y_limit)

        hb = ax.hexbin(
            x[mask].numpy(),
            y[mask].numpy(),
            gridsize=38,
            mincnt=1,
            cmap="Blues",
            linewidths=0,
            bins="log",
            alpha=0.95,
        )
        trend_x = [row["uncertainty_mean"] for row in rows]
        trend_y = [row["error_mean"] for row in rows]
        ax.plot(trend_x, trend_y, color="#111111", linewidth=1.4, marker="o", markersize=2.3, markeredgewidth=0)

        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, y_limit)
        ax.set_title(LABELS[split_name], pad=3)
        ax.grid(True, color="#dddddd", linewidth=0.4, alpha=0.85)
        annotate_panel(ax, panel_label)
        if ax is axes[0]:
            ax.set_ylabel("Conservative force MAE")
        ax.set_xlabel("Base-model force std")

    colorbar = fig.colorbar(hb, ax=axes.ravel().tolist(), fraction=0.026, pad=0.012)
    colorbar.set_label("log count", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    fig.subplots_adjust(left=0.065, right=0.925, bottom=0.22, top=0.84, wspace=0.34)
    fig.savefig(input_dir / "paper_uncertainty_error_density_panels.pdf", bbox_inches="tight")
    fig.savefig(input_dir / "paper_uncertainty_error_density_panels.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_paper_decile_bars(input_dir: Path, dpi: int) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=dpi)

    width = 0.18
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for offset, split_name in zip(offsets, SPLITS):
        x, y = load_split(input_dir, split_name)
        rows = quantile_table(x, y, bins=10)
        deciles = torch.tensor([row["bin"] for row in rows], dtype=torch.float64)
        values = [row["error_mean"] for row in rows]
        ax.bar((deciles + offset).tolist(), values, width=width, color=COLORS[split_name], alpha=0.86, label=LABELS[split_name])

    ax.set_xlabel("Uncertainty decile")
    ax.set_ylabel("Conservative force MAE")
    ax.set_xticks([1, 5, 10])
    ax.set_xlim(0.4, 10.6)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.85)
    ax.legend(frameon=False, ncol=1, loc="upper left", handlelength=1.3)
    fig.tight_layout(pad=0.35)
    fig.savefig(input_dir / "paper_uncertainty_decile_bars.pdf", bbox_inches="tight")
    fig.savefig(input_dir / "paper_uncertainty_decile_bars.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_combined_trend(input_dir: Path, bins: int, dpi: int) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=dpi)

    for split_name in SPLITS:
        x, y = load_split(input_dir, split_name)
        rows = quantile_table(x, y, bins)
        trend_x = [row["uncertainty_mean"] for row in rows]
        trend_y = [row["error_mean"] for row in rows]
        trend_q25 = [row["error_q25"] for row in rows]
        trend_q75 = [row["error_q75"] for row in rows]
        color = COLORS[split_name]
        ax.plot(trend_x, trend_y, color=color, linewidth=2.1, marker="o", markersize=3.4, label=LABELS[split_name])
        ax.fill_between(trend_x, trend_q25, trend_q75, color=color, alpha=0.12)

    ax.set_xlabel("Base-model force std")
    ax.set_ylabel("Conservative force MAE")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(input_dir / "combined_uncertainty_error_trend.png")
    fig.savefig(input_dir / "combined_uncertainty_error_trend.pdf")
    plt.close(fig)


def draw_decile_bar(input_dir: Path, bins: int, dpi: int) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.2), dpi=dpi, sharey=True)

    for ax, split_name in zip(axes, SPLITS):
        x, y = load_split(input_dir, split_name)
        rows = quantile_table(x, y, bins=10)
        deciles = [row["bin"] for row in rows]
        values = [row["error_mean"] for row in rows]
        ax.bar(deciles, values, color=COLORS[split_name], alpha=0.86, width=0.78)
        ax.set_title(LABELS[split_name])
        ax.set_xlabel("Uncertainty decile")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_xticks([1, 5, 10])

    axes[0].set_ylabel("Conservative force MAE")
    fig.tight_layout()
    fig.savefig(input_dir / "decile_error_bars.png")
    fig.savefig(input_dir / "decile_error_bars.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    for split_name in SPLITS:
        draw_split_figure(input_dir, split_name, args.bins, args.dpi)
    draw_combined_trend(input_dir, args.bins, args.dpi)
    draw_decile_bar(input_dir, args.bins, args.dpi)
    draw_paper_combined_trend(input_dir, args.bins, args.dpi)
    draw_paper_density_panels(input_dir, args.bins, args.dpi)
    draw_paper_decile_bars(input_dir, args.dpi)
    print(f"Saved cleaned figures to: {input_dir}")


if __name__ == "__main__":
    main()
