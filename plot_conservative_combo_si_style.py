from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SI-style combinatorial ensemble evaluation heatmaps."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("./reports/conservative_combo_metrics/metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./reports/conservative_combo_metrics"),
    )
    parser.add_argument("--metric", choices=["val_force_mae", "val_energy_mae"], default="val_force_mae")
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def load_metrics(path: Path, metric: str) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            size = int(row["num_models"])
            value = float(row[metric])
            grouped[size].append(
                {
                    "size": size,
                    "value": value,
                    "combo": row["combo_tag"],
                    "checkpoint": row["checkpoint_name"],
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda item: float(item["value"]))
    return dict(sorted(grouped.items()))


def setup_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

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
    cmap = LinearSegmentedColormap.from_list(
        "force_mae_low_dark",
        ["#08306B", "#2171B5", "#6BAED6", "#C6DBEF", "#F7FBFF"],
    )
    cmap.set_bad(color="#f0f0f0")
    return plt, cmap


def build_heatmap(grouped: dict[int, list[dict[str, object]]]) -> np.ndarray:
    sizes = list(grouped)
    max_count = max(len(grouped[size]) for size in sizes)
    matrix = np.full((max_count, len(sizes)), np.nan, dtype=float)
    for col, size in enumerate(sizes):
        values = [float(item["value"]) for item in grouped[size]]
        matrix[: len(values), col] = values
    return matrix


def summarize(grouped: dict[int, list[dict[str, object]]]) -> list[dict[str, object]]:
    rows = []
    for size, items in grouped.items():
        values = np.array([float(item["value"]) for item in items], dtype=float)
        best_item = min(items, key=lambda item: float(item["value"]))
        rows.append(
            {
                "size": size,
                "count": len(items),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "best": float(values.min()),
                "worst": float(values.max()),
                "best_combo": str(best_item["combo"]),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["size", "count", "mean", "std", "best", "worst", "best_combo"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_si_style(grouped: dict[int, list[dict[str, object]]], output_dir: Path, metric: str, dpi: int) -> None:
    plt, cmap = setup_matplotlib()
    sizes = list(grouped)
    matrix = build_heatmap(grouped)
    summary = summarize(grouped)

    valid_values = matrix[np.isfinite(matrix)]
    vmin = float(np.quantile(valid_values, 0.02))
    vmax = float(np.quantile(valid_values, 0.98))

    fig = plt.figure(figsize=(6.85, 3.2), dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.34)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_line = fig.add_subplot(grid[0, 1])

    image = ax_heat.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_heat.set_xticks(np.arange(len(sizes)))
    ax_heat.set_xticklabels([f"G{size}" for size in sizes])
    ax_heat.set_xlabel("Ensemble size")
    ax_heat.set_ylabel("Subsets sorted by force MAE")
    ax_heat.set_yticks([0, matrix.shape[0] - 1])
    ax_heat.set_yticklabels(["best", "worst"])
    ax_heat.tick_params(axis="y", length=0)

    for col, size in enumerate(sizes):
        count = len(grouped[size])
        ax_heat.text(col, -4.0, f"n={count}", ha="center", va="bottom", fontsize=6.5, clip_on=False)
        if count < matrix.shape[0]:
            ax_heat.axhline(count - 0.5, xmin=col / len(sizes), xmax=(col + 1) / len(sizes), color="white", linewidth=0.8)

    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.025)
    colorbar.set_label("Force MAE", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)

    xs = np.array([row["size"] for row in summary], dtype=float)
    means = np.array([row["mean"] for row in summary], dtype=float)
    stds = np.array([row["std"] for row in summary], dtype=float)
    best = np.array([row["best"] for row in summary], dtype=float)
    ax_line.plot(xs, means, color="#4D4D4D", marker="o", markersize=3.2, linewidth=1.4, label="mean")
    ax_line.fill_between(xs, means - stds, means + stds, color="#4D4D4D", alpha=0.16, linewidth=0, label="±1 std")
    ax_line.plot(xs, best, color="#0072B2", marker="s", markersize=3.0, linewidth=1.4, label="best")
    ax_line.set_xlabel("Ensemble size")
    ax_line.set_ylabel("Force MAE")
    ax_line.set_xticks(xs)
    ax_line.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.85)
    ax_line.legend(frameon=False, loc="upper right", handlelength=1.5)

    ax_heat.text(-0.20, 1.08, "a", transform=ax_heat.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")
    ax_line.text(-0.22, 1.08, "b", transform=ax_line.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")

    fig.savefig(output_dir / "paper_conservative_combo_si_style.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "paper_conservative_combo_si_style.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_only(grouped: dict[int, list[dict[str, object]]], output_dir: Path, metric: str, dpi: int) -> None:
    plt, cmap = setup_matplotlib()
    sizes = list(grouped)
    matrix = build_heatmap(grouped)
    valid_values = matrix[np.isfinite(matrix)]
    vmin = float(np.quantile(valid_values, 0.02))
    vmax = float(np.quantile(valid_values, 0.98))

    fig, ax = plt.subplots(figsize=(3.45, 2.85), dpi=dpi)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([f"G{size}" for size in sizes])
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Subsets sorted by force MAE")
    ax.set_yticks([0, matrix.shape[0] - 1])
    ax.set_yticklabels(["best", "worst"])
    ax.tick_params(axis="y", length=0)
    for col, size in enumerate(sizes):
        ax.text(col, -4.0, f"n={len(grouped[size])}", ha="center", va="bottom", fontsize=6.5, clip_on=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.030)
    colorbar.set_label("Force MAE", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, width=0.6, length=2)
    fig.savefig(output_dir / "paper_conservative_combo_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "paper_conservative_combo_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_metrics(args.metrics_csv.resolve(), args.metric)
    summary = summarize(grouped)
    write_summary(output_dir / "paper_conservative_combo_si_style_summary.csv", summary)
    plot_si_style(grouped, output_dir, args.metric, args.dpi)
    plot_heatmap_only(grouped, output_dir, args.metric, args.dpi)
    print(f"Saved SI-style combinatorial figures to: {output_dir}")


if __name__ == "__main__":
    main()
