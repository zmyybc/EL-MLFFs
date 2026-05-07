from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


BASE_MODELS = ["dp", "nep", "mtp", "soap", "painn", "schnet", "mace"]
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "el-mlffs" / "checkpoints" / "meta_models" / "direct_combo"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "direct_combo_metrics"


def parse_task_index(filename: str) -> int:
    return int(filename.split("_", 1)[0])


def load_rows(checkpoint_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ckpt_path in sorted(checkpoint_dir.glob("*.pth"), key=lambda path: parse_task_index(path.name)):
        payload = torch.load(ckpt_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        base_model_names = list(metadata.get("base_model_names", []))
        rows.append(
            {
                "task_index": parse_task_index(ckpt_path.name),
                "checkpoint_name": ckpt_path.name,
                "combo_tag": ckpt_path.stem.split("_", 1)[1],
                "base_model_names": base_model_names,
                "num_models": len(base_model_names),
                "val_energy_mae": float(metadata.get("val_energy_mae", 0.0)),
                "val_force_mae": float(metadata["val_force_mae"]),
            }
        )
    return rows


def save_csv(rows: list[dict[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_index",
                "checkpoint_name",
                "combo_tag",
                "num_models",
                "base_model_names",
                "val_energy_mae",
                "val_force_mae",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["task_index"],
                    row["checkpoint_name"],
                    row["combo_tag"],
                    row["num_models"],
                    " ".join(row["base_model_names"]),
                    row["val_energy_mae"],
                    row["val_force_mae"],
                ]
            )


def save_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    best_force = min(rows, key=lambda row: row["val_force_mae"])
    summary = {
        "num_models_total": len(rows),
        "best_force": {
            "checkpoint_name": best_force["checkpoint_name"],
            "val_force_mae": best_force["val_force_mae"],
            "base_model_names": best_force["base_model_names"],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_ensemble_size_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    summary_path = output_dir / "ensemble_size_summary.csv"
    groups = sorted({int(row["num_models"]) for row in rows})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_models",
                "count",
                "force_mean",
                "force_std",
                "force_best",
                "best_force_checkpoint",
            ]
        )
        for group in groups:
            group_rows = [row for row in rows if int(row["num_models"]) == group]
            force_values = np.array([float(row["val_force_mae"]) for row in group_rows], dtype=float)
            best_force_row = min(group_rows, key=lambda row: float(row["val_force_mae"]))
            writer.writerow(
                [
                    group,
                    len(group_rows),
                    f"{force_values.mean():.8f}",
                    f"{force_values.std(ddof=0):.8f}",
                    f"{force_values.min():.8f}",
                    best_force_row["checkpoint_name"],
                ]
            )


def plot_force_relationship(rows: list[dict[str, object]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)

    groups = sorted({int(row["num_models"]) for row in rows})
    values = [[float(row["val_force_mae"]) for row in rows if int(row["num_models"]) == group] for group in groups]
    means = np.array([np.mean(v) for v in values], dtype=float)
    bests = np.array([np.min(v) for v in values], dtype=float)
    positions = np.array(groups, dtype=float)

    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops={"facecolor": "#dbe9f6", "edgecolor": "#355070", "linewidth": 1.2},
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#355070", "linewidth": 1.1},
        capprops={"color": "#355070", "linewidth": 1.1},
        flierprops={"marker": "o", "markerfacecolor": "#9bb6d1", "markeredgecolor": "#355070", "markersize": 4, "alpha": 0.6},
    )

    rng = np.random.default_rng(42)
    for position, group_values in zip(positions, values):
        jitter = rng.normal(0.0, 0.06, size=len(group_values))
        ax.scatter(
            np.full(len(group_values), position) + jitter,
            group_values,
            s=24,
            color="#6d8fb3",
            alpha=0.45,
            edgecolors="none",
            zorder=2,
        )

    ax.plot(positions, means, color="#1d3557", marker="o", markersize=6, linewidth=2.2, label="Mean", zorder=3)
    ax.plot(
        positions,
        bests,
        color="#c1121f",
        marker="D",
        markersize=5,
        linewidth=1.8,
        linestyle="--",
        label="Best",
        zorder=3,
    )

    ax.set_xticks(groups)
    ax.set_xlabel("Number of Base Models in Ensemble")
    ax.set_ylabel("Validation Force MAE")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.set_title("Direct Ensemble: Size vs Force Accuracy")

    fig.suptitle("Direct Ensemble Size vs Force Accuracy", fontsize=16)
    output_path = output_dir / "ensemble_size_vs_force_mae.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_composition_heatmap(rows: list[dict[str, object]], output_dir: Path) -> None:
    sorted_by_force = sorted(rows, key=lambda row: row["val_force_mae"])
    top_rows = sorted_by_force[:20]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    heatmap_data = np.zeros((len(top_rows), len(BASE_MODELS)), dtype=float)
    for row_index, row in enumerate(top_rows):
        for col_index, model_name in enumerate(BASE_MODELS):
            heatmap_data[row_index, col_index] = 1.0 if model_name in row["base_model_names"] else 0.0

    ax.imshow(heatmap_data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(BASE_MODELS)))
    ax.set_xticklabels(BASE_MODELS, fontsize=11)
    ax.set_yticks(np.arange(len(top_rows)))
    bar_labels = [f"{row['task_index']}: {row['combo_tag']}" for row in top_rows]
    ax.set_yticklabels(bar_labels, fontsize=8)
    ax.set_title("Base-Model Composition of Top 20 Force-MAE Checkpoints", fontsize=15)
    ax.set_xlabel("Base Model")
    ax.set_ylabel("Checkpoint")

    for row_index in range(len(top_rows)):
        for col_index in range(len(BASE_MODELS)):
            mark = "✓" if heatmap_data[row_index, col_index] > 0 else ""
            ax.text(col_index, row_index, mark, ha="center", va="center", fontsize=9, color="black")

    fig.savefig(output_dir / "top20_composition_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize direct combo checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.checkpoint_dir)
    save_csv(rows, args.output_dir)
    save_summary(rows, args.output_dir)
    save_ensemble_size_summary(rows, args.output_dir)
    plot_force_relationship(rows, args.output_dir)
    plot_composition_heatmap(rows, args.output_dir)
    print(f"Wrote metrics for {len(rows)} checkpoints to {args.output_dir}")


if __name__ == "__main__":
    main()
