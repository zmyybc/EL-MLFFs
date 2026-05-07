#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

REQUIRED_KEYS = ("true_forces", "base_pred_forces", "meta_pred_forces")
DEFAULT_LABELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")


def parse_model_labels(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_LABELS)
    if "," in raw:
        labels = [item.strip() for item in raw.split(",")]
    else:
        labels = raw.split()
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError("--model-labels resolved to an empty list")
    return labels


def load_eval_pickle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data_list = pickle.load(handle)
    if not isinstance(data_list, list) or not data_list:
        raise ValueError(f"Expected a non-empty list in {path}")

    true_forces = []
    base_forces = []
    meta_forces = []
    for idx, item in enumerate(data_list):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {idx} is not a dict")
        missing = [key for key in REQUIRED_KEYS if key not in item]
        if missing:
            raise ValueError(f"Entry {idx} is missing keys: {missing}")
        true_forces.append(np.asarray(item["true_forces"], dtype=np.float64))
        base_forces.append(np.asarray(item["base_pred_forces"], dtype=np.float64))
        meta_forces.append(np.asarray(item["meta_pred_forces"], dtype=np.float64))

    true_np = np.concatenate(true_forces, axis=0)
    base_np = np.concatenate(base_forces, axis=0)
    meta_np = np.concatenate(meta_forces, axis=0)

    if true_np.ndim != 2 or true_np.shape[-1] != 3:
        raise ValueError(f"true_forces must concatenate to shape [atoms, 3], got {true_np.shape}")
    if meta_np.shape != true_np.shape:
        raise ValueError(f"meta_pred_forces shape {meta_np.shape} does not match true_forces {true_np.shape}")
    if base_np.ndim != 3:
        raise ValueError(f"base_pred_forces must concatenate to rank-3, got {base_np.shape}")
    if base_np.shape[0] != true_np.shape[0]:
        raise ValueError(f"base_pred_forces atom dimension {base_np.shape[0]} does not match {true_np.shape[0]}")
    return true_np, base_np, meta_np


def normalize_base_shape(base_np: np.ndarray, n_labels: int | None = None) -> np.ndarray:
    # Preferred/reference layout: [atoms, 3, n_models]. Also accept [atoms, n_models, 3].
    if base_np.shape[1] == 3:
        normalized = base_np
    elif base_np.shape[2] == 3:
        normalized = np.transpose(base_np, (0, 2, 1))
    else:
        raise ValueError(f"Cannot infer base_pred_forces layout from shape {base_np.shape}")
    if n_labels is not None and normalized.shape[2] != n_labels:
        raise ValueError(
            f"Number of model labels ({n_labels}) does not match base force model dimension ({normalized.shape[2]})"
        )
    return normalized


def infer_epoch(path: Path) -> str:
    match = re.search(r"epoch[_-](\d+)", path.name)
    return match.group(1) if match else "unknown"


def moving_average_ignore_nan(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1
    out = np.full_like(values, np.nan, dtype=np.float64)
    half = window_size // 2
    for idx in range(values.size):
        start = max(0, idx - half)
        end = min(values.size, idx + half + 1)
        window = values[start:end]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[idx] = finite.mean()
    return out


def compute_metrics(true_np: np.ndarray, base_np: np.ndarray, meta_np: np.ndarray, labels: list[str]) -> dict:
    true_flat = true_np.reshape(-1)
    meta_flat = meta_np.reshape(-1)
    base_maes = []
    for idx, label in enumerate(labels):
        base_flat = base_np[:, :, idx].reshape(-1)
        base_maes.append({"model": label, "force_mae": float(np.mean(np.abs(base_flat - true_flat)))})
    meta_mae = float(np.mean(np.abs(meta_flat - true_flat)))
    best = min(base_maes, key=lambda item: item["force_mae"])
    return {
        "num_structures": None,
        "num_atoms": int(true_np.shape[0]),
        "num_force_components": int(true_flat.size),
        "num_base_models": int(base_np.shape[2]),
        "meta_force_mae": meta_mae,
        "avg_base_force_mae": float(np.mean([item["force_mae"] for item in base_maes])),
        "best_base_model": best["model"],
        "best_base_force_mae": best["force_mae"],
        "base_force_mae": base_maes,
    }


def write_metrics(metrics: dict, output_dir: Path, prefix: str) -> None:
    json_path = output_dir / f"{prefix}_force_error_metrics.json"
    csv_path = output_dir / f"{prefix}_force_error_metrics.csv"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry", "force_mae"])
        writer.writerow(["meta", metrics["meta_force_mae"]])
        writer.writerow(["avg_base", metrics["avg_base_force_mae"]])
        writer.writerow(["best_base:" + str(metrics["best_base_model"]), metrics["best_base_force_mae"]])
        for item in metrics["base_force_mae"]:
            writer.writerow(["base:" + str(item["model"]), item["force_mae"]])


def find_dense_window(values: np.ndarray, abs_range: tuple[float, float], width: float, exclude: tuple[float, float] | None = None) -> tuple[float, float]:
    if values.size == 0:
        return -width / 2, width / 2
    max_abs = abs_range[1]
    hist, bins = np.histogram(values, bins=500, range=(-max_abs, max_abs))
    bin_width = bins[1] - bins[0]
    if exclude is not None:
        centers = bins[:-1] + bin_width / 2
        hist[(centers >= exclude[0]) & (centers <= exclude[1])] = 0
    window_bins = max(1, int(width / bin_width))
    counts = np.convolve(hist, np.ones(window_bins), mode="valid")
    if counts.size == 0:
        return -width / 2, width / 2
    start_idx = int(np.argmax(counts))
    center = bins[start_idx] + window_bins * bin_width / 2
    return center - width / 2, center + width / 2


def setup_style() -> None:
    mpl.rcParams.update({
        "font.sans-serif": ["Arial"],
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 300,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fafbfc",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, output_dir: Path, stem: str, formats: list[str], bbox_inches: str | None = "tight") -> None:
    for fmt in formats:
        kwargs = {"dpi": 300}
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        fig.savefig(output_dir / f"{stem}.{fmt}", **kwargs)


def style_axes(ax, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#e1e4e8", linewidth=1.0, linestyle="--", alpha=0.95, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#24292e")
        spine.set_linewidth(1.5)
    ax.tick_params(direction="out", length=5.0, width=1.5, colors="#24292e")


def plot_parity(true_np: np.ndarray, base_np: np.ndarray, meta_np: np.ndarray, labels: list[str], metrics: dict, output_dir: Path, prefix: str, args) -> None:
    true_flat = true_np.reshape(-1)
    meta_flat = meta_np.reshape(-1)
    abs_true = np.abs(true_flat)
    rng = np.random.RandomState(42)

    idx_low = np.where((abs_true >= 0) & (abs_true < 1))[0]
    idx_high = np.where((abs_true >= 1) & (abs_true <= args.plot_range))[0]
    idx_low = rng.choice(idx_low, min(idx_low.size, args.samples_per_range), replace=False) if idx_low.size else np.array([], dtype=int)
    idx_high = rng.choice(idx_high, min(idx_high.size, args.samples_per_range), replace=False) if idx_high.size else np.array([], dtype=int)
    sample_idx = np.concatenate([idx_low, idx_high])
    rng.shuffle(sample_idx)
    if sample_idx.size == 0:
        raise ValueError("No force components were selected for parity plotting")

    true_s = true_flat[sample_idx]
    meta_s = meta_flat[sample_idx]
    high_s_mask = np.abs(true_s) >= 1.0
    base_color = "#0366d6" # 经典学术蓝
    meta_color = "#9e2a2b" # 暗红，降低饱和度
    diag_color = "#24292e"

    fig, ax = plt.subplots(figsize=(8, 6)) # 统一画布比例
    for idx in range(base_np.shape[2]):
        ax.scatter(
            true_s,
            base_np[:, :, idx].reshape(-1)[sample_idx],
            color=base_color,
            alpha=0.028,
            s=4.0, # 稍微增大点径
            linewidths=0,
            zorder=1,
        )
    ax.scatter([], [], color=base_color, label="Base models", alpha=0.8, s=25)
    ax.scatter(true_s, meta_s, color=meta_color, label="Meta model", alpha=0.15, s=6.0, linewidths=0, zorder=3)

    # Reinforce sparse high-force components so the tails remain visible in print.
    if np.any(high_s_mask):
        for idx in range(base_np.shape[2]):
            ax.scatter(
                true_s[high_s_mask],
                base_np[:, :, idx].reshape(-1)[sample_idx][high_s_mask],
                color=base_color,
                alpha=0.18,
                s=18.0,
                linewidths=0.25,
                edgecolors="#023e8a",
                zorder=2,
            )
        ax.scatter(
            true_s[high_s_mask],
            meta_s[high_s_mask],
            color=meta_color,
            alpha=0.42,
            s=22.0,
            linewidths=0.3,
            edgecolors="#86181d",
            zorder=4,
        )
    
    # 理想对角线加白边防遮挡
    ax.plot([-args.plot_range, args.plot_range], [-args.plot_range, args.plot_range], 
            color=diag_color, linestyle="--", linewidth=1.5, label="Ideal", zorder=4,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="w")])

    ax.set_xlabel("True force (eV/Å)")
    ax.set_ylabel("Predicted force (eV/Å)")
    ax.set_xlim(-args.plot_range, args.plot_range)
    ax.set_ylim(-args.plot_range, args.plot_range)
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax)
    
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#24292e",
        framealpha=0.95,
        borderpad=0.5,
        handlelength=1.2,
        columnspacing=1.0,
        markerscale=1.0,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1)

    inset_width = args.inset_width
    low_values = true_flat[(abs_true >= 0) & (abs_true < 1)]
    lim1 = find_dense_window(low_values, (0, 1), inset_width)
    high_values = true_flat[(abs_true >= 1) & (abs_true <= args.plot_range)]
    lim2 = find_dense_window(high_values, (1, args.plot_range), inset_width, exclude=(lim1[0] - 3.0, lim1[1] + 3.0))

    def inset_maes(lim: tuple[float, float]) -> tuple[float, float]:
        idx = np.where((true_flat >= lim[0]) & (true_flat <= lim[1]))[0]
        if idx.size == 0:
            return float("nan"), float("nan")
        meta_mae = float(np.mean(np.abs(meta_flat[idx] - true_flat[idx])))
        base_mae = float(np.mean([np.mean(np.abs(base_np[:, :, j].reshape(-1)[idx] - true_flat[idx])) for j in range(base_np.shape[2])]))
        return meta_mae, base_mae

    for lim, bounds in [(lim1, [0.08, 0.63, 0.30, 0.30]), (lim2, [0.64, 0.08, 0.30, 0.30])]:
        idx = np.where((true_s >= lim[0]) & (true_s <= lim[1]))[0]
        high_inset = np.abs(true_s[idx]) >= 1.0
        ax_in = ax.inset_axes(bounds)
        for j in range(base_np.shape[2]):
            ax_in.scatter(true_s[idx], base_np[:, :, j].reshape(-1)[sample_idx][idx], color=base_color, alpha=0.06, s=5, linewidths=0)
        ax_in.scatter(true_s[idx], meta_s[idx], color=meta_color, alpha=0.26, s=6, linewidths=0)
        if np.any(high_inset):
            for j in range(base_np.shape[2]):
                ax_in.scatter(
                    true_s[idx][high_inset],
                    base_np[:, :, j].reshape(-1)[sample_idx][idx][high_inset],
                    color=base_color,
                    alpha=0.22,
                    s=18,
                    linewidths=0.25,
                    edgecolors="#023e8a",
                )
            ax_in.scatter(
                true_s[idx][high_inset],
                meta_s[idx][high_inset],
                color=meta_color,
                alpha=0.48,
                s=22,
                linewidths=0.3,
                edgecolors="#86181d",
            )
        ax_in.plot([lim[0], lim[1]], [lim[0], lim[1]], color=diag_color, linestyle="--", linewidth=1.2)
        meta_mae, base_mae = inset_maes(lim)
        ax_in.set_xlim(lim)
        ax_in.set_ylim(lim)
        ax_in.text(
            0.05,
            0.95,
            f"Meta: {meta_mae:.4f}\nBase:  {base_mae:.4f}",
            transform=ax_in.transAxes,
            va="top",
            ha="left",
            fontsize=10, # 增大局部放大图的字体
            fontweight='bold',
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff8e7", "edgecolor": "none", "alpha": 0.9},
        )
        ax_in.set_xticks([round(lim[0], 1), round(lim[1], 1)])
        ax_in.set_yticks([round(lim[0], 1), round(lim[1], 1)])
        ax_in.tick_params(labelsize=10, length=3.0, width=1.0)
        style_axes(ax_in)
        ax_in.set_aspect("equal", adjustable="box")
        ax.indicate_inset_zoom(ax_in, edgecolor="#586069", linewidth=1.5, alpha=0.85)

    fig.tight_layout(pad=0.5)
    save_figure(fig, output_dir, f"{prefix}_force_parity", args.formats, bbox_inches="tight")
    plt.close(fig)


def plot_binned_mae(true_np: np.ndarray, base_np: np.ndarray, meta_np: np.ndarray, labels: list[str], output_dir: Path, prefix: str, args) -> None:
    true_flat = true_np.reshape(-1)
    meta_err = np.abs(meta_np.reshape(-1) - true_flat)
    base_errs = [np.abs(base_np[:, :, idx].reshape(-1) - true_flat) for idx in range(base_np.shape[2])]

    bins = np.arange(-args.plot_range, args.plot_range + args.bin_width, args.bin_width)
    centers = (bins[:-1] + bins[1:]) / 2
    digitized = np.digitize(true_flat, bins)

    meta_binned = []
    base_binned = [[] for _ in range(base_np.shape[2])]
    counts = []
    for bin_idx in range(1, len(bins)):
        idx = np.where(digitized == bin_idx)[0]
        counts.append(int(idx.size))
        if idx.size < args.min_points_per_bin:
            meta_binned.append(np.nan)
            for j in range(base_np.shape[2]):
                base_binned[j].append(np.nan)
            continue
        meta_binned.append(float(meta_err[idx].mean()))
        for j in range(base_np.shape[2]):
            base_binned[j].append(float(base_errs[j][idx].mean()))

    meta_smoothed = moving_average_ignore_nan(np.asarray(meta_binned), args.smoothing_window)
    base_smoothed = [moving_average_ignore_nan(np.asarray(values), args.smoothing_window) for values in base_binned]

    csv_path = output_dir / f"{prefix}_binned_force_mae.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_center", "count", "meta_mae", *[f"{label}_mae" for label in labels]])
        for i, center in enumerate(centers):
            writer.writerow([center, counts[i], meta_binned[i], *[base_binned[j][i] for j in range(base_np.shape[2])]])

    fig, ax = plt.subplots(figsize=(8, 6)) # 放大画布
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#76B7B2", "#9C755F", "#B07AA1", "#6B7280"]
    for idx, label in enumerate(labels):
        # 增加 base 线的宽度
        ax.plot(centers, base_smoothed[idx], color=colors[idx % len(colors)], linewidth=2.0, alpha=0.75, label=label)
    
    # Meta 线标红、加粗并加上白边防遮挡
    ax.plot(
        centers, meta_smoothed, color="#cb2431", linewidth=3.5, alpha=1.0, label="Meta", zorder=base_np.shape[2] + 1,
        path_effects=[pe.withStroke(linewidth=6, foreground="w")]
    )
    
    ax.set_xlabel("True force (eV/Å)")
    ax.set_ylabel("Mean absolute error (eV/Å)")
    ax.set_xlim(-args.plot_range, args.plot_range)
    ax.set_ylim(bottom=args.ymin)
    ax.set_yscale("log")
    style_axes(ax)
    
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#24292e",
        framealpha=0.95,
        ncol=4,
        columnspacing=1.0,
        handlelength=1.5,
        borderpad=0.5,
    )
    fig.tight_layout(pad=0.5)
    save_figure(fig, output_dir, f"{prefix}_binned_force_mae", args.formats)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot force parity and binned force-error distributions from eval_data pickle files.")
    parser.add_argument("--pickle-file", required=True, type=Path, help="Pickle containing true/base/meta force arrays.")
    parser.add_argument("--model-labels", help="Comma- or space-separated base model labels. Defaults to dp,nep,mtp,soap,painn,schnet,mace.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/force_error_distribution"))
    parser.add_argument("--prefix", help="Output filename prefix. Defaults to pickle stem.")
    parser.add_argument("--plots", choices=["both", "parity", "binned"], default="both")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-range", type=int, default=250000)
    parser.add_argument("--plot-range", type=float, default=10.0)
    parser.add_argument("--inset-width", type=float, default=2.0)
    parser.add_argument("--bin-width", type=float, default=0.25)
    parser.add_argument("--min-points-per-bin", type=int, default=100)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--ymin", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    labels = parse_model_labels(args.model_labels)
    true_np, base_np, meta_np = load_eval_pickle(args.pickle_file)
    base_np = normalize_base_shape(base_np, len(labels))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.pickle_file.stem

    metrics = compute_metrics(true_np, base_np, meta_np, labels)
    metrics["pickle_file"] = str(args.pickle_file)
    metrics["epoch"] = infer_epoch(args.pickle_file)
    metrics["model_labels"] = labels
    write_metrics(metrics, args.output_dir, prefix)

    if args.plots in {"both", "parity"}:
        plot_parity(true_np, base_np, meta_np, labels, metrics, args.output_dir, prefix, args)
    if args.plots in {"both", "binned"}:
        plot_binned_mae(true_np, base_np, meta_np, labels, args.output_dir, prefix, args)

    print(json.dumps(metrics, indent=2))
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
