from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import torch


DEFAULT_CHECKPOINT_DIR = Path(
    "el-mlffs/checkpoints/meta_models/conservative_combo"
)
DEFAULT_OUTPUT_DIR = Path("./reports/conservative_combo_metrics")
REFERENCE_CHECKPOINT = Path("./el-mlffs/checkpoints/meta_models/conservative_meta_current_bases_8gpu.pth")
DEFAULT_METRICS_CSV = Path("./reports/conservative_combo_metrics/metrics.csv")


def parse_task_index(filename: str) -> int:
    return int(filename.split("_", 1)[0])


def load_rows_from_checkpoints(checkpoint_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ckpt_path in sorted(checkpoint_dir.glob("*.pth"), key=lambda path: parse_task_index(path.name)):
        payload = torch.load(ckpt_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        base_model_names = list(metadata.get("base_model_names", []))
        rows.append(
            {
                "task_index": parse_task_index(ckpt_path.name),
                "checkpoint_name": ckpt_path.name,
                "num_models": len(base_model_names),
                "val_energy_mae": float(metadata["val_energy_mae"]),
                "val_force_mae": float(metadata["val_force_mae"]),
            }
        )
    return rows


def load_rows_from_csv(csv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "task_index": int(row["task_index"]),
                    "checkpoint_name": row["checkpoint_name"],
                    "num_models": int(row["num_models"]),
                    "val_energy_mae": float(row["val_energy_mae"]),
                    "val_force_mae": float(row["val_force_mae"]),
                }
            )
    return rows


def grouped_stats(rows: list[dict[str, object]], metric_key: str) -> tuple[list[int], list[list[float]], np.ndarray, np.ndarray]:
    groups = sorted({int(row["num_models"]) for row in rows})
    values = [[float(row[metric_key]) for row in rows if int(row["num_models"]) == group] for group in groups]
    means = np.array([np.mean(group_values) for group_values in values], dtype=float)
    bests = np.array([np.min(group_values) for group_values in values], dtype=float)
    return groups, values, means, bests


def save_summary_csv(rows: list[dict[str, object]], output_dir: Path) -> Path:
    summary_path = output_dir / "ensemble_size_summary.csv"
    groups = sorted({int(row["num_models"]) for row in rows})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "num_models",
                "count",
                "energy_mean",
                "energy_std",
                "energy_best",
                "force_mean",
                "force_std",
                "force_best",
                "best_force_checkpoint",
                "best_energy_checkpoint",
            ]
        )
        for group in groups:
            group_rows = [row for row in rows if int(row["num_models"]) == group]
            energy_values = np.array([float(row["val_energy_mae"]) for row in group_rows], dtype=float)
            force_values = np.array([float(row["val_force_mae"]) for row in group_rows], dtype=float)
            best_force_row = min(group_rows, key=lambda row: float(row["val_force_mae"]))
            best_energy_row = min(group_rows, key=lambda row: float(row["val_energy_mae"]))
            writer.writerow(
                [
                    group,
                    len(group_rows),
                    f"{energy_values.mean():.8f}",
                    f"{energy_values.std(ddof=0):.8f}",
                    f"{energy_values.min():.8f}",
                    f"{force_values.mean():.8f}",
                    f"{force_values.std(ddof=0):.8f}",
                    f"{force_values.min():.8f}",
                    best_force_row["checkpoint_name"],
                    best_energy_row["checkpoint_name"],
                ]
            )
    return summary_path


def load_reference_force_point(reference_checkpoint: Path) -> tuple[int, float] | None:
    if not reference_checkpoint.exists():
        return None
    payload = torch.load(reference_checkpoint, map_location="cpu")
    metadata = payload.get("metadata", {})
    base_model_names = list(metadata.get("base_model_names", []))
    val_force_mae = metadata.get("val_force_mae")
    if val_force_mae is None:
        return None
    return len(base_model_names), float(val_force_mae)


def plot_force_relationship(rows: list[dict[str, object]], output_dir: Path, reference_checkpoint: Path) -> Path:
    # 设置全局字体为 Arial
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 设置更高的分辨率和画布尺寸
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=300)

    # 背景与网格美化
    ax.set_facecolor('#fafbfc')
    fig.patch.set_facecolor('#ffffff')
    ax.grid(axis="y", color="#e1e4e8", linestyle="--", linewidth=1.0, zorder=0)

    # 恢复四面边框并加粗 (封闭式坐标轴)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#24292e")
        spine.set_linewidth(1.5)

    groups, values, means, bests = grouped_stats(rows, "val_force_mae")
    positions = np.array(groups, dtype=float)

    # 创建渐变颜色映射 (使用蓝紫色系)
    cmap = plt.get_cmap("PuBu")
    norm = mcolors.Normalize(vmin=min(groups) - 2, vmax=max(groups) + 1)

    # 绘制箱线图
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # 关闭默认异常值，用 scatter 绘制
        zorder=2
    )

    # 动态为每个箱体上色
    for i, box in enumerate(bp['boxes']):
        color = cmap(norm(groups[i]))
        box.set_facecolor(mcolors.to_rgba(color, alpha=0.7))
        box.set_edgecolor("#24292e")
        box.set_linewidth(1.5)

    # 统一设置须、帽、中位线的样式
    plt.setp(bp['whiskers'], color='#24292e', linewidth=1.5, linestyle='--')
    plt.setp(bp['caps'], color='#24292e', linewidth=1.5)
    plt.setp(bp['medians'], color='#cb2431', linewidth=2.5) # 中位线加粗加红

    # 散点 (Strip plot) 美化
    rng = np.random.default_rng(42)
    for i, (position, group_values) in enumerate(zip(positions, values)):
        jitter = rng.normal(0.0, 0.05, size=len(group_values))
        color = cmap(norm(groups[i]))
        ax.scatter(
            np.full(len(group_values), position) + jitter,
            group_values,
            s=45, # 稍微增大散点
            color=color,
            edgecolors="#ffffff", 
            linewidths=0.8,
            alpha=0.85,
            zorder=3,
        )

    # Mean 和 Best 折线美化
    ax.plot(
        positions, means, color="#0366d6", marker="o", markersize=8, 
        linewidth=2.5, label="Mean Accuracy", zorder=4,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")]
    )
    ax.plot(
        positions, bests, color="#28a745", marker="D", markersize=7, 
        linewidth=2.5, linestyle="--", label="Best Accuracy", zorder=4,
        path_effects=[pe.withStroke(linewidth=4, foreground="w")]
    )

    # 绘制参考基准点
    reference_point = load_reference_force_point(reference_checkpoint)
    if reference_point is not None:
        ax.scatter(
            [reference_point[0]],
            [reference_point[1]],
            marker="*",
            s=350, # 增大基准点
            color="#ffd33d",
            edgecolors="#9e6a03",
            linewidths=1.5,
            zorder=5,
            label="Prior 7-model reference"
        )
        ax.annotate(
            f"{reference_point[1]:.4f}",
            (reference_point[0], reference_point[1]),
            textcoords="offset points",
            xytext=(15, -15),
            fontsize=13, # 增大标注字体
            fontweight='bold',
            color="#735c0f",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e7", ec="none", alpha=0.9)
        )

    # 坐标轴美化 (增加字体与单位)
    ax.set_xticks(groups)
    ax.set_xlabel("Number of Base Models in Ensemble", fontsize=16, fontweight='bold', labelpad=12)
    ax.set_ylabel("Validation Force MAE (eV/Å)", fontsize=16, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=14, colors="#24292e", width=1.5, length=5)
    
    # 强制设置纵坐标上限
    ax.set_ylim(top=0.014)

    # 图例美化 (增加字体)
    ax.legend(
        frameon=True, 
        fancybox=False, # 使用直角/方正风格更贴合学术图 
        framealpha=0.95, 
        edgecolor="#24292e", 
        fontsize=13, 
        loc="upper right"
    )

    # 去掉原有的标题和副标题设置代码
    
    output_path = output_dir / "ensemble_size_vs_force_mae.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return output_path


def _percentile(sorted_vals: list[float], p: float) -> float:
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def filter_upper_outliers(rows: list[dict[str, object]], metric_key: str = "val_force_mae") -> list[dict[str, object]]:
    """Remove rows whose metric value exceeds the IQR upper bound within each num_models group."""
    from collections import defaultdict

    by_group: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[int(row["num_models"])].append(row)

    filtered: list[dict[str, object]] = []
    for group, group_rows in sorted(by_group.items()):
        values = sorted([float(r[metric_key]) for r in group_rows])
        n = len(values)
        if n < 4:
            filtered.extend(group_rows)
            continue
        q1 = _percentile(values, 0.25)
        q3 = _percentile(values, 0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        kept = [r for r in group_rows if float(r[metric_key]) <= upper]
        removed = [r for r in group_rows if float(r[metric_key]) > upper]
        if removed:
            print(f"  Removed {len(removed)} outlier(s) from {group}-model group (>{upper:.5f}):")
            for r in removed:
                print(f"    {r['checkpoint_name']:40s} {metric_key}={float(r[metric_key]):.5f}")
        filtered.extend(kept)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot conservative-combo accuracy vs ensemble size.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--from-csv", action="store_true", default=False, help="Load data from metrics.csv instead of checkpoints")
    parser.add_argument("--remove-outliers", action="store_true", default=False, help="Remove upper IQR outliers per ensemble-size group")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        rows = load_rows_from_csv(args.metrics_csv)
    else:
        rows = load_rows_from_checkpoints(args.checkpoint_dir)
    if args.remove_outliers:
        print("Filtering upper outliers...")
        rows = filter_upper_outliers(rows, "val_force_mae")
    summary_path = save_summary_csv(rows, args.output_dir)
    figure_path = plot_force_relationship(rows, args.output_dir, REFERENCE_CHECKPOINT)
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
