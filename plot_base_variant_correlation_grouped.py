from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grouped 21x21 base-variant correlation heatmap.")
    parser.add_argument(
        "--csv-path",
        default="./base_variant_correlation_21_outputs/base_variant_force_error_pearson_matrix.csv",
    )
    parser.add_argument(
        "--output-path",
        default="./base_variant_correlation_21_outputs/base_variant_force_error_pearson_matrix_group_bracketed.png",
    )
    return parser.parse_args()


GROUP_ORDER = ["dp", "nep", "mtp", "soap", "painn", "schnet", "mace"]
VARIANT_ORDER = ["baseline", "compact", "tiny", "expressive"]
VARIANT_TO_INDEX = {
    "baseline": 1,
    "compact": 2,
    "tiny": 3,
    "expressive": 3,
}


def split_label(label: str) -> tuple[str, str]:
    family, variant = label.split("_", 1)
    return family, variant


def sort_key(label: str) -> tuple[int, int, str]:
    family, variant = split_label(label)
    return (
        GROUP_ORDER.index(family),
        VARIANT_ORDER.index(variant),
        label,
    )


def pretty_label(label: str) -> str:
    family, variant = split_label(label)
    return f"{family}{VARIANT_TO_INDEX[variant]}"


def draw_horizontal_bracket(ax, start: float, end: float, y: float, label: str) -> None:
    color = "#24292e"
    lw = 2.0
    tick_len = 0.25 # 指向热力图的小刻度长度
    
    # 绘制水平主线
    ax.plot([start, end], [y, y], color=color, linewidth=lw, clip_on=False)
    # 绘制两端向上的小刻度 (因为y轴方向是0在最上，21在最下，所以向上是 y - tick_len)
    ax.plot([start, start], [y, y - tick_len], color=color, linewidth=lw, clip_on=False)
    ax.plot([end, end], [y, y - tick_len], color=color, linewidth=lw, clip_on=False)
    
    # 标注文字 (大写)
    ax.text((start + end) / 2, y + 0.15, label.upper(), ha="center", va="top", 
            fontsize=18, fontweight="bold", color=color, clip_on=False)


def draw_vertical_bracket(ax, start: float, end: float, x: float, label: str) -> None:
    color = "#24292e"
    lw = 2.0
    tick_len = 0.25
    
    # 绘制垂直主线
    ax.plot([x, x], [start, end], color=color, linewidth=lw, clip_on=False)
    # 绘制两端向右的小刻度 (向热力图方向)
    ax.plot([x, x + tick_len], [start, start], color=color, linewidth=lw, clip_on=False)
    ax.plot([x, x + tick_len], [end, end], color=color, linewidth=lw, clip_on=False)
    
    # 标注文字 (大写)
    ax.text(x - 0.2, (start + end) / 2, label.upper(), ha="right", va="center", 
            fontsize=18, fontweight="bold", color=color, clip_on=False)


def load_matrix(csv_path: Path) -> tuple[list[str], np.ndarray]:
    rows = list(csv.reader(csv_path.open()))
    labels = rows[0][1:]
    matrix = np.array([[float(x) for x in row[1:]] for row in rows[1:]], dtype=float)
    return labels, matrix


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 全局字体与样式设置
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    labels, matrix = load_matrix(csv_path)
    perm = sorted(range(len(labels)), key=lambda idx: sort_key(labels[idx]))
    ordered_labels = [labels[idx] for idx in perm]
    ordered_matrix = matrix[np.ix_(perm, perm)]

    # 2. 放大画布以容纳粗体文字和边框
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    sns.heatmap(
        ordered_matrix,
        cmap="PuBu", # 匹配之前的蓝紫色系
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.6,      # 单元格之间的白色间隙
        linecolor="white",
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        cbar_kws={"shrink": 0.85, "pad": 0.04},
    )

    # 去掉 Seaborn 默认生成的四面无用边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 3. 绘制族群内部分割线 (加深颜色与线宽，突出区块感)
    boundaries = []
    last_family = None
    for i, label in enumerate(ordered_labels):
        family, _ = split_label(label)
        if last_family is not None and family != last_family:
            boundaries.append(i)
        last_family = family
        
    for boundary in boundaries:
        ax.axhline(boundary, color="#24292e", linewidth=1.8, zorder=5)
        ax.axvline(boundary, color="#24292e", linewidth=1.8, zorder=5)

    # 给整个矩阵画一个最外围的粗边框框起来
    rect = plt.Rectangle((0, 0), len(ordered_labels), len(ordered_labels),
                         fill=False, edgecolor="#24292e", linewidth=3.0, clip_on=False, zorder=6)
    ax.add_patch(rect)

    # 4. 绘制外部的分组括号
    family_spans = []
    start = 0
    for family in GROUP_ORDER:
        end = start + 3
        family_spans.append((family, start, end))
        start = end

    # 间距微调，使 Bracket 距离主图更自然
    x_offset = -0.6
    y_offset = len(ordered_labels) + 0.6

    for family, start, end in family_spans:
        draw_horizontal_bracket(ax, start, end, y_offset, family)
        draw_vertical_bracket(ax, start, end, x_offset, family)

    # 放大绘图区范围，防止括号和文字被 tight_layout 裁剪掉
    ax.set_xlim(-2.5, len(ordered_labels) + 0.5)
    ax.set_ylim(len(ordered_labels) + 1.5, -0.5)

    # 5. 美化 Colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=15, length=6, width=1.5, colors="#24292e", direction="out")
    colorbar.set_label("Pearson correlation", size=18, weight="bold", color="#24292e", labelpad=15)
    # 为 Colorbar 加上外边框
    colorbar.outline.set_visible(True)
    colorbar.outline.set_color("#24292e")
    colorbar.outline.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close()
    print(f"Saved aesthetically enhanced heatmap: {output_path}")


if __name__ == "__main__":
    main()