from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


OUTPUT_DIR = Path("/mnt/bn/bangchen/EL-MLFFs/reports/conservative_combo_metrics")
OUTPUT_PNG = OUTPUT_DIR / "paper_diversity_replacement_effect_scatter.png"
OUTPUT_PDF = OUTPUT_DIR / "paper_diversity_replacement_effect_scatter.pdf"
OUTPUT_CSV = OUTPUT_DIR / "paper_diversity_replacement_effect_scatter.csv"

# 基准数据
REFERENCE_NAME = "7-distinct"
REFERENCE_FORCE_MAE = 0.007585

# 替换数据
ROWS = [
    ("dp", 0.010108),
    ("nep", 0.010253),
    ("mtp", 0.010460),
    ("soap", 0.010101),
    ("painn", 0.009477),
    ("schnet", 0.010008),
    ("mace", 0.009821),
]

FAMILY_ORDER = ["dp", "nep", "mtp", "soap", "painn", "schnet", "mace"]


def write_csv(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setting", "val_force_mae", "delta_vs_reference"])
        writer.writerow([REFERENCE_NAME, f"{REFERENCE_FORCE_MAE:.6f}", "0.000000"])
        for name, value in ROWS:
            writer.writerow([f"diverse_{name}", f"{value:.6f}", f"{value - REFERENCE_FORCE_MAE:+.6f}"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_CSV)

    # 1. 全局字体与样式设置 (学术标准 Arial)
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.family"] = "sans-serif"

    # 数据准备
    ordered_rows = [(name, dict(ROWS)[name]) for name in FAMILY_ORDER]
    names_for_plot = [f"Replace {name.upper()}" for name, _ in ordered_rows]
    maes_for_plot = np.array([value for _, value in ordered_rows], dtype=float)
    y_pos = np.arange(len(names_for_plot))

    # 2. 画布建立
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # 绘制水平细虚线引导视线 (替代棒棒糖的粗线)
    ax.grid(axis="y", color="#e1e4e8", linestyle=":", linewidth=1.5, zorder=0)

    # 保留左侧和下侧边框，去除上侧和右侧，使散点图更透气
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#24292e")
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_color("#24292e")
    ax.spines["bottom"].set_linewidth(1.8)

    # 3. 动态扩展 X 轴边界
    x_min = min(maes_for_plot.min(), REFERENCE_FORCE_MAE) - 0.0002
    x_max = max(maes_for_plot.max(), REFERENCE_FORCE_MAE) + 0.0007
    ax.set_xlim(x_min, x_max)

    # 4. 背景语义阴影区 (优越区 vs 劣化区)
    # 绿色区域：误差低于基准 (Better)
    ax.axvspan(x_min, REFERENCE_FORCE_MAE, facecolor="#28a745", alpha=0.06, zorder=0)
    # 红色区域：误差高于基准 (Worse)
    ax.axvspan(REFERENCE_FORCE_MAE, x_max, facecolor="#cb2431", alpha=0.06, zorder=0)

    # 5. 绘制基准线 (Baseline)
    ax.axvline(
        REFERENCE_FORCE_MAE,
        color="#24292e",
        linestyle="--",
        linewidth=2.0,
        zorder=1,
    )
    
    # 基准线顶部标注
    ax.text(
        REFERENCE_FORCE_MAE, -0.8,
        f"Baseline: {REFERENCE_NAME} ({REFERENCE_FORCE_MAE:.6f})",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="#24292e",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#24292e", lw=1.2, alpha=0.9),
        zorder=5
    )

    # 6. 绘制核心散点
    worse_color = "#cb2431"
    better_color = "#28a745"

    for i, (name, mae) in enumerate(zip(names_for_plot, maes_for_plot)):
        delta = mae - REFERENCE_FORCE_MAE
        is_worse = delta > 0
        color = worse_color if is_worse else better_color
        
        # 散点主体
        ax.scatter(
            mae, y_pos[i], 
            s=180,  # 点的大小
            color=color, 
            edgecolors="#ffffff", 
            linewidths=1.5, 
            zorder=4,
            path_effects=[pe.withStroke(linewidth=3, foreground="#ffffff")]
        )
        
        # 散点旁边的数值标注
        prefix = "+" if is_worse else ""
        label_text = f"{mae:.6f} ({prefix}{delta:.5f})"
        
        # 调整文本位置，防止遮挡散点
        text_x = mae + 0.00015
        
        ax.text(
            text_x, y_pos[i], 
            label_text,
            va="center", ha="left", 
            fontsize=11.5, fontweight="bold", color=color,
            zorder=5
        )

    # 7. 坐标轴格式化
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_for_plot, fontsize=14, fontweight="medium")
    ax.invert_yaxis() # 第一条在最上
    ax.set_ylim(len(names_for_plot) - 0.5, -1.2) # 为顶部的 Baseline 标注留出空间

    ax.set_xlabel("Validation Force MAE (eV/Å)", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_ylabel("Ensemble Setting", fontsize=16, fontweight="bold", labelpad=12)
    
    ax.tick_params(axis="x", which="major", labelsize=13, colors="#24292e", width=1.5, length=6)
    ax.tick_params(axis="y", which="major", labelsize=13, colors="#24292e", width=1.5, length=0) # 隐藏 Y 轴短刻度

    # 8. 图例说明 (通过简单的文本放置在合适位置，比带边框的 legend 更融洽)
    ax.text(
        0.02, 0.04, 
        "Shaded Green: Performance Improvement\nShaded Red: Performance Degradation", 
        transform=ax.transAxes,
        fontsize=11, color="#586069", style="italic",
        bbox=dict(boxstyle="square,pad=0.5", fc="#ffffff", ec="none", alpha=0.8)
    )

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", transparent=False)
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"Saved aesthetically enhanced scatter figure: {OUTPUT_PNG}")
    print(f"Saved aesthetically enhanced scatter figure: {OUTPUT_PDF}")
    print(f"Saved summary: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()