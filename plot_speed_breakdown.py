"""Stacked bar chart: inference time breakdown for ensemble methods."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parent


def setup_style():
    """全局字体和现代学术样式配置"""
    plt.rcParams.update({
        "font.sans-serif": ["Arial"],
        "font.family": "sans-serif",
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#24292e",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.color": "#24292e",
        "ytick.color": "#24292e",
        "text.color": "#24292e",
        "axes.labelcolor": "#24292e",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
    })


def main():
    parser = argparse.ArgumentParser()
    # 为防止覆盖你的原文件，默认输出名加上了 _refined
    parser.add_argument("--output", default=str(ROOT / "reports" / "speed_benchmark_breakdown_refined.png"))
    parser.add_argument("--output-pdf", default=str(ROOT / "reports" / "speed_benchmark_breakdown_refined.pdf"))
    args = parser.parse_args()

    # ── Data (median ms / batch, bs=4, A100) ─────────────────────────
    lsm_direct = 6.9
    lsm_conserv = 267.0

    avg7_base = 124.6
    lnc7_base = 122.9
    lnc7_head = 2.3
    direct7_base = 115.5
    direct7_meta = 4.2
    conserv7_base = 134.0
    conserv7_meta = 5.8
    conserv7_ag_base_e = 70.3
    conserv7_ag_meta_corr = 14.2
    conserv7_ag_vjp = 70.0

    # ── 组织数据 ──────────────────────────────────────────────────
    labels = [
        "LSM\nDirect",
        "LSM\nConserv.",
        "AVG-7",
        "LNC-7",
        "Ensemble\nDirect-7",
        "Ensemble\nConserv.-7",
    ]
    n = len(labels)
    x = np.arange(n)
    bar_width = 0.55

    # 语义化配色方案: 冷色(Forward) vs 暖色渐变(Autograd Overhead)
    c_fwd = "#1D4E89"       # 深学术蓝
    c_meta = "#7DB2D9"      # 浅钢蓝
    c_ag_base = "#F4A261"   # 柔和金橘
    c_ag_meta = "#E76F51"   # 珊瑚红
    c_ag_vjp = "#9B2226"    # 深酒红

    fwd      = [lsm_direct, lsm_conserv, avg7_base, lnc7_base, direct7_base, conserv7_base]
    meta     = [0, 0, 0, lnc7_head, direct7_meta, conserv7_meta]
    ag_base  = [0, 0, 0, 0, 0, conserv7_ag_base_e]
    ag_mcorr = [0, 0, 0, 0, 0, conserv7_ag_meta_corr]
    ag_vjp   = [0, 0, 0, 0, 0, conserv7_ag_vjp]

    totals = [f+m+a1+a2+a3 for f,m,a1,a2,a3 in zip(fwd,meta,ag_base,ag_mcorr,ag_vjp)]

    # ── 开始绘图 ──────────────────────────────────────────────────
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300, constrained_layout=True)

    # 绘制轻量级背景网格 (仅 Y 轴)
    ax.grid(axis="y", color="#e1e4e8", linestyle="--", linewidth=1.2, zorder=0)

    # 堆叠柱状图
    bottoms = np.zeros(n)
    
    # 统一定义柱子的边框样式，增加层次感
    bar_kwargs = dict(width=bar_width, edgecolor="#ffffff", linewidth=1.2, zorder=3)

    ax.bar(x, fwd, bottom=bottoms, color=c_fwd, label="Base model forward", **bar_kwargs)
    bottoms += fwd

    ax.bar(x, meta, bottom=bottoms, color=c_meta, label="Meta model forward", **bar_kwargs)
    bottoms += meta

    ax.bar(x, ag_base, bottom=bottoms, color=c_ag_base, label="Autograd: base energy", **bar_kwargs)
    bottoms += ag_base

    ax.bar(x, ag_mcorr, bottom=bottoms, color=c_ag_meta, label="Autograd: meta correction", **bar_kwargs)
    bottoms += ag_mcorr

    ax.bar(x, ag_vjp, bottom=bottoms, color=c_ag_vjp, label="Autograd: force-feature VJP", **bar_kwargs)
    bottoms += ag_vjp

    # ── 标注与修饰 ────────────────────────────────────────────────
    # 1. 在每个柱子顶部标注总时间
    for i, t in enumerate(totals):
        # 针对 6.9ms 保留一位小数，其余取整
        label_str = f"{t:.1f}" if t < 10 else f"{t:.0f}"
        
        ax.text(
            x[i], t + 6, label_str, 
            ha="center", va="bottom", 
            fontsize=11, fontweight="bold", color="#24292e",
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
            zorder=5
        )

    # 2. 定制精美的高精度 Autograd 分组括号（放在左侧）
    ag_total = conserv7_ag_base_e + conserv7_ag_meta_corr + conserv7_ag_vjp
    br_y_bottom = conserv7_base + conserv7_meta
    br_y_top = totals[5]
    br_x = x[5] - bar_width / 2 - 0.08  # 括号主线移到柱子左侧
    tick_len = 0.06                     # 括号横向短线长度
    br_color = "#586069"

    # 绘制垂直线段和上下两端的短横线
    ax.plot([br_x, br_x], [br_y_bottom, br_y_top], color=br_color, lw=1.5, clip_on=False, zorder=4)
    ax.plot([br_x, br_x + tick_len], [br_y_bottom, br_y_bottom], color=br_color, lw=1.5, clip_on=False, zorder=4)
    ax.plot([br_x, br_x + tick_len], [br_y_top, br_y_top], color=br_color, lw=1.5, clip_on=False, zorder=4)

    # 括号旁的文本说明（放在左侧）
    ax.text(
        br_x - 0.08, (br_y_bottom + br_y_top) / 2,
        f"Autograd Overhead\n{ag_total:.0f} ms",
        va="center", ha="right",
        fontsize=10.5, color=br_color, fontweight="bold", linespacing=1.3,
        clip_on=False
    )

    # ── 坐标轴与图例设置 ──────────────────────────────────────────
    # 保留四面边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#24292e")
        spine.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="medium")
    ax.set_ylabel("Median Latency (ms / batch, bs=4)", fontsize=14, fontweight="bold", labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    
    # X 轴边界：左侧给 bracket 留空间，右侧收紧
    ax.set_xlim(-0.6, 5.8)
    # Y 轴高度留白 15%
    ax.set_ylim(0, max(totals) * 1.15)

    # 利用中间空出的负空间放置完美双列图例
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.50, 0.98), 
        ncol=2,
        frameon=True, 
        fancybox=False,
        facecolor="#ffffff",
        edgecolor="#24292e",
        framealpha=0.75,
        fontsize=11.5,
        borderpad=0.6,
        handlelength=1.5
    )

    # 保存图表
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", transparent=False)
    fig.savefig(args.output_pdf, dpi=300, bbox_inches="tight", transparent=False)
    print(f"✅ Saved aesthetically enhanced figure: {args.output}")
    print(f"✅ Saved aesthetically enhanced figure: {args.output_pdf}")


if __name__ == "__main__":
    main()