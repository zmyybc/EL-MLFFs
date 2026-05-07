from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


SPLITS = ("val_id", "ood_ads", "ood_cat", "ood_both")
LABELS = {
    "val_id": "ID",
    "ood_ads": "OOD-Ads",
    "ood_cat": "OOD-Cat",
    "ood_both": "OOD-Both",
}
# Use a softer, lower-saturation palette while preserving split separability.
COLORS = {
    "val_id": "#5B84B1",      # muted blue
    "ood_ads": "#7AA6A1",     # soft teal
    "ood_cat": "#C9A66B",     # muted sand
    "ood_both": "#B78AA3",    # dusty mauve
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-panel ID/OOD uncertainty-error plot for OC20.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/mnt/bn/changsu-data3/ybc/repos/EL-MLFFs_ybc_delivery_bundle/"
            "oc20_ood_eval_runs/uncertainty_vs_conserv_error"
        ),
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300) # 统一改为标准的 300 DPI
    return parser.parse_args()


def load_split(input_dir: Path, split_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(input_dir / f"per_structure_{split_name}.pt", map_location="cpu")
    return payload["component_mean_std"].to(torch.float64), payload["conserv_force_mae"].to(torch.float64)


def load_spearman(input_dir: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with (input_dir / "summary.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["uncertainty_metric"] == "component_mean_std":
                values[row["split"]] = float(row["spearman"])
    return values


def quantile_rows(x: torch.Tensor, y: torch.Tensor, bins: int) -> list[dict[str, float]]:
    order = torch.argsort(x)
    rows = []
    n = x.numel()
    for idx in range(bins):
        start = round(idx * n / bins)
        end = round((idx + 1) * n / bins)
        selected = order[start:end]
        xb = x[selected]
        yb = y[selected]
        rows.append(
            {
                "uncertainty_mean": float(xb.mean()),
                "error_mean": float(yb.mean()),
                "error_q25": float(torch.quantile(yb, 0.25)),
                "error_q75": float(torch.quantile(yb, 0.75)),
            }
        )
    return rows


def setup_matplotlib():
    """配置全局绘图样式，统一为现代高品质学术风格"""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial"],
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
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
        }
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    spearman = load_spearman(input_dir)
    setup_matplotlib()

    # 放大画布尺寸，适应新的字体和版式
    fig, ax = plt.subplots(figsize=(8, 6), dpi=args.dpi, constrained_layout=True)
    
    # 绘制更干净现代的网格系统
    ax.grid(True, color="#e1e4e8", linestyle="--", linewidth=1.2, zorder=0)

    # 强制设置坐标轴的颜色和线宽
    for spine in ax.spines.values():
        spine.set_color("#24292e")
        spine.set_linewidth(1.8)

    for split_name in SPLITS:
        x, y = load_split(input_dir, split_name)
        rows = quantile_rows(x, y, args.bins)
        xs = [row["uncertainty_mean"] for row in rows]
        ys = [row["error_mean"] for row in rows]
        q25 = [row["error_q25"] for row in rows]
        q75 = [row["error_q75"] for row in rows]
        
        color = COLORS[split_name]
        # 使用 LaTeX 渲染 $\rho$ 符号，让其更加专业
        label = rf"{LABELS[split_name]} ($\rho$={spearman[split_name]:.2f})"
        
        # 1. 绘制阴影区 (置信区间/分位数区间)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.12, linewidth=0, zorder=2)
        
        # 2. 绘制主折线和散点，并加入 Path Effect 防遮挡白边
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.5,
            marker="o",
            markevery=2, # 每隔一个点画一个 marker 以保持清爽
            markersize=8,
            markeredgecolor="#ffffff",
            markeredgewidth=1.2,
            label=label,
            zorder=4,
            path_effects=[pe.withStroke(linewidth=4.5, foreground="white")]
        )

    # 完善单位 (使用正规的 Å 符号)
    ax.set_xlabel("Base-Model Force Std (eV/Å)", labelpad=12)
    ax.set_ylabel("Meta-Model Force MAE (eV/Å)", labelpad=12)
    
    # 稍微拓宽坐标轴范围，以免加粗后的线条和阴影触边
    ax.set_xlim(0.015, 0.29)
    ax.set_ylim(0.015, 0.35)
    
    ax.tick_params(direction="out", length=6, width=1.5, colors="#24292e")

    # 优化图例 (Legend)
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        facecolor="#ffffff",
        edgecolor="#24292e",
        framealpha=0.95,
        loc="upper left",
        handlelength=1.5,
        borderpad=0.5,
        labelspacing=0.5,
    )
    legend.set_zorder(5)

    output_pdf = input_dir / "paper_uncertainty_id_ood_single_panel_refined.pdf"
    output_png = input_dir / "paper_uncertainty_id_ood_single_panel_refined.png"
    
    # 禁用背景透明，保证不论在哪里查看都拥有纯净白底
    fig.savefig(output_pdf, bbox_inches="tight", transparent=False)
    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)
    
    print(f"Saved aesthetically enhanced figure: {output_pdf}")
    print(f"Saved aesthetically enhanced figure: {output_png}")


if __name__ == "__main__":
    main()
