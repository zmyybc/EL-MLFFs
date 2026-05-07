from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# 1. 原始数据
CSV_DATA = """dataset,model_type,model_name,val_force_mae_eV_A
methanol,base,mace,0.0140
methanol,base,painn,0.0153
methanol,base,schnet,0.0184
methanol,base,soap,0.0289
methanol,base,mtp,0.0487
methanol,base,dp,0.0557
methanol,base,nep,0.0721
methanol,meta,direct,0.0082
methanol,meta,conserv,0.0073
peptide,base,schnet,0.1603
peptide,base,soap,0.1687
peptide,base,mtp,0.1870
peptide,base,nep,0.2308
peptide,meta,direct,0.1320
peptide,meta,conserv,0.1041"""

# 用于美化模型名称显示的字典
PRETTY_NAMES = {
    "mace": "MACE",
    "painn": "PaiNN",
    "schnet": "SchNet",
    "soap": "SOAP",
    "mtp": "MTP",
    "dp": "DP",
    "nep": "NEP",
    "direct": "Meta (Direct)",
    "conserv": "Meta (Conserv)",
}

# 严格对齐前面图表的全局色彩字典
COLOR_BASE = "#0366d6"      # 学术蓝 (对应前面的 Base / ID 数据)
COLOR_META_DIR = "#e36209"  # 活力橙 (对应前面的过渡 / OOD-Cat 数据)
COLOR_META_CON = "#cb2431"  # 醒目红 (对应前面的 Meta 最终结果 / OOD-Both)
EDGE_COLOR = "#24292e"      # 统一的深空灰边框


def parse_data() -> dict[str, list[dict]]:
    """解析 CSV 字符串并按 dataset 分组"""
    reader = csv.DictReader(io.StringIO(CSV_DATA))
    data_by_dataset = {"methanol": [], "peptide": []}
    
    for row in reader:
        dataset = row["dataset"]
        data_by_dataset[dataset].append({
            "type": row["model_type"],
            "name": row["model_name"],
            "mae": float(row["val_force_mae_eV_A"])
        })
    return data_by_dataset


def sort_models(models: list[dict]) -> list[dict]:
    """排序逻辑：Base 模型按误差降序排，Meta 模型放在最后"""
    base_models = [m for m in models if m["type"] == "base"]
    meta_models = [m for m in models if m["type"] == "meta"]
    
    # Base 误差降序 (形成下坡阶梯感)
    base_models.sort(key=lambda x: x["mae"], reverse=True)
    # Meta 固定顺序：先 direct, 后 conserv
    meta_models.sort(key=lambda x: 0 if x["name"] == "direct" else 1)
    
    return base_models + meta_models


def setup_style():
    """全局字体和样式配置 (与之前所有图表完全一致)"""
    plt.rcParams.update({
        "font.sans-serif": ["Arial"],
        "font.family": "sans-serif",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fafbfc",
    })


def plot_single_panel(models: list[dict], output_path: Path):
    """绘制单面板图（无标题），适合作为6子图大图的一个子图"""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")
    
    models = sort_models(models)
    
    names = [PRETTY_NAMES.get(m["name"], m["name"]) for m in models]
    maes = [m["mae"] for m in models]
    
    # 根据模型类型分配高度统一的颜色
    colors = []
    for m in models:
        if m["type"] == "base":
            colors.append(COLOR_BASE)
        elif m["name"] == "direct":
            colors.append(COLOR_META_DIR)
        else:
            colors.append(COLOR_META_CON)

    x_pos = np.arange(len(names))
    
    # 1. 绘制网格线
    ax.grid(axis="y", color="#e1e4e8", linestyle="--", linewidth=1.2, zorder=0)
    
    # 2. 绘制柱状图
    bars = ax.bar(
        x_pos, maes, 
        color=colors, 
        edgecolor=EDGE_COLOR, 
        linewidth=1.5, 
        alpha=0.9,
        width=0.65,
        zorder=3
    )

    # 3. 添加带白边的悬浮数值标签
    for bar, mae in zip(bars, maes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height() + (max(maes) * 0.02),
            f"{mae:.4f}", 
            ha="center", va="bottom", 
            fontsize=11.5, fontweight="bold", color=EDGE_COLOR,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
            zorder=4
        )

    # 4. 强制加粗四面边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(EDGE_COLOR)
        spine.set_linewidth(1.8)

    # 5. 格式化坐标轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=13, fontweight="medium")
    ax.set_ylabel("Validation Force MAE (eV/Å)", fontsize=15, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12, colors=EDGE_COLOR, width=1.5, length=6)
    
    # 动态延长 Y 轴，防止标签越界
    ax.set_ylim(0, max(maes) * 1.15)

    # 强制关闭透明度以保证纯净白底
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"✅ 单面板图已保存至: {output_path.absolute()}")


def main():
    data_by_ds = parse_data()
    
    output_dir = Path("/mnt/bn/bangchen/EL-MLFFs/reports/conservative_combo_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成两张独立图片
    plot_single_panel(
        data_by_ds["methanol"], 
        output_dir / "panel_methanol_mae.png"
    )
    plot_single_panel(
        data_by_ds["peptide"], 
        output_dir / "panel_peptide_mae.png"
    )


if __name__ == "__main__":
    main()
