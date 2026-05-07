from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

DIRECT_CSV = Path("reports/direct_model_contribution_7model/model_contribution_by_atom_type.csv")
CONSERV_CSV = Path("reports/conservative_model_contribution/model_contribution_by_atom_type.csv")
OUTDIR = Path("reports/direct_vs_conservative_contribution")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_all_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_overall(rows):
    return {r["base_model"]: float(r["normalized_contribution"]) for r in rows if r["group"] == "All"}


def main():
    direct_rows = load_all_rows(DIRECT_CSV)
    conserv_rows = load_all_rows(CONSERV_CSV)

    direct_all = extract_overall(direct_rows)
    conserv_all = extract_overall(conserv_rows)

    models = ["dp", "nep", "mtp", "soap", "painn", "schnet", "mace"]
    direct_vals = [direct_all.get(m, 0.0) * 100 for m in models]
    conserv_vals = [conserv_all.get(m, 0.0) * 100 for m in models]
    labels = [m.upper() for m in models]

    import matplotlib.font_manager as fm
    from pathlib import Path
    roboto_dir = Path("/opt/tiger/miniconda3/envs/horm/lib/python3.11/site-packages/font_roboto/files")
    for fpath in roboto_dir.glob("*.ttf"):
        fm.fontManager.addfont(str(fpath))

    plt.rcParams.update({
        "font.family": "Roboto",
        "font.size": 10,
        "font.weight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=300, sharey=True)
    x = np.arange(len(labels))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for idx, (ax, vals, title) in enumerate(zip(axes, [direct_vals, conserv_vals], ["(a) Direct ensemble", "(b) Conservative ensemble"])):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 42)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.5, linestyle=":")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Normalized contribution (%)")

    fig.tight_layout(pad=0.5)
    fig.savefig(OUTDIR / "direct_vs_conservative_contribution.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "direct_vs_conservative_contribution.png", dpi=300, bbox_inches="tight")
    print(f"Saved to {OUTDIR}")


if __name__ == "__main__":
    main()
