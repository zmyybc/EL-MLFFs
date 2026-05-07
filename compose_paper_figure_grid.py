from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image


PANELS = [
    ("(a)", Path("reports/conservative_combo_metrics/panel_methanol_mae.png")),
    ("(b)", Path("reports/conservative_combo_metrics/panel_peptide_mae.png")),
    ("(c)", Path("reports/conservative_combo_metrics/ensemble_size_vs_force_mae.png")),
    ("(d)", Path("base_variant_correlation_21_outputs/base_variant_force_error_pearson_matrix_group_bracketed.png")),
    ("(e)", Path("reports/force_error_distribution_current_7model_paper/current_7model_paper_force_parity.png")),
    ("(f)", Path("reports/oc20_ood_eval_runs/uncertainty_vs_conserv_error/paper_uncertainty_id_ood_single_panel_refined.png")),
]

OUTPUT_DIR = Path("reports/conservative_combo_metrics")
OUTPUT_PNG = OUTPUT_DIR / "paper_six_panel_summary.png"
OUTPUT_PDF = OUTPUT_DIR / "paper_six_panel_summary.pdf"
PANEL_SIZE = (2400, 1900)


def load_panel_image(image_path: Path):
    image = Image.open(image_path).convert("RGBA")
    # 保持比例缩放到刚好 fit 进 PANEL_SIZE（小图允许放大，大图缩小，不裁剪）
    if "parity" in image_path.name:
        # 图e: 直接resize到目标尺寸 2100×1745
        image = image.resize((2100, 1745), Image.Resampling.LANCZOS)
    else:
        scale = min(PANEL_SIZE[0] / image.width, PANEL_SIZE[1] / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", PANEL_SIZE, (255, 255, 255, 255))
    offset = ((PANEL_SIZE[0] - image.width) // 2, (PANEL_SIZE[1] - image.height) // 2)
    canvas.paste(image, offset, image)
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(24, 12.4), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, (tag, image_path) in zip(axes.flat, PANELS):
        image = load_panel_image(image_path)
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.01,
            0.98,
            tag,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
        )

    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUTPUT_PDF, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
