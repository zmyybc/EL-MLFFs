from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


BASE_MODELS = ("dp", "nep", "mtp", "soap", "painn", "schnet", "mace")
SPLITS = ("val_id", "ood_ads", "ood_cat", "ood_both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate 7-base-model force prediction spread to conservative meta-model "
            "per-structure force error on OC20 OOD subsets."
        )
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("reports/oc20_ood_eval_runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/oc20_ood_eval_runs/uncertainty_vs_conserv_error"),
    )
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def force_cache_path(eval_root: Path, model_name: str, split_name: str) -> Path:
    for subdir in ("force_cache_gpu0", "force_cache_gpu1"):
        path = eval_root / subdir / f"{model_name}_{split_name}_forces.pt"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing force cache for model={model_name}, split={split_name}")


def conserv_cache_path(eval_root: Path, split_name: str) -> Path:
    path = eval_root / "conserv_force_cache" / f"oc20_conserv_7model_{split_name}_forces.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing conservative force cache for split={split_name}: {path}")
    return path


def assert_same_indices(reference: list[int], candidate: list[int], label: str) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"sample_indices length mismatch for {label}: {len(reference)} vs {len(candidate)}")
    if reference != candidate:
        raise ValueError(f"sample_indices mismatch for {label}")


def graph_mean(values_per_atom: torch.Tensor, natoms: torch.Tensor) -> torch.Tensor:
    graph_ids = torch.arange(natoms.numel(), dtype=torch.long).repeat_interleave(natoms)
    sums = torch.zeros(natoms.numel(), dtype=torch.float64)
    sums.index_add_(0, graph_ids, values_per_atom.to(torch.float64))
    return sums / natoms.to(torch.float64).clamp_min(1)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom.item() == 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, x.numel() + 1, dtype=torch.float64)
    return ranks


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return pearson(rankdata(x), rankdata(y))


def quantile_bins(x: torch.Tensor, y: torch.Tensor, bins: int) -> list[dict[str, float]]:
    order = torch.argsort(x)
    rows: list[dict[str, float]] = []
    n = x.numel()
    for bin_idx in range(bins):
        start = round(bin_idx * n / bins)
        end = round((bin_idx + 1) * n / bins)
        idx = order[start:end]
        xb = x[idx].to(torch.float64)
        yb = y[idx].to(torch.float64)
        rows.append(
            {
                "bin": bin_idx + 1,
                "count": int(idx.numel()),
                "uncertainty_min": float(xb.min()),
                "uncertainty_max": float(xb.max()),
                "uncertainty_mean": float(xb.mean()),
                "conserv_force_mae_mean": float(yb.mean()),
                "conserv_force_mae_median": float(yb.median()),
            }
        )
    return rows


def load_split_metrics(eval_root: Path, split_name: str) -> dict[str, torch.Tensor | list[int]]:
    base_payloads = []
    for model_name in BASE_MODELS:
        payload = torch.load(force_cache_path(eval_root, model_name, split_name), map_location="cpu")
        base_payloads.append(payload)

    conserv = torch.load(conserv_cache_path(eval_root, split_name), map_location="cpu")

    sample_indices = list(base_payloads[0]["sample_indices"])
    natoms = base_payloads[0]["natoms"].to(torch.long)
    target_forces = base_payloads[0]["target_forces"]
    for model_name, payload in zip(BASE_MODELS, base_payloads):
        assert_same_indices(sample_indices, list(payload["sample_indices"]), f"{model_name}/{split_name}")
        if not torch.equal(natoms, payload["natoms"].to(torch.long)):
            raise ValueError(f"natoms mismatch for {model_name}/{split_name}")
        if not torch.allclose(target_forces, payload["target_forces"]):
            raise ValueError(f"target forces mismatch for {model_name}/{split_name}")

    assert_same_indices(sample_indices, list(conserv["sample_indices"]), f"conserv/{split_name}")
    if not torch.equal(natoms, conserv["natoms"].to(torch.long)):
        raise ValueError(f"conserv natoms mismatch for {split_name}")

    pred_stack = torch.stack([payload["pred_forces"] for payload in base_payloads], dim=0)
    base_std = pred_stack.std(dim=0, unbiased=False)
    del pred_stack

    component_mean_std = graph_mean(base_std.mean(dim=1), natoms)
    component_rms_std = graph_mean(torch.sqrt((base_std * base_std).mean(dim=1)), natoms)
    vector_std_norm = graph_mean(torch.linalg.vector_norm(base_std, dim=1), natoms)

    conserv_force_mae = graph_mean(
        torch.abs(conserv["pred_forces"] - conserv["target_forces"]).mean(dim=1),
        natoms,
    )

    return {
        "sample_indices": sample_indices,
        "natoms": natoms,
        "component_mean_std": component_mean_std,
        "component_rms_std": component_rms_std,
        "vector_std_norm": vector_std_norm,
        "conserv_force_mae": conserv_force_mae,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(output_dir: Path, split_name: str, x: torch.Tensor, y: torch.Tensor) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Skipping plots for {split_name}: {exc}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=180)
    ax.hexbin(x.numpy(), y.numpy(), gridsize=60, mincnt=1, cmap="Blues")
    ax.set_xlabel("7-base force std (component mean)")
    ax.set_ylabel("Conservative meta force MAE")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_uncertainty_vs_error_hexbin.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    all_uncertainty = []
    all_error = []

    for split_name in args.splits:
        metrics = load_split_metrics(args.eval_root.resolve(), split_name)
        component_mean_std = metrics["component_mean_std"]
        component_rms_std = metrics["component_rms_std"]
        vector_std_norm = metrics["vector_std_norm"]
        conserv_force_mae = metrics["conserv_force_mae"]

        torch.save(metrics, output_dir / f"per_structure_{split_name}.pt")

        per_structure_rows = []
        for idx, sample_index in enumerate(metrics["sample_indices"]):
            per_structure_rows.append(
                {
                    "sample_index": int(sample_index),
                    "natoms": int(metrics["natoms"][idx]),
                    "base_force_std_component_mean": float(component_mean_std[idx]),
                    "base_force_std_component_rms": float(component_rms_std[idx]),
                    "base_force_std_vector_norm": float(vector_std_norm[idx]),
                    "conserv_force_mae": float(conserv_force_mae[idx]),
                }
            )
        write_csv(
            output_dir / f"per_structure_{split_name}.csv",
            per_structure_rows,
            [
                "sample_index",
                "natoms",
                "base_force_std_component_mean",
                "base_force_std_component_rms",
                "base_force_std_vector_norm",
                "conserv_force_mae",
            ],
        )

        for metric_name, uncertainty in (
            ("component_mean_std", component_mean_std),
            ("component_rms_std", component_rms_std),
            ("vector_std_norm", vector_std_norm),
        ):
            summary_rows.append(
                {
                    "split": split_name,
                    "uncertainty_metric": metric_name,
                    "sample_count": int(conserv_force_mae.numel()),
                    "uncertainty_mean": float(uncertainty.to(torch.float64).mean()),
                    "uncertainty_median": float(uncertainty.to(torch.float64).median()),
                    "conserv_force_mae_mean": float(conserv_force_mae.to(torch.float64).mean()),
                    "conserv_force_mae_median": float(conserv_force_mae.to(torch.float64).median()),
                    "pearson": pearson(uncertainty, conserv_force_mae),
                    "spearman": spearman(uncertainty, conserv_force_mae),
                }
            )

        bin_rows = quantile_bins(component_mean_std, conserv_force_mae, args.bins)
        for row in bin_rows:
            row["split"] = split_name
            row["uncertainty_metric"] = "component_mean_std"
        write_csv(
            output_dir / f"bins_{split_name}.csv",
            bin_rows,
            [
                "split",
                "uncertainty_metric",
                "bin",
                "count",
                "uncertainty_min",
                "uncertainty_max",
                "uncertainty_mean",
                "conserv_force_mae_mean",
                "conserv_force_mae_median",
            ],
        )

        if args.make_plots:
            maybe_plot(output_dir, split_name, component_mean_std, conserv_force_mae)

        all_uncertainty.append(component_mean_std)
        all_error.append(conserv_force_mae)
        print(
            f"{split_name}: pearson={pearson(component_mean_std, conserv_force_mae):.4f}, "
            f"spearman={spearman(component_mean_std, conserv_force_mae):.4f}",
            flush=True,
        )

    combined_uncertainty = torch.cat(all_uncertainty)
    combined_error = torch.cat(all_error)
    summary_rows.append(
        {
            "split": "combined",
            "uncertainty_metric": "component_mean_std",
            "sample_count": int(combined_error.numel()),
            "uncertainty_mean": float(combined_uncertainty.to(torch.float64).mean()),
            "uncertainty_median": float(combined_uncertainty.to(torch.float64).median()),
            "conserv_force_mae_mean": float(combined_error.to(torch.float64).mean()),
            "conserv_force_mae_median": float(combined_error.to(torch.float64).median()),
            "pearson": pearson(combined_uncertainty, combined_error),
            "spearman": spearman(combined_uncertainty, combined_error),
        }
    )

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "split",
            "uncertainty_metric",
            "sample_count",
            "uncertainty_mean",
            "uncertainty_median",
            "conserv_force_mae_mean",
            "conserv_force_mae_median",
            "pearson",
            "spearman",
        ],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
