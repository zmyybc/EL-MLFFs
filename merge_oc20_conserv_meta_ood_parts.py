from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge OC20 conserv-meta OOD part results.")
    parser.add_argument("--ads-tsv", type=Path, required=True)
    parser.add_argument("--cat-tsv", type=Path, required=True)
    parser.add_argument("--both-half1-tsv", type=Path, required=True)
    parser.add_argument("--both-half2-tsv", type=Path, required=True)
    parser.add_argument("--both-half1-force", type=Path, required=True)
    parser.add_argument("--both-half2-force", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-force", type=Path, required=True)
    return parser.parse_args()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def weighted_merge(row1: dict[str, str], row2: dict[str, str]) -> dict[str, str]:
    n1 = int(row1["sample_count"])
    n2 = int(row2["sample_count"])
    e1 = float(row1["energy_mae"])
    e2 = float(row2["energy_mae"])
    f1 = float(row1["force_mae"])
    f2 = float(row2["force_mae"])
    total = n1 + n2
    merged = dict(row1)
    merged["sample_count"] = str(total)
    merged["energy_mae"] = f"{((e1 * n1 + e2 * n2) / total):.12f}"
    merged["force_mae"] = f"{((f1 * n1 + f2 * n2) / total):.12f}"
    merged["force_cache"] = str(args.output_force.resolve())
    return merged


def main() -> None:
    global args
    args = parse_args()

    ads = read_single_row(args.ads_tsv)
    cat = read_single_row(args.cat_tsv)
    both1 = read_single_row(args.both_half1_tsv)
    both2 = read_single_row(args.both_half2_tsv)
    both = weighted_merge(both1, both2)
    both["split"] = "ood_both"

    payload1 = torch.load(args.both_half1_force, map_location="cpu")
    payload2 = torch.load(args.both_half2_force, map_location="cpu")

    merged_force = {
        "model": payload1["model"],
        "split": "ood_both",
        "sample_count": payload1["sample_count"] + payload2["sample_count"],
        "sample_indices": list(payload1["sample_indices"]) + list(payload2["sample_indices"]),
        "pred_forces": torch.cat([payload1["pred_forces"], payload2["pred_forces"]], dim=0),
        "target_forces": torch.cat([payload1["target_forces"], payload2["target_forces"]], dim=0),
        "natoms": torch.cat([payload1["natoms"], payload2["natoms"]], dim=0),
        "checkpoint": payload1["checkpoint"],
    }

    args.output_force.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_force, args.output_force)

    rows = [ads, cat, both]
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "split", "sample_count", "energy_mae", "force_mae", "checkpoint", "force_cache"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved merged TSV to: {args.output_tsv}")
    print(f"Saved merged force cache to: {args.output_force}")


if __name__ == "__main__":
    main()
