from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge OC20 conservative meta val_id split halves.")
    parser.add_argument("--half1-tsv", type=Path, required=True)
    parser.add_argument("--half2-tsv", type=Path, required=True)
    parser.add_argument("--half1-force", type=Path, required=True)
    parser.add_argument("--half2-force", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-force", type=Path, required=True)
    return parser.parse_args()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def weighted_merge(row1: dict[str, str], row2: dict[str, str], output_force: Path) -> dict[str, str]:
    n1 = int(row1["sample_count"])
    n2 = int(row2["sample_count"])
    total = n1 + n2
    merged = dict(row1)
    merged["split"] = "val_id"
    merged["sample_count"] = str(total)
    merged["energy_mae"] = f"{((float(row1['energy_mae']) * n1 + float(row2['energy_mae']) * n2) / total):.12f}"
    merged["force_mae"] = f"{((float(row1['force_mae']) * n1 + float(row2['force_mae']) * n2) / total):.12f}"
    merged["force_cache"] = str(output_force.resolve())
    return merged


def main() -> None:
    args = parse_args()
    row1 = read_single_row(args.half1_tsv)
    row2 = read_single_row(args.half2_tsv)
    merged_row = weighted_merge(row1, row2, args.output_force)

    payload1 = torch.load(args.half1_force, map_location="cpu")
    payload2 = torch.load(args.half2_force, map_location="cpu")
    merged_force = {
        "model": payload1["model"],
        "split": "val_id",
        "tag": "",
        "sample_count": int(payload1["sample_count"]) + int(payload2["sample_count"]),
        "sample_indices": list(payload1["sample_indices"]) + list(payload2["sample_indices"]),
        "pred_forces": torch.cat([payload1["pred_forces"], payload2["pred_forces"]], dim=0),
        "target_forces": torch.cat([payload1["target_forces"], payload2["target_forces"]], dim=0),
        "natoms": torch.cat([payload1["natoms"], payload2["natoms"]], dim=0),
        "checkpoint": payload1["checkpoint"],
    }

    args.output_force.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_force, args.output_force)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "split", "sample_count", "energy_mae", "force_mae", "checkpoint", "force_cache"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(merged_row)

    print(f"Saved merged TSV to: {args.output_tsv}")
    print(f"Saved merged force cache to: {args.output_force}")
    print(
        f"val_id samples={merged_row['sample_count']} "
        f"energy_mae={merged_row['energy_mae']} force_mae={merged_row['force_mae']}"
    )


if __name__ == "__main__":
    main()
