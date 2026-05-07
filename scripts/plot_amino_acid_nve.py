#!/usr/bin/env python
"""Plot amino acid NVE trajectories from CSV."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_nve(csv_path: str, out_prefix: str, models: list[str] | None = None) -> None:
    """Read NVE CSV and plot energy drift + temperature."""
    if models is None:
        models = ["mtp", "nep", "soap", "mace", "schnet"]

    colors = {
        "mtp": "#2ca02c",
        "nep": "#1f77b4",
        "soap": "#ff7f0e",
        "mace": "#d62728",
        "schnet": "#9467bd",
    }
    labels = {
        "mtp": "MTP",
        "nep": "NEP",
        "soap": "SOAP",
        "mace": "Meta",
        "schnet": "SchNet",
    }

    data = {m: {"time": [], "drift": [], "temp": []} for m in models}

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            model = row[0]
            if model not in data:
                continue
            data[model]["time"].append(float(row[2]))
            data[model]["drift"].append(float(row[6])*100)
            data[model]["temp"].append(float(row[8]))

    # --- Energy drift ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in models:
        ax.plot(
            data[model]["time"],
            data[model]["drift"],
            label=labels[model],
            color=colors[model],
            linewidth=0.8,
            alpha=0.8,
        )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel("Energy drift (meV/atom)", fontsize=12)
    ax.set_title("Amino acid NVE: Energy drift vs time", fontsize=13)
    ax.legend(loc="upper left", ncol=len(models))
    ax.set_xlim(0, max(data[models[0]]["time"]) if data[models[0]]["time"] else 1000000)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_drift.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_prefix}_drift.png")

    # --- Temperature ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for model in models:
        ax.plot(
            data[model]["time"],
            data[model]["temp"],
            label=labels[model],
            color=colors[model],
            linewidth=0.8,
            alpha=0.8,
        )
    ax.axhline(y=150, color="black", linestyle="--", linewidth=0.5, label="Initial 150K")
    ax.set_xlabel("Time (fs)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.set_title("Amino acid NVE: Temperature vs time", fontsize=13)
    ax.legend(loc="upper left", ncol=3)
    ax.set_xlim(0, max(data[models[0]]["time"]) if data[models[0]]["time"] else 1000000)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_temperature.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_prefix}_temperature.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot amino acid NVE trajectories.")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--out-prefix", required=True, help="Output file prefix")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mtp", "nep", "soap", "mace", "schnet"],
        help="Models to plot",
    )
    args = parser.parse_args()
    plot_nve(args.csv, args.out_prefix, args.models)


if __name__ == "__main__":
    main()
