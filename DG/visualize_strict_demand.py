#!/usr/bin/env python
"""Plot mean demand and reproducibly sampled demand profiles from a UC dataset."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=SCRIPT_DIR / "uc_new_data_strict.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "strict_demand_profiles.png",
    )
    parser.add_argument("--expected-samples", type=int, default=50_000)
    parser.add_argument("--num-random-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_demand(path):
    with np.load(path) as data:
        demand = np.asarray(data["X_demand"], dtype=np.float32)
    if demand.ndim != 3 or demand.shape[-1] != 1:
        raise ValueError(f"Expected X_demand shape (samples, hours, 1), got {demand.shape}")
    return demand.squeeze(-1)


def main():
    args = parse_args()
    demand = load_demand(args.dataset)
    num_samples, num_hours = demand.shape
    if num_samples != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples:,} samples, got {num_samples:,}"
        )
    if not 1 <= args.num_random_samples <= num_samples:
        raise ValueError("--num-random-samples must be between 1 and the dataset size")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(
        num_samples, size=args.num_random_samples, replace=False
    )
    hourly_mean = np.mean(demand, axis=0)
    hours = np.arange(1, num_hours + 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for index in sample_indices:
        ax.plot(
            hours,
            demand[index],
            color="tab:blue",
            alpha=0.35,
            linewidth=1.1,
            label="Random samples" if index == sample_indices[0] else None,
        )
    ax.plot(
        hours,
        hourly_mean,
        color="black",
        linewidth=3,
        label="Hourly mean demand",
    )
    ax.set(
        title="Strict UC Dataset Demand Profiles",
        xlabel="Hour",
        ylabel="Demand (MW)",
        xticks=hours,
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    print(f"Dataset: {args.dataset}")
    print(f"Samples: {num_samples:,}")
    print(f"Hours per sample: {num_hours}")
    print(f"Overall mean demand: {np.mean(demand):,.2f} MW")
    print(f"Random sample indices (seed={args.seed}): {sample_indices.tolist()}")
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
