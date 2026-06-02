#!/usr/bin/env python
"""Print a Markdown summary row for a completed training experiment."""

import argparse
import json
from datetime import date
from pathlib import Path


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--description", default="TODO")
    parser.add_argument("--notes", default="TODO")
    args = parser.parse_args()

    config = load_json(args.output_dir / "run_configuration.json")
    metrics = load_json(args.output_dir / "evaluation.json")
    job_id = args.output_dir.name.removeprefix("rnncell_strict_allocation_")
    job_id = job_id.removeprefix("rnncell_strict_")
    job_id = job_id.removeprefix("rnncell_allocation_")
    job_id = job_id.removeprefix("rnncell_")
    commit = config.get("git_commit", "unknown")

    print(
        f"| {date.today().isoformat()} | {job_id} | {commit} | "
        f"{args.description} | "
        f"{metrics['status_accuracy_percent']:.2f}% | "
        f"{metrics['power_mae_mw']:.2f} MW | "
        f"{metrics['mismatch_mae_mw']:.2f} MW | "
        f"{metrics['mismatch_percent_of_mean_demand']:.2f}% | "
        f"{metrics['ai_average_daily_cost']:,.2f} | "
        f"{args.notes} |"
    )


if __name__ == "__main__":
    main()
