#!/usr/bin/env python
"""Validate physical constraints in a strict UC training dataset."""

import argparse
import csv
from pathlib import Path

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
        "--specs",
        type=Path,
        default=SCRIPT_DIR / "generator_specs.csv",
    )
    parser.add_argument("--tolerance-mw", type=float, default=1e-3)
    parser.add_argument("--limit-samples", type=int)
    return parser.parse_args()


def load_specs(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    def values(column):
        return np.asarray([row[column] for row in rows], dtype=np.float64)

    return {
        "p_max": values("MaxProd"),
        "p_min": values("MinProd"),
        "init_power": values("IniProd"),
        "init_state": values("IniState"),
        "init_status": (values("IniState") > 0.0).astype(np.float64),
        "su_cap": values("SUcap"),
        "sd_cap": values("SDcap"),
        "ramp_up": values("RampUp"),
        "ramp_down": values("RampDw"),
        "min_up": values("MinTU").astype(np.int64),
        "min_down": values("MinTD").astype(np.int64),
    }


def report(name, values, tolerance):
    max_value = float(np.max(values))
    count = int(np.count_nonzero(values > tolerance))
    print(f"{name}: max={max_value:.6g}, violations={count}")
    return count


def count_minimum_time_violations(status, specs):
    num_samples, num_hours, num_gens = status.shape
    init_status = specs["init_status"]
    previous_status = np.concatenate(
        (
            np.repeat(init_status[np.newaxis, np.newaxis, :], num_samples, axis=0),
            status[:, :-1, :],
        ),
        axis=1,
    )
    startup = (previous_status < 0.5) & (status > 0.5)
    shutdown = (previous_status > 0.5) & (status < 0.5)
    violation_count = 0

    for generator in range(num_gens):
        min_up = specs["min_up"][generator]
        min_down = specs["min_down"][generator]
        for hour in range(num_hours):
            if hour + min_up > num_hours:
                violation_count += int(np.count_nonzero(startup[:, hour, generator]))
            elif min_up > 0:
                stays_on = np.all(status[:, hour : hour + min_up, generator] > 0.5, axis=1)
                violation_count += int(np.count_nonzero(startup[:, hour, generator] & ~stays_on))

            if hour + min_down > num_hours:
                violation_count += int(np.count_nonzero(shutdown[:, hour, generator]))
            elif min_down > 0:
                stays_off = np.all(status[:, hour : hour + min_down, generator] < 0.5, axis=1)
                violation_count += int(np.count_nonzero(shutdown[:, hour, generator] & ~stays_off))

        elapsed = abs(specs["init_state"][generator])
        if specs["init_status"][generator] > 0.5:
            remaining = int(np.ceil(max(0.0, min_up - elapsed)))
            if remaining:
                violation_count += int(
                    np.count_nonzero(np.any(status[:, :remaining, generator] < 0.5, axis=1))
                )
        else:
            remaining = int(np.ceil(max(0.0, min_down - elapsed)))
            if remaining:
                violation_count += int(
                    np.count_nonzero(np.any(status[:, :remaining, generator] > 0.5, axis=1))
                )

    return violation_count


def main():
    args = parse_args()
    specs = load_specs(args.specs)
    with np.load(args.dataset) as data:
        sample_slice = slice(None, args.limit_samples)
        demand = np.asarray(data["X_demand"][sample_slice], dtype=np.float64).squeeze(-1)
        power = np.asarray(data["Y_power"][sample_slice], dtype=np.float64)
        status_raw = np.asarray(data["Y_status"][sample_slice], dtype=np.float64)

    status = (status_raw > 0.5).astype(np.float64)
    num_samples, num_hours, num_gens = power.shape
    expected_shape = (num_samples, num_hours, num_gens)
    if demand.shape != (num_samples, num_hours):
        raise ValueError(f"Unexpected demand shape: {demand.shape}")
    if status.shape != expected_shape:
        raise ValueError(f"Unexpected status shape: {status.shape}")
    if num_gens != len(specs["p_max"]):
        raise ValueError(f"Dataset has {num_gens} generators, specs have {len(specs['p_max'])}")

    init_status = specs["init_status"]
    init_power = specs["init_power"]
    previous_status = np.concatenate(
        (
            np.repeat(init_status[np.newaxis, np.newaxis, :], num_samples, axis=0),
            status[:, :-1, :],
        ),
        axis=1,
    )
    previous_power = np.concatenate(
        (
            np.repeat(init_power[np.newaxis, np.newaxis, :], num_samples, axis=0),
            power[:, :-1, :],
        ),
        axis=1,
    )
    online = status > 0.5
    startup = (previous_status < 0.5) & online
    shutdown = (previous_status > 0.5) & ~online
    stay_online = (previous_status > 0.5) & online

    print(f"samples={num_samples}, hours={num_hours}, generators={num_gens}")
    violations = 0
    violations += report(
        "balance_mismatch_mw",
        np.abs(np.sum(power, axis=-1) - demand),
        args.tolerance_mw,
    )
    violations += report(
        "status_integrality",
        np.abs(status_raw - status),
        1e-5,
    )
    violations += report(
        "offline_generation_mw",
        np.maximum(power, 0.0) * ~online,
        args.tolerance_mw,
    )
    violations += report(
        "negative_generation_mw",
        np.maximum(-power, 0.0),
        args.tolerance_mw,
    )
    violations += report(
        "capacity_upper_mw",
        np.maximum(power - specs["p_max"], 0.0) * online,
        args.tolerance_mw,
    )
    violations += report(
        "capacity_lower_mw",
        np.maximum(specs["p_min"] - power, 0.0) * online,
        args.tolerance_mw,
    )
    violations += report(
        "stay_online_ramp_up_mw",
        np.maximum(power - previous_power - specs["ramp_up"], 0.0) * stay_online,
        args.tolerance_mw,
    )
    violations += report(
        "stay_online_ramp_down_mw",
        np.maximum(previous_power - power - specs["ramp_down"], 0.0) * stay_online,
        args.tolerance_mw,
    )
    violations += report(
        "startup_cap_mw",
        np.maximum(power - specs["su_cap"], 0.0) * startup,
        args.tolerance_mw,
    )
    violations += report(
        "shutdown_cap_mw",
        np.maximum(previous_power - specs["sd_cap"], 0.0) * shutdown,
        args.tolerance_mw,
    )
    minimum_time_violations = count_minimum_time_violations(status, specs)
    print(f"minimum_time_violations: {minimum_time_violations}")
    violations += minimum_time_violations

    if violations:
        raise SystemExit(f"Validation failed with {violations} violations")
    print("Validation passed")


if __name__ == "__main__":
    main()
