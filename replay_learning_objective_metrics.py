#!/usr/bin/env python
"""Replay saved learning-objective models and record commitment diagnostics."""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

# Import modules so Keras can deserialize registered custom layers.
import legacy.rnncell_model  # noqa: F401
import rnncell_strict_allocation_model  # noqa: F401
import rnncell_strict_allocation_multistep_ramp_position_model  # noqa: F401
from legacy.train_rnncell import (
    SCRIPT_DIR,
    configure_environment,
    load_dataset,
    make_static_features,
    set_deterministic_seed,
    split_indices,
)
from train_rnncell_strict_allocation import (
    load_strict_specs,
    make_strict_model_inputs,
)


RUNS = [
    ("baseline", "0.0", "30714", "outputs/rnncell_strict_allocation_multistep_ramp_position_30714"),
    ("asym_bce", "0.5", "39104", "outputs/rnncell_strict_asym_bce_alpha0.5_2gpu_39104"),
    ("asym_bce", "1.0", "40234", "outputs/rnncell_strict_asym_bce_alpha1.0_2gpu_40234"),
    ("asym_bce", "1.5", "41987", "outputs/rnncell_strict_asym_bce_alpha1.5_2gpu_41987"),
    ("transition", "0.5", "42010", "outputs/rnncell_strict_transition_w0.5_2gpu_42010"),
    ("transition", "1.0", "42043", "outputs/rnncell_strict_transition_w1.0_2gpu_42043"),
    ("transition", "1.5", "42059", "outputs/rnncell_strict_transition_w1.5_2gpu_42059"),
    ("transition", "2.0", "42060", "outputs/rnncell_strict_transition_w2_2gpu_42060"),
    ("transition", "5.0", "42061", "outputs/rnncell_strict_transition_w5_2gpu_42061"),
    ("transition", "10.0", "42062", "outputs/rnncell_strict_transition_w10_2gpu_42062"),
    ("online_hours", "0.05", "42765", "outputs/rnncell_strict_online_hours_w0.05_2gpu_42765"),
    ("online_hours", "0.1", "58498", "outputs/rnncell_strict_online_hours_w0.1_2gpu_58498"),
    ("online_hours", "0.2", "58816", "outputs/rnncell_strict_online_hours_w0.2_2gpu_58816"),
    ("online_hours", "0.3", "59479", "outputs/rnncell_strict_online_hours_w0.3_2gpu_59479"),
    ("online_hours", "0.5", "59629", "outputs/rnncell_strict_online_hours_w0.5_2gpu_59629"),
    ("online_hours", "1.0", "59673", "outputs/rnncell_strict_online_hours_w1.0_2gpu_59673"),
]


FIELDNAMES = [
    "family",
    "weight",
    "job",
    "status_accuracy_percent",
    "power_mae_mw",
    "cost_gap_percent",
    "ai_average_daily_cost",
    "cost_delta_vs_30714",
    "false_on_rate_percent",
    "false_off_rate_percent",
    "false_on_count_per_day",
    "false_off_count_per_day",
    "online_hours_true",
    "online_hours_pred",
    "online_hours_delta",
    "startup_count_true",
    "startup_count_pred",
    "startup_count_delta",
    "shutdown_count_true",
    "shutdown_count_pred",
    "shutdown_count_delta",
    "startup_event_mae",
    "shutdown_event_mae",
    "transition_event_mae",
    "mismatch_max_mw",
    "mismatch_over_10mw_percent",
]


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def model_path(output_dir):
    candidates = sorted(output_dir.glob("saved_uc_RC_*_model.keras"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one saved model in {output_dir}")
    return candidates[0]


def transition_metrics(pred_status_bin, true_status, init_status):
    num_samples = len(pred_status_bin)
    initial = np.repeat(init_status[np.newaxis, np.newaxis, :], num_samples, axis=0)
    pred_previous = np.concatenate([initial, pred_status_bin[:, :-1, :]], axis=1)
    true_previous = np.concatenate([initial, true_status[:, :-1, :]], axis=1)

    pred_startup = ((pred_previous < 0.5) & (pred_status_bin > 0.5)).astype(np.float32)
    pred_shutdown = ((pred_previous > 0.5) & (pred_status_bin < 0.5)).astype(np.float32)
    true_startup = ((true_previous < 0.5) & (true_status > 0.5)).astype(np.float32)
    true_shutdown = ((true_previous > 0.5) & (true_status < 0.5)).astype(np.float32)
    false_on = ((pred_status_bin > 0.5) & (true_status < 0.5)).astype(np.float32)
    false_off = ((pred_status_bin < 0.5) & (true_status > 0.5)).astype(np.float32)

    startup_event_mae = float(np.mean(np.abs(pred_startup - true_startup)))
    shutdown_event_mae = float(np.mean(np.abs(pred_shutdown - true_shutdown)))
    return {
        "false_on_rate_percent": float(np.mean(false_on) * 100.0),
        "false_off_rate_percent": float(np.mean(false_off) * 100.0),
        "false_on_count_per_day": float(np.mean(np.sum(false_on, axis=(1, 2)))),
        "false_off_count_per_day": float(np.mean(np.sum(false_off, axis=(1, 2)))),
        "online_hours_true": float(np.mean(np.sum(true_status, axis=(1, 2)))),
        "online_hours_pred": float(np.mean(np.sum(pred_status_bin, axis=(1, 2)))),
        "startup_count_true": float(np.mean(np.sum(true_startup, axis=(1, 2)))),
        "startup_count_pred": float(np.mean(np.sum(pred_startup, axis=(1, 2)))),
        "shutdown_count_true": float(np.mean(np.sum(true_shutdown, axis=(1, 2)))),
        "shutdown_count_pred": float(np.mean(np.sum(pred_shutdown, axis=(1, 2)))),
        "startup_event_mae": startup_event_mae,
        "shutdown_event_mae": shutdown_event_mae,
        "transition_event_mae": startup_event_mae + shutdown_event_mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "DG" / "uc_new_data_strict.npz")
    parser.add_argument("--specs", type=Path, default=SCRIPT_DIR / "DG" / "generator_specs.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "LEARNING_OBJECTIVE_REPLAY_METRICS.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    configure_environment(args.seed)
    import tensorflow as tf

    set_deterministic_seed(tf, args.seed)
    specs = load_strict_specs(args.specs)
    demand, status, power = load_dataset(args.data)
    _, _, test_indices = split_indices(len(demand), args.seed)
    static_features = make_static_features(specs)
    test_inputs = make_strict_model_inputs(
        demand[test_indices], specs, static_features
    )
    true_status = status[test_indices]

    baseline_eval = load_json(
        SCRIPT_DIR
        / "outputs"
        / "rnncell_strict_allocation_multistep_ramp_position_30714"
        / "evaluation.json"
    )
    baseline_cost = baseline_eval["ai_average_daily_cost"]

    rows = []
    for family, weight, job, output_dir_name in RUNS:
        output_dir = SCRIPT_DIR / output_dir_name
        evaluation = load_json(output_dir / "evaluation.json")
        print(f"Replaying {job}: {output_dir.name}", flush=True)
        model = tf.keras.models.load_model(model_path(output_dir), compile=False)
        pred_status, _ = model.predict(test_inputs, batch_size=args.batch_size, verbose=0)
        pred_status_bin = (pred_status > 0.5).astype(np.float32)
        metrics = transition_metrics(pred_status_bin, true_status, specs["init_status"])
        ai_cost = evaluation["ai_average_daily_cost"]
        cplex_cost = evaluation["cplex_average_daily_cost"]
        row = {
            "family": family,
            "weight": weight,
            "job": job,
            "status_accuracy_percent": evaluation["status_accuracy_percent"],
            "power_mae_mw": evaluation["power_mae_mw"],
            "cost_gap_percent": (ai_cost - cplex_cost) / cplex_cost * 100.0,
            "ai_average_daily_cost": ai_cost,
            "cost_delta_vs_30714": ai_cost - baseline_cost,
            "mismatch_max_mw": evaluation["mismatch_max_mw"],
            "mismatch_over_10mw_percent": evaluation["mismatch_over_10mw_percent"],
            **metrics,
        }
        row["online_hours_delta"] = row["online_hours_pred"] - row["online_hours_true"]
        row["startup_count_delta"] = row["startup_count_pred"] - row["startup_count_true"]
        row["shutdown_count_delta"] = row["shutdown_count_pred"] - row["shutdown_count_true"]
        rows.append(row)
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
