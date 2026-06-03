#!/usr/bin/env python
"""Train and evaluate the strict ramp-aware proportional-allocation UC RNN."""

import csv
import inspect
import json

import numpy as np

from legacy.train_rnncell import (
    SCRIPT_DIR,
    calculate_average_cost,
    configure_environment,
    get_git_commit,
    load_dataset,
    load_specs,
    make_static_features,
    parse_args,
    save_json,
    set_deterministic_seed,
    split_indices,
    train_two_phases,
    validate_shapes,
)
from DG.validate_strict_uc_dataset import count_minimum_time_violations


def load_strict_specs(path):
    specs = load_specs(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    def values(column):
        return np.asarray([row[column] for row in rows], dtype=np.float32)

    init_state = values("IniState")
    specs.update(
        {
            "init_state_hours": init_state,
            "init_up_hours": np.maximum(init_state, 0.0),
            "init_down_hours": np.maximum(-init_state, 0.0),
            "startup_cap": values("SUcap"),
            "shutdown_cap": values("SDcap"),
        }
    )
    if np.any(specs["startup_cap"] < specs["p_min"]) or np.any(
        specs["startup_cap"] > specs["p_max"]
    ):
        raise ValueError("SUcap must be between MinProd and MaxProd")
    if np.any(specs["shutdown_cap"] < specs["p_min"]) or np.any(
        specs["shutdown_cap"] > specs["p_max"]
    ):
        raise ValueError("SDcap must be between MinProd and MaxProd")
    return specs


def make_strict_model_inputs(demand, specs, static_features):
    num_samples = len(demand)

    def repeat(values):
        return np.repeat(values[np.newaxis, :], num_samples, axis=0)

    return [
        demand,
        repeat(static_features),
        repeat(specs["init_status"]),
        repeat(specs["init_power_mw"]),
        repeat(specs["init_up_hours"]),
        repeat(specs["init_down_hours"]),
    ]


def evaluate_strict(model, test_inputs, demand, true_status, true_power, specs, verbose):
    pred_status, pred_power = model.predict(test_inputs, verbose=verbose)
    pred_status_bin = (pred_status > 0.5).astype(np.float32)
    num_samples = len(pred_status_bin)
    mean_demand = float(np.mean(demand))

    previous_status = np.concatenate(
        (
            np.repeat(specs["init_status"][np.newaxis, np.newaxis, :], num_samples, axis=0),
            pred_status_bin[:, :-1, :],
        ),
        axis=1,
    )
    previous_power = np.concatenate(
        (
            np.repeat(specs["init_power_mw"][np.newaxis, np.newaxis, :], num_samples, axis=0),
            pred_power[:, :-1, :],
        ),
        axis=1,
    )
    online = pred_status_bin > 0.5
    startup = (previous_status < 0.5) & online
    shutdown = (previous_status > 0.5) & ~online
    stay_online = (previous_status > 0.5) & online

    pred_total_power = np.sum(pred_power, axis=-1)
    power_balance_error = pred_total_power - demand.squeeze(-1)
    absolute_mismatch = np.abs(power_balance_error)
    mismatch = float(np.mean(absolute_mismatch))
    ghost = np.maximum(pred_power, 0.0) * ~online
    capacity = (
        np.maximum(pred_power - specs["p_max"], 0.0)
        + np.maximum(specs["p_min"] - pred_power, 0.0)
    ) * online
    ramp = (
        np.maximum(pred_power - previous_power - specs["ramp_up"], 0.0)
        + np.maximum(previous_power - pred_power - specs["ramp_down"], 0.0)
    ) * stay_online
    startup_cap = np.maximum(pred_power - specs["startup_cap"], 0.0) * startup
    shutdown_cap = np.maximum(previous_power - specs["shutdown_cap"], 0.0) * shutdown
    validation_specs = {
        "init_status": specs["init_status"],
        "init_state": specs["init_state_hours"],
        "min_up": specs["mut"],
        "min_down": specs["mdt"],
    }

    return {
        "status_accuracy_percent": float(np.mean(pred_status_bin == true_status) * 100),
        "power_mae_mw": float(np.mean(np.abs(pred_power - true_power))),
        "mismatch_mae_mw": mismatch,
        "mismatch_percent_of_mean_demand": mismatch / mean_demand * 100,
        "mismatch_max_mw": float(np.max(absolute_mismatch)),
        "mismatch_p95_mw": float(np.percentile(absolute_mismatch, 95)),
        "mismatch_p99_mw": float(np.percentile(absolute_mismatch, 99)),
        "mismatch_over_10mw_percent": float(np.mean(absolute_mismatch > 10.0) * 100),
        "shortage_mae_mw": float(np.mean(np.maximum(-power_balance_error, 0.0))),
        "excess_mae_mw": float(np.mean(np.maximum(power_balance_error, 0.0))),
        "ghost_power_mw": float(np.mean(np.sum(ghost, axis=-1))),
        "capacity_violation_mw": float(np.mean(np.sum(capacity, axis=-1))),
        "ramp_violation_mw": float(np.mean(np.sum(ramp, axis=-1))),
        "startup_cap_violation_mw": float(np.mean(np.sum(startup_cap, axis=-1))),
        "shutdown_cap_violation_mw": float(np.mean(np.sum(shutdown_cap, axis=-1))),
        "minimum_time_violations_total": count_minimum_time_violations(
            pred_status_bin, validation_specs
        ),
        "cplex_average_daily_cost": calculate_average_cost(true_status, true_power, specs),
        "ai_average_daily_cost": calculate_average_cost(pred_status_bin, pred_power, specs),
    }


def train_and_evaluate(
    build_model,
    *,
    default_output_name,
    saved_model_name,
    model_variant,
):
    args = parse_args()
    default_data = SCRIPT_DIR / "DG" / "uc_new_data.npz"
    default_output_dir = SCRIPT_DIR / "outputs" / "rnncell"
    if args.data == default_data:
        args.data = SCRIPT_DIR / "DG" / "uc_new_data_strict.npz"
    if args.output_dir == default_output_dir:
        args.output_dir = SCRIPT_DIR / "outputs" / default_output_name
    configure_environment(args.seed)

    import tensorflow as tf
    set_deterministic_seed(tf, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_strict_specs(args.specs)
    demand, status, power = load_dataset(args.data, args.limit_samples)
    validate_shapes(demand, status, power, specs)
    train_indices, val_indices, test_indices = split_indices(len(demand), args.seed)
    print(
        f"Dataset loaded: train={len(train_indices)}, val={len(val_indices)}, "
        f"test={len(test_indices)}",
        flush=True,
    )

    static_features = make_static_features(specs)
    train_inputs = make_strict_model_inputs(demand[train_indices], specs, static_features)
    val_inputs = make_strict_model_inputs(demand[val_indices], specs, static_features)
    train_targets = {
        "out_status": status[train_indices],
        "out_power": power[train_indices],
    }
    val_targets = {
        "out_status": status[val_indices],
        "out_power": power[val_indices],
    }

    power_normalizer_mw = float(np.mean(specs["p_max"]))
    demand_normalizer_mw = float(np.mean(demand[train_indices]))
    print(
        f"Loss normalizers: average generator capacity={power_normalizer_mw:.2f} MW, "
        f"average training demand={demand_normalizer_mw:.2f} MW",
        flush=True,
    )
    model_kwargs = {}
    if "lookahead_safety_margin_mw" in inspect.signature(build_model).parameters:
        model_kwargs["lookahead_safety_margin_mw"] = args.lookahead_safety_margin_mw

    model = build_model(
        specs,
        demand_normalizer_mw=demand_normalizer_mw,
        balance_loss_weight=args.phase1_balance_loss_weight,
        num_hours=demand.shape[1],
        **model_kwargs,
    )
    model.summary()
    phase1_history, phase2_history = train_two_phases(
        tf,
        model,
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        power_normalizer_mw,
        demand_normalizer_mw,
        args,
    )
    model.save(args.output_dir / saved_model_name)
    save_json(
        args.output_dir / "training_history.json",
        {"phase1": phase1_history, "phase2": phase2_history},
    )
    save_json(
        args.output_dir / "run_configuration.json",
        {
            **vars(args),
            "data": str(args.data),
            "specs": str(args.specs),
            "output_dir": str(args.output_dir),
            "git_commit": get_git_commit(),
            "model_variant": model_variant,
            "power_normalizer_mw": power_normalizer_mw,
            "demand_normalizer_mw": demand_normalizer_mw,
        },
    )

    if not args.skip_evaluation:
        test_inputs = make_strict_model_inputs(demand[test_indices], specs, static_features)
        metrics = evaluate_strict(
            model,
            test_inputs,
            demand[test_indices],
            status[test_indices],
            power[test_indices],
            specs,
            args.verbose,
        )
        save_json(args.output_dir / "evaluation.json", metrics)
        print("\n=== Test evaluation ===", flush=True)
        print(json.dumps(metrics, indent=2), flush=True)

    print(f"\nArtifacts saved in: {args.output_dir}", flush=True)


def main():
    from rnncell_strict_allocation_model import build_hybrid_uc_strict_allocation_model

    train_and_evaluate(
        build_hybrid_uc_strict_allocation_model,
        default_output_name="rnncell_strict_allocation",
        saved_model_name="saved_uc_RC_strict_allocation_model.keras",
        model_variant="strict_ramp_aware_proportional_allocation",
    )


if __name__ == "__main__":
    main()
