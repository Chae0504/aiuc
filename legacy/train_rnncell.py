#!/usr/bin/env python
"""Train and evaluate the physics-informed UC RNN from uc_118_RNNcell.ipynb."""

import argparse
import contextlib
import csv
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent.parent
NUM_STATIC_FEATURES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the 118-bus physics-informed UC RNN."
    )
    parser.add_argument(
        "--data", type=Path, default=SCRIPT_DIR / "DG" / "uc_new_data.npz"
    )
    parser.add_argument(
        "--specs", type=Path, default=SCRIPT_DIR / "DG" / "generator_specs.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "rnncell"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--distribution-strategy",
        choices=["none", "mirrored"],
        default="none",
        help="TensorFlow distribution strategy. Use mirrored for one-process multi-GPU.",
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="Fail early unless TensorFlow sees exactly this many GPUs.",
    )
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=150)
    parser.add_argument("--phase1-patience", type=int, default=10)
    parser.add_argument("--phase2-patience", type=int, default=40)
    parser.add_argument("--phase1-learning-rate", type=float, default=1e-3)
    parser.add_argument("--phase2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--phase1-status-loss-weight", type=float, default=1.0)
    parser.add_argument("--phase1-power-loss-weight", type=float, default=0.0)
    parser.add_argument("--phase1-balance-loss-weight", type=float, default=0.25)
    parser.add_argument("--phase1-cost-loss-weight", type=float, default=0.0)
    parser.add_argument("--phase2-status-loss-weight", type=float, default=1.0)
    parser.add_argument("--phase2-power-loss-weight", type=float, default=1.0)
    parser.add_argument("--phase2-balance-loss-weight", type=float, default=2.0)
    parser.add_argument("--phase2-cost-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--status-loss-mode",
        choices=["bce", "cost_weighted_bce", "transition_bce"],
        default="bce",
        help=(
            "Status imitation loss. cost_weighted_bce penalizes false ONs more; "
            "transition_bce adds startup/shutdown timing imitation."
        ),
    )
    parser.add_argument(
        "--status-false-on-alpha",
        type=float,
        default=0.5,
        help=(
            "Extra false-ON weight scale for cost_weighted_bce. "
            "A generator's OFF-target weight is 1 + alpha * normalized_cost."
        ),
    )
    parser.add_argument(
        "--status-transition-loss-weight",
        type=float,
        default=0.5,
        help="Startup/shutdown transition MAE weight for transition_bce.",
    )
    parser.add_argument("--reduce-lr-patience", type=int, default=8)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--lookahead-safety-margin-mw",
        type=float,
        default=0.0,
        help=(
            "Extra MW reserve required by look-ahead shutdown checks. "
            "Used only by look-ahead repair variants."
        ),
    )
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Keras output mode. Use 2 for one summary line per epoch in SLURM logs.",
    )
    return parser.parse_args()


def configure_environment(seed):
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic_seed(tf, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    print(f"Deterministic seed configured: {seed}", flush=True)


def load_specs(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No generator specifications found in {path}")

    def values(column, dtype=np.float32):
        return np.asarray([row[column] for row in rows], dtype=dtype)

    return {
        "names": np.asarray([row["Generator"] for row in rows]),
        "p_max": values("MaxProd"),
        "p_min": values("MinProd"),
        "init_power_mw": values("IniProd"),
        "init_status": (values("IniState") > 0).astype(np.float32),
        "linear_cost": values("SlopeVarCost"),
        "noload_cost": values("InterVarCost"),
        "startup_cost": values("SUcost1"),
        "ramp_up": values("RampUp"),
        "ramp_down": values("RampDw"),
        "mut": values("MinTU", dtype=np.int32),
        "mdt": values("MinTD", dtype=np.int32),
    }


def load_dataset(path, limit_samples=None):
    def load_array(data, key):
        values = data[key]
        if limit_samples is not None:
            values = values[:limit_samples]
        return np.asarray(values, dtype=np.float32)

    with np.load(path) as data:
        demand = load_array(data, "X_demand")
        status = (load_array(data, "Y_status") > 0.5).astype(np.float32)
        power = load_array(data, "Y_power")
    return demand, status, power


def validate_shapes(demand, status, power, specs):
    num_samples, num_hours, demand_features = demand.shape
    num_gens = len(specs["p_max"])
    expected_target_shape = (num_samples, num_hours, num_gens)
    if demand_features != 1:
        raise ValueError(f"Expected one demand feature, got {demand.shape}")
    if status.shape != expected_target_shape:
        raise ValueError(f"Unexpected Y_status shape: {status.shape}")
    if power.shape != expected_target_shape:
        raise ValueError(f"Unexpected Y_power shape: {power.shape}")


def split_indices(num_samples, seed):
    from sklearn.model_selection import train_test_split

    indices = np.arange(num_samples)
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.2, random_state=seed
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=seed
    )
    return train_indices, val_indices, test_indices


def make_static_features(specs):
    features = np.column_stack(
        (
            specs["init_status"],
            specs["init_power_mw"] / specs["p_max"],
            specs["linear_cost"] / specs["linear_cost"].max(),
            specs["noload_cost"] / specs["noload_cost"].max(),
        )
    )
    return features.astype(np.float32).ravel()


def make_model_inputs(demand, specs, static_features):
    num_samples = len(demand)

    def repeat(values):
        return np.repeat(values[np.newaxis, :], num_samples, axis=0)

    return [
        demand,
        repeat(static_features),
        repeat(specs["init_status"]),
        repeat(specs["init_power_mw"]),
    ]


def compile_model(
    tf,
    model,
    power_normalizer_mw,
    demand_normalizer_mw,
    learning_rate,
    status_loss_weight,
    power_loss_weight,
):
    from legacy.rnncell_model import NormalizedMeanAbsoluteError, PowerBalanceMismatchMAE

    mismatch_layer = model.get_layer("mismatch_loss_layer")
    status_loss = getattr(mismatch_layer, "status_loss", "binary_crossentropy")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "out_status": status_loss,
            "out_power": NormalizedMeanAbsoluteError(power_normalizer_mw),
        },
        loss_weights={
            "out_status": status_loss_weight,
            "out_power": power_loss_weight,
        },
        metrics={
            "out_status": [
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
            "out_power": [
                tf.keras.metrics.MeanAbsoluteError(name="mae_mw"),
                PowerBalanceMismatchMAE(name="mismatch_mae_mw"),
                PowerBalanceMismatchMAE(
                    normalizer=demand_normalizer_mw,
                    name="normalized_mismatch",
                ),
            ],
        },
    )


def make_callbacks(tf, output_dir, phase, monitor, patience, args, reduce_lr=False):
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / f"{phase}_best.keras",
            monitor=monitor,
            save_best_only=True,
            mode="min",
        ),
        tf.keras.callbacks.CSVLogger(
            output_dir / f"{phase}_training.csv"
        ),
    ]
    if reduce_lr:
        callbacks.insert(
            1,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=args.reduce_lr_factor,
                patience=args.reduce_lr_patience,
                min_lr=args.min_learning_rate,
                mode="min",
                verbose=1,
            ),
        )
    return callbacks


def print_loss_configuration(
    phase,
    status_loss_weight,
    power_loss_weight,
    balance_loss_weight,
    cost_loss_weight,
    power_normalizer_mw,
    demand_normalizer_mw,
    status_loss_mode,
):
    print(
        f"{phase} loss = {status_loss_weight:g} * {status_loss_mode}"
        f" + {power_loss_weight:g} * (power_MAE_MW / {power_normalizer_mw:.2f})"
        f" + {balance_loss_weight:g} * (mismatch_MAE_MW / {demand_normalizer_mw:.2f})"
        f" + {cost_loss_weight:g} * normalized_commitment_cost_proxy",
        flush=True,
    )


def train_two_phases(
    tf,
    model,
    train_inputs,
    train_targets,
    val_inputs,
    val_targets,
    power_normalizer_mw,
    demand_normalizer_mw,
    args,
    strategy=None,
):
    def strategy_scope():
        return strategy.scope() if strategy is not None else contextlib.nullcontext()

    mismatch_layer = model.get_layer("mismatch_loss_layer")
    print("\n=== Phase 1: status-focused training ===", flush=True)
    mismatch_layer.set_balance_loss_weight(args.phase1_balance_loss_weight)
    if hasattr(mismatch_layer, "set_cost_loss_weight"):
        mismatch_layer.set_cost_loss_weight(args.phase1_cost_loss_weight)
    print_loss_configuration(
        "Phase 1",
        args.phase1_status_loss_weight,
        args.phase1_power_loss_weight,
        args.phase1_balance_loss_weight,
        args.phase1_cost_loss_weight,
        power_normalizer_mw,
        demand_normalizer_mw,
        args.status_loss_mode,
    )
    with strategy_scope():
        compile_model(
            tf,
            model,
            power_normalizer_mw=power_normalizer_mw,
            demand_normalizer_mw=demand_normalizer_mw,
            learning_rate=args.phase1_learning_rate,
            status_loss_weight=args.phase1_status_loss_weight,
            power_loss_weight=args.phase1_power_loss_weight,
        )
    history_phase1 = model.fit(
        x=train_inputs,
        y=train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=args.phase1_epochs,
        batch_size=args.batch_size,
        callbacks=make_callbacks(
            tf,
            args.output_dir,
            "phase1",
            "val_out_status_loss",
            args.phase1_patience,
            args,
        ),
        verbose=args.verbose,
    )
    model.save(args.output_dir / "phase1_final.keras")

    print("\n=== Phase 2: hybrid fine-tuning ===", flush=True)
    mismatch_layer.set_balance_loss_weight(args.phase2_balance_loss_weight)
    if hasattr(mismatch_layer, "set_cost_loss_weight"):
        mismatch_layer.set_cost_loss_weight(args.phase2_cost_loss_weight)
    print_loss_configuration(
        "Phase 2",
        args.phase2_status_loss_weight,
        args.phase2_power_loss_weight,
        args.phase2_balance_loss_weight,
        args.phase2_cost_loss_weight,
        power_normalizer_mw,
        demand_normalizer_mw,
        args.status_loss_mode,
    )
    with strategy_scope():
        compile_model(
            tf,
            model,
            power_normalizer_mw=power_normalizer_mw,
            demand_normalizer_mw=demand_normalizer_mw,
            learning_rate=args.phase2_learning_rate,
            status_loss_weight=args.phase2_status_loss_weight,
            power_loss_weight=args.phase2_power_loss_weight,
        )
    history_phase2 = model.fit(
        x=train_inputs,
        y=train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        callbacks=make_callbacks(
            tf,
            args.output_dir,
            "phase2",
            "val_loss",
            args.phase2_patience,
            args,
            reduce_lr=True,
        ),
        verbose=args.verbose,
    )
    return history_phase1.history, history_phase2.history


def calculate_average_cost(status, power_mw, specs):
    operation_cost = (
        power_mw * specs["linear_cost"] + status * specs["noload_cost"]
    ).sum(axis=(1, 2))
    initial_status = specs["init_status"][np.newaxis, :]
    previous_status = np.concatenate(
        (np.repeat(initial_status[:, np.newaxis, :], len(status), axis=0), status[:, :-1]),
        axis=1,
    )
    startups = np.maximum(status - previous_status, 0.0)
    startup_cost = (startups * specs["startup_cost"]).sum(axis=(1, 2))
    return float(np.mean(operation_cost + startup_cost))


def calculate_temporal_violations(status, specs):
    mut_violation_hours = 0.0
    mdt_violation_hours = 0.0
    for sample_status in status:
        for generator, generator_status in enumerate(sample_status.T):
            for hour in range(1, len(generator_status)):
                if generator_status[hour - 1] == 0 and generator_status[hour] == 1:
                    check_len = min(specs["mut"][generator], len(generator_status) - hour)
                    actual_on = np.sum(generator_status[hour : hour + check_len])
                    mut_violation_hours += check_len - actual_on
                elif generator_status[hour - 1] == 1 and generator_status[hour] == 0:
                    check_len = min(specs["mdt"][generator], len(generator_status) - hour)
                    mdt_violation_hours += np.sum(generator_status[hour : hour + check_len])
    return {
        "mut_violation_hours_per_sample": float(mut_violation_hours / len(status)),
        "mdt_violation_hours_per_sample": float(mdt_violation_hours / len(status)),
    }


def evaluate(model, test_inputs, demand, true_status, true_power, specs, verbose):
    pred_status, pred_power = model.predict(test_inputs, verbose=verbose)
    pred_status_bin = (pred_status > 0.5).astype(np.float32)
    mean_demand = float(np.mean(demand))

    pred_total_power = np.sum(pred_power, axis=-1)
    mismatch = float(np.mean(np.abs(pred_total_power - demand.squeeze(-1))))
    ghost = np.maximum(pred_power, 0.0) * (1.0 - pred_status_bin)
    capacity = (
        np.maximum(pred_power - specs["p_max"], 0.0)
        + np.maximum(specs["p_min"] - pred_power, 0.0)
    ) * pred_status_bin
    delta_power = pred_power[:, 1:, :] - pred_power[:, :-1, :]
    stay_on = pred_status_bin[:, 1:, :] * pred_status_bin[:, :-1, :]
    ramp = (
        np.maximum(delta_power - specs["ramp_up"], 0.0)
        + np.maximum(-delta_power - specs["ramp_down"], 0.0)
    ) * stay_on

    metrics = {
        "status_accuracy_percent": float(np.mean(pred_status_bin == true_status) * 100),
        "power_mae_mw": float(np.mean(np.abs(pred_power - true_power))),
        "mismatch_mae_mw": mismatch,
        "mismatch_percent_of_mean_demand": mismatch / mean_demand * 100,
        "ghost_power_mw": float(np.mean(np.sum(ghost, axis=-1))),
        "capacity_violation_mw": float(np.mean(np.sum(capacity, axis=-1))),
        "ramp_violation_mw": float(np.mean(np.sum(ramp, axis=-1))),
        "cplex_average_daily_cost": calculate_average_cost(true_status, true_power, specs),
        "ai_average_daily_cost": calculate_average_cost(pred_status_bin, pred_power, specs),
    }
    metrics.update(calculate_temporal_violations(pred_status_bin, specs))
    return metrics


def save_json(path, content):
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def main():
    args = parse_args()
    configure_environment(args.seed)

    import tensorflow as tf
    from legacy.rnncell_model import build_hybrid_uc_model

    set_deterministic_seed(tf, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs(args.specs)
    demand, status, power = load_dataset(args.data, args.limit_samples)
    validate_shapes(demand, status, power, specs)
    train_indices, val_indices, test_indices = split_indices(len(demand), args.seed)
    print(
        f"Dataset loaded: train={len(train_indices)}, val={len(val_indices)}, "
        f"test={len(test_indices)}",
        flush=True,
    )

    static_features = make_static_features(specs)
    train_inputs = make_model_inputs(demand[train_indices], specs, static_features)
    val_inputs = make_model_inputs(demand[val_indices], specs, static_features)
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
    model = build_hybrid_uc_model(
        specs,
        demand_normalizer_mw=demand_normalizer_mw,
        balance_loss_weight=args.phase1_balance_loss_weight,
        num_hours=demand.shape[1],
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
    model.save(args.output_dir / "saved_uc_RC_model.keras")
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
            "power_normalizer_mw": power_normalizer_mw,
            "demand_normalizer_mw": demand_normalizer_mw,
        },
    )

    if not args.skip_evaluation:
        test_inputs = make_model_inputs(demand[test_indices], specs, static_features)
        metrics = evaluate(
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


if __name__ == "__main__":
    main()
