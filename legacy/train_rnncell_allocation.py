#!/usr/bin/env python
"""Train and evaluate the ramp-aware proportional-allocation UC RNN."""

import json

import numpy as np

from legacy.train_rnncell import (
    SCRIPT_DIR,
    configure_environment,
    evaluate,
    get_git_commit,
    load_dataset,
    load_specs,
    make_model_inputs,
    make_static_features,
    parse_args,
    save_json,
    set_deterministic_seed,
    split_indices,
    train_two_phases,
    validate_shapes,
)


def main():
    args = parse_args()
    default_output_dir = SCRIPT_DIR / "outputs" / "rnncell"
    if args.output_dir == default_output_dir:
        args.output_dir = SCRIPT_DIR / "outputs" / "rnncell_allocation"
    configure_environment(args.seed)

    import tensorflow as tf
    from legacy.rnncell_allocation_model import build_hybrid_uc_allocation_model

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
    model = build_hybrid_uc_allocation_model(
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
    model.save(args.output_dir / "saved_uc_RC_allocation_model.keras")
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
            "model_variant": "ramp_aware_proportional_allocation",
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
