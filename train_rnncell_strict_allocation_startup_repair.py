#!/usr/bin/env python
"""Train and evaluate the look-ahead startup-repair cost-aware UC RNN."""

from rnncell_strict_allocation_startup_repair_model import (
    build_hybrid_uc_strict_allocation_startup_repair_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_startup_repair_model,
        default_output_name="rnncell_strict_allocation_startup_repair",
        saved_model_name="saved_uc_RC_strict_allocation_startup_repair_model.keras",
        model_variant=(
            "strict_ramp_aware_cost_aware_allocation_with_lookahead_startup_repair"
        ),
    )


if __name__ == "__main__":
    main()
