#!/usr/bin/env python
"""Train and evaluate the multi-step ramp-position-aware strict UC RNN."""

from rnncell_strict_allocation_multistep_ramp_position_model import (
    build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
        default_output_name="rnncell_strict_allocation_multistep_ramp_position",
        saved_model_name=(
            "saved_uc_RC_strict_allocation_multistep_ramp_position_model.keras"
        ),
        model_variant=(
            "strict_ramp_aware_cost_aware_allocation_with_startup_repair_and_"
            "multistep_ramp_positioning"
        ),
    )


if __name__ == "__main__":
    main()
