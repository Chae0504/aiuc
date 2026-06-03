#!/usr/bin/env python
"""Train and evaluate cost-aware allocation with look-ahead safety margin."""

from rnncell_strict_allocation_cost_aware_model import (
    build_hybrid_uc_strict_allocation_cost_aware_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_cost_aware_model,
        default_output_name="rnncell_strict_allocation_cost_aware_margin",
        saved_model_name="saved_uc_RC_strict_allocation_cost_aware_margin_model.keras",
        model_variant=(
            "strict_ramp_aware_cost_aware_allocation_with_lookahead_safety_margin"
        ),
    )


if __name__ == "__main__":
    main()
