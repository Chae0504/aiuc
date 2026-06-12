#!/usr/bin/env python
"""Train and evaluate the economic-loss strict UC RNN.

This keeps the 30714 multi-step ramp-position architecture, but changes the
Phase 2 objective toward a cost proxy instead of additional balance pressure.
"""

from rnncell_strict_allocation_multistep_ramp_position_model import (
    build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
        default_output_name="rnncell_strict_econ",
        saved_model_name="saved_uc_RC_strict_econ_model.keras",
        model_variant="strict_econ_multistep_ramp_position",
    )


if __name__ == "__main__":
    main()
