#!/usr/bin/env python
"""Train and evaluate the strict UC RNN with asymmetric status BCE.

This keeps the 30714 multi-step ramp-position architecture and changes the
status imitation term so false-ON errors are weighted by generator commitment
cost. The deterministic allocation/repair layers remain unchanged.
"""

from rnncell_strict_allocation_multistep_ramp_position_model import (
    build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
        default_output_name="rnncell_strict_asym_bce",
        saved_model_name="saved_uc_RC_strict_asym_bce_model.keras",
        model_variant="strict_asym_bce_multistep_ramp_position",
    )


if __name__ == "__main__":
    main()
