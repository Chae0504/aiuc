#!/usr/bin/env python
"""Train and evaluate the strict UC RNN with transition-aware status loss.

This keeps the 30714 multi-step ramp-position architecture and changes only the
status imitation objective to include startup/shutdown timing loss.
"""

from rnncell_strict_allocation_multistep_ramp_position_model import (
    build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_multistep_ramp_position_model,
        default_output_name="rnncell_strict_transition",
        saved_model_name="saved_uc_RC_strict_transition_model.keras",
        model_variant="strict_transition_multistep_ramp_position",
    )


if __name__ == "__main__":
    main()
