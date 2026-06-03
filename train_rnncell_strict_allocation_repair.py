#!/usr/bin/env python
"""Train and evaluate the MUT/MDT-aware commitment-repair allocation UC RNN."""

from rnncell_strict_allocation_repair_model import (
    build_hybrid_uc_strict_allocation_repair_model,
)
from train_rnncell_strict_allocation import train_and_evaluate


def main():
    train_and_evaluate(
        build_hybrid_uc_strict_allocation_repair_model,
        default_output_name="rnncell_strict_allocation_repair",
        saved_model_name="saved_uc_RC_strict_allocation_repair_model.keras",
        model_variant="strict_ramp_aware_proportional_allocation_with_mut_mdt_repair",
    )


if __name__ == "__main__":
    main()
