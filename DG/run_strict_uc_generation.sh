#!/bin/bash

set -euo pipefail

if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module load julia/1.11.3 gurobi/13.0.1
cd "$(dirname "$0")"

export NUM_SAMPLES="${NUM_SAMPLES:-50000}"
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-150000}"
export RANDOM_SEED="${RANDOM_SEED:-42}"
export MIP_GAP="${MIP_GAP:-0.001}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
export OUTPUT_PATH="${OUTPUT_PATH:-uc_new_data_strict.npz}"

exec julia --project=. generate_strict_uc_dataset.jl
