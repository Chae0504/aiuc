#!/bin/bash

set -euo pipefail

module purge
module load cuda/12.8.1
module load cudnn/cuda12/9.8.0.87

source /home/yoonjoo_chae/myenv/bin/activate
cd /home/yoonjoo_chae

export PYTHONUNBUFFERED=1

COST_PROXY_WEIGHT="${COST_PROXY_WEIGHT:-5.0}"
DISTRIBUTION_STRATEGY="${DISTRIBUTION_STRATEGY:-none}"
EXPECTED_GPUS="${EXPECTED_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="AIUC/outputs/rnncell_strict_econ_costw${COST_PROXY_WEIGHT}_${EXPECTED_GPUS}gpu_${JOB_ID}"
DATA_PATH="AIUC/DG/uc_new_data_strict.npz"
SPECS_PATH="AIUC/DG/generator_specs.csv"

python - "$COST_PROXY_WEIGHT" "$EXPECTED_GPUS" "$GLOBAL_BATCH_SIZE" <<'PY'
import math
import sys

weight = float(sys.argv[1])
expected_gpus = int(sys.argv[2])
global_batch_size = int(sys.argv[3])
if not math.isfinite(weight) or weight < 0:
    raise ValueError("COST_PROXY_WEIGHT must be a finite non-negative number")
if expected_gpus < 1:
    raise ValueError("EXPECTED_GPUS must be positive")
if global_batch_size < expected_gpus or global_batch_size % expected_gpus:
    raise ValueError("GLOBAL_BATCH_SIZE must be divisible by EXPECTED_GPUS")
print(f"Phase 2 cost proxy weight: {weight:g}")
print(
    f"GPU count: {expected_gpus}; global batch size: {global_batch_size}; "
    f"per-replica batch size: {global_batch_size // expected_gpus}"
)
PY

python - "$DATA_PATH" <<'PY'
import sys

import numpy as np

path = sys.argv[1]
with np.load(path) as data:
    expected_shapes = {
        "X_demand": (50_000, 24, 1),
        "Y_status": (50_000, 24, 54),
        "Y_power": (50_000, 24, 54),
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = data[key].shape
        if actual_shape != expected_shape:
            raise ValueError(f"{key}: expected {expected_shape}, got {actual_shape}")
    if int(data["accepted_samples"][0]) != 50_000:
        raise ValueError("Strict UC dataset generation did not finish cleanly")
print(f"Strict UC dataset preflight passed: {path}")
PY

python -u AIUC/train_rnncell_strict_econ.py \
  --data "$DATA_PATH" \
  --specs "$SPECS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$GLOBAL_BATCH_SIZE" \
  --distribution-strategy "$DISTRIBUTION_STRATEGY" \
  --expected-gpus "$EXPECTED_GPUS" \
  --phase2-epochs 150 \
  --phase2-patience 40 \
  --phase2-learning-rate 3e-5 \
  --reduce-lr-patience 8 \
  --reduce-lr-factor 0.5 \
  --phase2-status-loss-weight 1 \
  --phase2-power-loss-weight 1 \
  --phase2-balance-loss-weight 0 \
  --phase2-cost-loss-weight "$COST_PROXY_WEIGHT" \
  --verbose 2
