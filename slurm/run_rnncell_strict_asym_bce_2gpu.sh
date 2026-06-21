#!/bin/bash
#SBATCH -J rnncell_asym_bce
#SBATCH -p g1
#SBATCH --nodelist=ego-g01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_4090:2
#SBATCH -o /home/yoonjoo_chae/AIUC/logs/slurm/rnncell_strict_asym_bce_2gpu_%j.out

set -euo pipefail

module purge
module load cuda/12.8.1
module load cudnn/cuda12/9.8.0.87

source /home/yoonjoo_chae/myenv/bin/activate
cd /home/yoonjoo_chae

export PYTHONUNBUFFERED=1

FALSE_ON_ALPHA="${FALSE_ON_ALPHA:-0.5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="AIUC/outputs/rnncell_strict_asym_bce_alpha${FALSE_ON_ALPHA}_2gpu_${JOB_ID}"
DATA_PATH="AIUC/DG/uc_new_data_strict.npz"
SPECS_PATH="AIUC/DG/generator_specs.csv"

python - "$FALSE_ON_ALPHA" "$GLOBAL_BATCH_SIZE" <<'PY'
import math
import sys

alpha = float(sys.argv[1])
global_batch_size = int(sys.argv[2])
expected_gpus = 2
if not math.isfinite(alpha) or alpha < 0:
    raise ValueError("FALSE_ON_ALPHA must be a finite non-negative number")
if global_batch_size < expected_gpus or global_batch_size % expected_gpus:
    raise ValueError("GLOBAL_BATCH_SIZE must be divisible by 2")
print(f"False-ON alpha: {alpha:g}")
print(
    f"GPU count: 2; global batch size: {global_batch_size}; "
    f"per-replica batch size: {global_batch_size // 2}"
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

python -u AIUC/train_rnncell_strict_asym_bce.py \
  --data "$DATA_PATH" \
  --specs "$SPECS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$GLOBAL_BATCH_SIZE" \
  --distribution-strategy mirrored \
  --expected-gpus 2 \
  --status-loss-mode cost_weighted_bce \
  --status-false-on-alpha "$FALSE_ON_ALPHA" \
  --phase2-epochs 150 \
  --phase2-patience 40 \
  --phase2-learning-rate 3e-5 \
  --reduce-lr-patience 8 \
  --reduce-lr-factor 0.5 \
  --phase2-status-loss-weight 1 \
  --phase2-power-loss-weight 1 \
  --phase2-balance-loss-weight 5 \
  --phase2-cost-loss-weight 0 \
  --verbose 2
