#!/bin/bash
#SBATCH -J rnncell_strict_alloc_repair
#SBATCH -p g1
#SBATCH --nodelist=ego-g01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -o rnncell_strict_allocation_repair_%j.out

set -euo pipefail

module purge
module load cuda/12.8.1
module load cudnn/cuda12/9.8.0.87

source /home/yoonjoo_chae/myenv/bin/activate
cd /home/yoonjoo_chae

export PYTHONUNBUFFERED=1

OUTPUT_DIR="AIUC/outputs/rnncell_strict_allocation_repair_${SLURM_JOB_ID}"
DATA_PATH="AIUC/DG/uc_new_data_strict.npz"
SPECS_PATH="AIUC/DG/generator_specs.csv"

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

python -u AIUC/train_rnncell_strict_allocation_repair.py \
  --data "$DATA_PATH" \
  --specs "$SPECS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --phase2-epochs 150 \
  --phase2-patience 40 \
  --phase2-learning-rate 3e-5 \
  --reduce-lr-patience 8 \
  --reduce-lr-factor 0.5 \
  --phase2-balance-loss-weight 5 \
  --verbose 2
