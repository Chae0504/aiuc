#!/bin/bash
#SBATCH -J rnncell_alloc
#SBATCH -p g1
#SBATCH --nodelist=ego-g01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -o rnncell_allocation_%j.out

set -euo pipefail

module purge
module load cuda/12.8.1
module load cudnn/cuda12/9.8.0.87

source /home/yoonjoo_chae/myenv/bin/activate
cd /home/yoonjoo_chae

export PYTHONUNBUFFERED=1

OUTPUT_DIR="AIUC/outputs/rnncell_allocation_${SLURM_JOB_ID}"

python -u AIUC/train_rnncell_allocation.py \
  --output-dir "$OUTPUT_DIR" \
  --phase2-epochs 150 \
  --phase2-patience 40 \
  --reduce-lr-patience 8 \
  --reduce-lr-factor 0.5 \
  --phase2-balance-loss-weight 5 \
  --verbose 2
