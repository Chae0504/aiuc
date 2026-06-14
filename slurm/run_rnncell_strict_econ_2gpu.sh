#!/bin/bash
#SBATCH -J rnncell_econ_2gpu
#SBATCH -p g1
#SBATCH --nodelist=ego-g01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtx_4090:2
#SBATCH -o /home/yoonjoo_chae/AIUC/logs/slurm/rnncell_strict_econ_2gpu_%j.out

set -euo pipefail

export DISTRIBUTION_STRATEGY=mirrored
export EXPECTED_GPUS=2
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"

exec /home/yoonjoo_chae/AIUC/slurm/run_rnncell_strict_econ_worker.sh
