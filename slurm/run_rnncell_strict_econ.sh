#!/bin/bash
#SBATCH -J rnncell_strict_econ
#SBATCH -p g1
#SBATCH --nodelist=ego-g01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -o /home/yoonjoo_chae/AIUC/logs/slurm/rnncell_strict_econ_%j.out

set -euo pipefail

export DISTRIBUTION_STRATEGY=none
export EXPECTED_GPUS=1
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"

exec /home/yoonjoo_chae/AIUC/slurm/run_rnncell_strict_econ_worker.sh
