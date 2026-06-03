#!/bin/bash
#SBATCH -J lstm_ipynb
#SBATCH -p g1
#SBATCH --nodelist=ego-g01
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -o lstm_%j.out

module purge
module load cuda/12.8.1
module load cudnn/cuda12/9.8.0.87

source /home/yoonjoo_chae/myenv/bin/activate
cd /home/yoonjoo_chae

export PYTHONUNBUFFERED=1

python -u AIUC/ipynb/run_notebook_live.py \
  AIUC/ipynb/uc_118_LSTM.ipynb \
  --output AIUC/ipynb/uc_118_LSTM_executed.ipynb \
  --timeout -1
