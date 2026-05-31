# AIUC

Physics-informed recurrent neural network experiments for unit commitment.

## SLURM Training

Submit the modular RNN-cell training job:

```bash
sbatch run_rnncell.sh
```

Each SLURM job writes artifacts to its own directory:

```text
outputs/rnncell_<job_id>/
```

Important files:

```text
run_configuration.json  # hyperparameters, normalizers, and Git commit
phase1_training.csv     # epoch metrics for status-focused training
phase2_training.csv     # epoch metrics for hybrid fine-tuning
evaluation.json         # final test metrics
phase1_best.keras       # best Phase 1 checkpoint
phase2_best.keras       # best Phase 2 checkpoint
```

Monitor Phase 2 training:

```bash
tail -F outputs/rnncell_<job_id>/phase2_training.csv
```

## Experiment Log

Models, datasets, and raw logs remain local because they are large. Record the
compact result summary in [EXPERIMENTS.md](EXPERIMENTS.md) after each run.

Generate a Markdown row from an output directory:

```bash
python summarize_experiment.py outputs/rnncell_<job_id>
```
