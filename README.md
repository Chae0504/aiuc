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

The training CSV files include `out_power_mismatch_mae_mw` and
`out_power_normalized_mismatch` so the power-balance trade-off can be monitored
at each epoch.

## Ramp-Aware Allocation Variant

The baseline model remains in `rnncell_model.py`. To train the separate variant
that proportionally allocates residual demand over online generators within
their ramp-aware headroom, submit:

```bash
sbatch run_rnncell_allocation.sh
```

Allocation experiments write artifacts to a separate directory:

```text
outputs/rnncell_allocation_<job_id>/
```

Generate their experiment-log row with:

```bash
python summarize_experiment.py outputs/rnncell_allocation_<job_id>
```

The allocation layer stays inside the recurrent cell so the adjusted generator
outputs are used as the previous-hour state for the next ramp calculation.
Residual demand that cannot be covered by the online generators remains in the
balance loss and can still guide the commitment decision.

## Experiment Log

Models, datasets, and raw logs remain local because they are large. Record the
compact result summary in [EXPERIMENTS.md](EXPERIMENTS.md) after each run.

Generate a Markdown row from an output directory:

```bash
python summarize_experiment.py outputs/rnncell_<job_id>
```
