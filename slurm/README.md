# Slurm Runners

Submit strict-data training jobs from the repository root with:

```bash
sbatch slurm/<script>.sh
```

Each script writes Slurm stdout/stderr to `logs/slurm/` and training artifacts
to `outputs/<experiment>_<job_id>/`.

For controlled economic-loss runs, vary only the cost proxy scale:

```bash
COST_PROXY_WEIGHT=5 sbatch slurm/run_rnncell_strict_econ.sh
```

Use two GPUs on one node with TensorFlow `MirroredStrategy`:

```bash
COST_PROXY_WEIGHT=5 sbatch slurm/run_rnncell_strict_econ_2gpu.sh
```

Both runners default to global batch `64`, so the 2-GPU result remains directly
comparable with the 1-GPU experiment. This gives batch `32` per replica.
For a throughput-oriented run, use `GLOBAL_BATCH_SIZE=128`, but treat it as a
separate hyperparameter experiment because it changes optimization behavior.

For the 30714 architecture with cost-weighted false-ON BCE:

```bash
FALSE_ON_ALPHA=0.5 sbatch slurm/run_rnncell_strict_asym_bce_2gpu.sh
```

This branch keeps balance weight `5` and cost-proxy weight `0`; only the status
BCE is changed. Larger `FALSE_ON_ALPHA` penalizes expensive false-ON commitment
errors more strongly.

For the 30714 architecture with startup/shutdown transition imitation:

```bash
TRANSITION_WEIGHT=0.5 sbatch slurm/run_rnncell_strict_transition_2gpu.sh
```

This branch keeps the physical layers unchanged and changes only the status
loss to `BCE + TRANSITION_WEIGHT * transition_MAE`.

For the 30714 architecture with total online-hour imitation:

```bash
ONLINE_HOURS_WEIGHT=0.1 sbatch slurm/run_rnncell_strict_online_hours_2gpu.sh
```

This branch keeps the physical layers unchanged and changes only the status
loss to `BCE + ONLINE_HOURS_WEIGHT * normalized_online_hour_error`.
