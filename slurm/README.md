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
