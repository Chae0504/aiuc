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
