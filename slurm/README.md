# Slurm Runners

Submit strict-data training jobs from the repository root with:

```bash
sbatch slurm/<script>.sh
```

Each script writes Slurm stdout/stderr to `logs/slurm/` and training artifacts
to `outputs/<experiment>_<job_id>/`.

