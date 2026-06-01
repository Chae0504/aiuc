# Experiments

Keep one row per completed experiment. Compare cost only when the power-balance
mismatch is sufficiently small.

| Date | Job ID | Git Commit | Description | Status Acc. | Power MAE | Mismatch MAE | Mismatch | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-05-31 | baseline | pre-git | Before dimensionless loss | 94.45% | 12.90 MW | 457.03 MW | 11.97% | Balance mismatch remains too large |
| 2026-05-31 | 22210 | b4beacb | Dimensionless loss; Phase 2 balance weight=2 | 93.73% | 13.65 MW | 415.92 MW | 10.90% | Early stopped at epoch 55; restored best epoch 15. Mismatch improved 9.00%, but remains too large |
| 2026-06-01 | 22223 | 3ac2243 | Long Phase 2 patience; balance weight=2 | 93.73% | 13.65 MW | 415.92 MW | 10.90% | Early stopped at epoch 95; restored best epoch 15. Longer training did not improve the selected model |
| 2026-06-01 | 22248 | 3ac2243 | Dimensionless loss; Phase 2 balance weight=5 | 92.61% | 13.66 MW | 256.55 MW | 6.72% | Early stopped at epoch 62; restored best epoch 22. Mismatch reduced 38.32% vs 22223, with status accuracy down 1.12 percentage points |

## Workflow

1. Commit code changes before submitting an experiment.
2. Submit with `sbatch run_rnncell.sh`.
3. Inspect `outputs/rnncell_<job_id>/evaluation.json`.
4. Generate a summary row with `python summarize_experiment.py outputs/rnncell_<job_id>`.
5. Add the row above with a short description and commit the updated log.
