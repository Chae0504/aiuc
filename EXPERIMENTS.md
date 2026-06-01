# Experiments

Keep one row per completed experiment. Compare cost only when the power-balance
mismatch is sufficiently small.

The CPLEX reference average daily cost is 982,572.44.

| Date | Job ID | Git Commit | Description | Status Acc. | Power MAE | Mismatch MAE | Mismatch | AI Daily Cost | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-31 | baseline | pre-git | Before dimensionless loss | 94.45% | 12.90 MW | 457.03 MW | 11.97% | 961,247.75 | Balance mismatch remains too large |
| 2026-05-31 | 22210 | b4beacb | Dimensionless loss; Phase 2 balance weight=2 | 93.73% | 13.65 MW | 415.92 MW | 10.90% | 961,678.12 | Early stopped at epoch 55; restored best epoch 15. Mismatch improved 9.00%, but remains too large |
| 2026-06-01 | 22223 | 3ac2243 | Long Phase 2 patience; balance weight=2 | 93.73% | 13.65 MW | 415.92 MW | 10.90% | 961,678.12 | Early stopped at epoch 95; restored best epoch 15. Longer training did not improve the selected model |
| 2026-06-01 | 22248 | 3ac2243 | Dimensionless loss; Phase 2 balance weight=5 | 92.61% | 13.66 MW | 256.55 MW | 6.72% | 963,483.44 | Early stopped at epoch 62; restored best epoch 22. Mismatch reduced 38.32% vs 22223, with status accuracy down 1.12 percentage points |
| 2026-06-01 | 22250 | 5bd0e9b | Dimensionless loss; Phase 2 balance weight=10 | 89.21% | 14.93 MW | 90.21 MW | 2.36% | 995,466.12 | Early stopped at epoch 76; restored best epoch 36. Mismatch reduced 64.84% vs 22248, while AI daily cost increased 3.32% |
| 2026-06-01 | 22271 | e1092ed | Dimensionless loss; Phase 2 balance weight=20 | 84.03% | 19.50 MW | 112.43 MW | 2.95% | 1,016,172.94 | Early stopped at epoch 68; restored best epoch 28. Worse than 22250 on the meaningful evaluation metrics; AI daily cost is 3.42% above CPLEX |

## Workflow

1. Commit code changes before submitting an experiment.
2. Submit with `sbatch run_rnncell.sh`.
3. Inspect `outputs/rnncell_<job_id>/evaluation.json`.
4. Generate a summary row with `python summarize_experiment.py outputs/rnncell_<job_id>`.
5. Add the row above with a short description and commit the updated log.
