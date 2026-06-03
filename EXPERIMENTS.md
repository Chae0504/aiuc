# Experiments

Keep one row per completed experiment. Compare cost only when the power-balance
mismatch is sufficiently small.

## Strict Dataset

The strict Gurobi reference average daily cost is 939,381.69.

| Date | Job ID | Git Commit | Description | Status Acc. | Power MAE | Mismatch MAE | Mismatch | AI Daily Cost | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-02 | 22368 | 87df85e | Strict ramp-aware proportional allocation; Phase 2 balance weight=5 | 82.81% | 19.90 MW | 59.18 MW | 1.63% | 991,582.44 | Early stopped at epoch 45; restored best epoch 5. All evaluated hard-constraint violations are zero. AI daily cost is 5.56% above strict Gurobi |
| 2026-06-02 | 22376 | 87df85e | Strict clipping baseline without allocation; Phase 2 balance weight=5 | 74.47% | 31.63 MW | 459.56 MW | 12.67% | 1,092,881.75 | Early stopped at epoch 52; restored best epoch 12. All evaluated hard-constraint violations are zero, but mismatch increased 676.55% vs 22368 and AI daily cost is 16.34% above strict Gurobi |
| 2026-06-02 | 22377 | 1dcbce8 | Strict clipping baseline without allocation; Phase 2 balance weight=10 | 74.50% | 31.52 MW | 432.77 MW | 11.93% | 1,084,168.38 | Early stopped at epoch 64; restored best epoch 24. All evaluated hard-constraint violations are zero. Mismatch improved 5.83% vs 22376, but remains 631.28% above strict allocation and AI daily cost is 15.41% above strict Gurobi |
| 2026-06-03 | 22441 | ba3573f + local changes | Strict allocation with vectorized MUT/MDT-aware commitment repair; Phase 2 balance weight=5 | 75.89% | 22.45 MW | 2.04 MW | 0.06% | 1,048,823.00 | Early stopped at epoch 41; restored best epoch 1. All evaluated hard-constraint violations are zero. AI daily cost is 11.65% above strict Gurobi. Mean mismatch is small, but inspect the tail-risk diagnostic below |
| 2026-06-03 | 22453 | ba3573f + local changes | Strict allocation with look-ahead commitment repair; Phase 2 balance weight=5 | 71.99% | 18.70 MW | 2.50 MW | 0.07% | 1,020,904.44 | Early stopped at epoch 47; restored best epoch 7. All evaluated hard-constraint violations are zero. The future-aware shutdown check nearly eliminates shortage, but excess generation dominates the remaining mismatch and AI daily cost is 8.68% above strict Gurobi |
| 2026-06-03 | 22454 | ba3573f + local changes | Strict look-ahead repair with cost-aware allocation; Phase 2 balance weight=5 | 77.54% | 12.92 MW | 0.02 MW | 0.00% | 980,708.12 | Early stopped at epoch 59; restored best epoch 19. All evaluated hard-constraint violations are zero. Cost-aware dispatch removes almost all excess generation, reduces AI daily cost to 4.40% above strict Gurobi, and is the strongest strict-data candidate so far. Rare tail shortages remain: max mismatch 378.04 MW, 0.018% of test hours above 10 MW |

### 22441 Tail-Risk Diagnostic

The row above retains the metrics saved at training completion. A replay of the
saved model on the same test split produced slightly different mean values
(`2.01 MW` mismatch and `1,048,893.62` cost), and exposed the following tail
risk:

| Metric | Value |
| --- | ---: |
| Maximum absolute mismatch | 859.03 MW |
| P95 absolute mismatch | 0.00049 MW |
| P99 absolute mismatch | 71.72 MW |
| Test hours with absolute mismatch above 10 MW | 1.68% |
| Mean shortage contribution | 0.73 MW |
| Mean excess contribution | 1.28 MW |

The residual is concentrated rather than broadly distributed. Excess generation
is concentrated around hour 4 because online units cannot yet shut down under
the `SDcap` rule. Shortage is concentrated around hours 9-11 because previously
stopped units have not yet completed `MDT`. The hourly repair layer cannot
anticipate that later demand increase. Job `22441` was submitted from local
uncommitted changes based on `ba3573f`, so preserve the worktree before treating
it as a reproducible baseline.

## Legacy Dataset

The legacy CPLEX reference average daily cost is 982,572.44. These experiments
used the earlier dataset and clipping equations, so retain them only for
historical comparison.

| Date | Job ID | Git Commit | Description | Status Acc. | Power MAE | Mismatch MAE | Mismatch | AI Daily Cost | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-05-31 | baseline | pre-git | Before dimensionless loss | 94.45% | 12.90 MW | 457.03 MW | 11.97% | 961,247.75 | Balance mismatch remains too large |
| 2026-05-31 | 22210 | b4beacb | Dimensionless loss; Phase 2 balance weight=2 | 93.73% | 13.65 MW | 415.92 MW | 10.90% | 961,678.12 | Early stopped at epoch 55; restored best epoch 15. Mismatch improved 9.00%, but remains too large |
| 2026-06-01 | 22223 | 3ac2243 | Long Phase 2 patience; balance weight=2 | 93.73% | 13.65 MW | 415.92 MW | 10.90% | 961,678.12 | Early stopped at epoch 95; restored best epoch 15. Longer training did not improve the selected model |
| 2026-06-01 | 22248 | 3ac2243 | Dimensionless loss; Phase 2 balance weight=5 | 92.61% | 13.66 MW | 256.55 MW | 6.72% | 963,483.44 | Early stopped at epoch 62; restored best epoch 22. Mismatch reduced 38.32% vs 22223, with status accuracy down 1.12 percentage points |
| 2026-06-01 | 22250 | 5bd0e9b | Dimensionless loss; Phase 2 balance weight=10 | 89.21% | 14.93 MW | 90.21 MW | 2.36% | 995,466.12 | Early stopped at epoch 76; restored best epoch 36. Mismatch reduced 64.84% vs 22248, while AI daily cost increased 3.32% |
| 2026-06-01 | 22271 | e1092ed | Dimensionless loss; Phase 2 balance weight=20 | 84.03% | 19.50 MW | 112.43 MW | 2.95% | 1,016,172.94 | Early stopped at epoch 68; restored best epoch 28. Worse than 22250 on the meaningful evaluation metrics; AI daily cost is 3.42% above CPLEX |
| 2026-06-01 | 22272 | 8b349d3 | Ramp-aware proportional allocation; Phase 2 balance weight=5 | 93.28% | 15.40 MW | 43.17 MW | 1.13% | 987,577.75 | Early stopped at epoch 41; restored best epoch 1. Mismatch reduced 52.14% vs 22250, while AI daily cost is only 0.51% above CPLEX |

## Workflow

1. Commit code changes before submitting an experiment.
2. Submit a strict experiment with
   `sbatch baselines/strict_clipping/run_rnncell_strict.sh` or
   `sbatch run_rnncell_strict_allocation.sh` or
   `sbatch run_rnncell_strict_allocation_repair.sh` or
   `sbatch run_rnncell_strict_allocation_lookahead_repair.sh` or
   `sbatch run_rnncell_strict_allocation_cost_aware.sh`.
3. Inspect `outputs/rnncell_strict_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_repair_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_lookahead_repair_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_cost_aware_<job_id>/evaluation.json`.
4. Generate a summary row with `python summarize_experiment.py <output_dir>`.
5. Add the row above with a short description and commit the updated log.
