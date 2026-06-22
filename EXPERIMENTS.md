# Experiments

Keep one row per completed experiment. Compare cost only when the power-balance
mismatch is sufficiently small.

## Strict Dataset

The strict Gurobi reference average daily cost is 939,381.69.

For the causal experiment narrative from `22368` to `22496`, excluding the
no-allocation baselines `22376` and `22377`, see
[`STRICT_EXPERIMENT_FLOW.md`](STRICT_EXPERIMENT_FLOW.md).

For a compact map of the active model branches, see
[`STRICT_MODEL_SUMMARY.md`](STRICT_MODEL_SUMMARY.md).

The controlled commitment-cost proxy sweep is recorded in
[`COST_PROXY_EXPERIMENTS.md`](COST_PROXY_EXPERIMENTS.md).

The asymmetric false-ON BCE sweep is recorded in
[`ASYMMETRIC_BCE_EXPERIMENTS.md`](ASYMMETRIC_BCE_EXPERIMENTS.md).

The transition-loss sweep is recorded in
[`TRANSITION_LOSS_EXPERIMENTS.md`](TRANSITION_LOSS_EXPERIMENTS.md).

### Strict Baselines Without Allocation

| Date | Job ID | Git Commit | Description | Status Acc. | Power MAE | Mismatch MAE (% demand) | Cost Diff. | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-06-02 | 22376 | 87df85e | Strict clipping baseline; Phase 2 balance weight=5 | 74.47% | 31.63 MW | 459.56 MW (12.67%) | +16.34% | Early stopped at epoch 52; restored best epoch 12. All evaluated hard-constraint violations are zero |
| 2026-06-02 | 22377 | 1dcbce8 | Strict clipping baseline; Phase 2 balance weight=10 | 74.50% | 31.52 MW | 432.77 MW (11.93%) | +15.41% | Early stopped at epoch 64; restored best epoch 24. Mismatch improved slightly vs 22376, but remains too large |

### Strict Allocation Variants

Safety-margin runs from `22454` are recorded separately in
`MARGIN_EXPERIMENTS.md`.

| Date | Job ID | Git Commit | Description | Status Acc. | Power MAE | Mismatch MAE (% demand) | Cost Diff. | Mismatch Max | Mismatch >10MW | Shortage / Excess MAE | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-02 | 22368 | 87df85e | Proportional allocation; Phase 2 balance weight=5 | 82.81% | 19.90 MW | 59.18 MW (1.63%) | +5.56% | N/A | N/A | N/A | Early stopped at epoch 45; restored best epoch 5. Tail metrics were not recorded by the old evaluator |
| 2026-06-03 | 22441 | ba3573f + local changes | MUT/MDT-aware commitment repair; Phase 2 balance weight=5 | 75.89% | 22.45 MW | 2.04 MW (0.06%) | +11.65% | 859.03 MW | 1.68% | 0.73 / 1.28 MW | Early stopped at epoch 41; restored best epoch 1. Tail metrics are from a saved-model replay; official mean mismatch is from training-completion evaluation |
| 2026-06-03 | 22453 | ba3573f + local changes | Look-ahead commitment repair; Phase 2 balance weight=5 | 71.99% | 18.70 MW | 2.50 MW (0.07%) | +8.68% | 376.09 MW | 2.32% | 0.008 / 2.49 MW | Early stopped at epoch 47; restored best epoch 7. Future-aware shutdown check nearly eliminates shortage, but excess dominates the residual |
| 2026-06-03 | 22454 | ba3573f + local changes | Look-ahead repair with cost-aware allocation; Phase 2 balance weight=5 | 77.54% | 12.92 MW | 0.016 MW (0.0004%) | +4.40% | 378.04 MW | 0.018% | 0.016 / 0.000002 MW | Early stopped at epoch 59; restored best epoch 19. Strongest strict-data candidate so far; rare tail shortages remain |
| 2026-06-03 | 22496 | 052b0df | Look-ahead startup repair with cost-aware allocation; Phase 2 balance weight=5 | 76.79% | 12.86 MW | 0.015 MW (0.0004%) | +4.60% | 269.35 MW | 0.019% | 0.015 / 0.000002 MW | Early stopped at epoch 42; restored best epoch 2. Reduces worst-case shortage vs 22454, but cost and over-10MW frequency are slightly worse |
| 2026-06-10 | 28562 | 3e51f4d | Ramp-position-aware allocation with startup repair and cost-aware dispatch; Phase 2 balance weight=5 | 77.87% | 12.81 MW | 0.001 MW (0.00003%) | +4.25% | 117.57 MW | 0.00083% | 0.00098 / 0.000002 MW | Completed Phase 2 epoch 150; restored best epoch 148. Strongest strict-data result so far: improves average mismatch, tail shortage, power MAE, and cost vs 22496 |
| 2026-06-11 | 30714 | 8bd3ee4 | Multi-step ramp-position-aware allocation with startup repair and cost-aware dispatch; Phase 2 balance weight=5 | 78.10% | 12.92 MW | 0.000004 MW (0.0000001%) | +4.31% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 42; restored best epoch 2. Generator-side hard constraints and balance are satisfied to numerical tolerance; cost is slightly worse than 28562 |
| 2026-06-12 | 33220 | 7578feb | `30714` architecture with Phase 2 commitment cost proxy; status/power/balance/cost weights=`0.5/1/0/0.05` | 77.79% | 12.88 MW | 0.000004 MW (0.0000001%) | +4.39% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 43; restored best epoch 3. Physical feasibility is preserved, but cost worsens by 758.25/day vs 30714. Replay shows +4.38 online generator-hours/day and +0.252 startups/day; inference is 6.61 ms/sample on 5,000 samples |
| 2026-06-14 | 35385 | b798c7f | Controlled cost-proxy sweep; weight=1, BCE/power/balance=`1/1/0`; 2 GPUs, global batch=128 | 77.84% | 13.04 MW | 0.000004 MW (0.0000001%) | +4.36% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 45; restored best epoch 5. Proxy fell 1.36%, but daily cost remains 478.06 above 30714 |
| 2026-06-15 | 35926 | b798c7f | Controlled cost-proxy sweep; weight=5, BCE/power/balance=`1/1/0`; 2 GPUs, global batch=128 | 77.99% | 12.95 MW | 0.000004 MW (0.0000001%) | +4.32% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 45; restored best epoch 5. Best sweep result; daily cost is only 45.25 above 30714 |
| 2026-06-15 | 36329 | b798c7f | Controlled cost-proxy sweep; weight=10, BCE/power/balance=`1/1/0`; 2 GPUs, global batch=128 | 77.89% | 12.96 MW | 0.000004 MW (0.0000001%) | +4.33% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 45; restored best epoch 5. Proxy falls slightly further, but higher linear production cost makes total cost worse than weight 5 |
| 2026-06-18 | 39104 | d66c424 | `30714` architecture with cost-weighted false-ON BCE; alpha=0.5; 2 GPUs, global batch=64 | 77.90% | 12.61 MW | 0.000004 MW (0.0000001%) | +4.38% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 41; restored best epoch 1. Best asymmetric-BCE run, but daily cost is 590.31 above 30714 |
| 2026-06-19 | 40234 | d66c424 | `30714` architecture with cost-weighted false-ON BCE; alpha=1.0; 2 GPUs, global batch=64 | 76.45% | 12.33 MW | 0.000004 MW (0.0000001%) | +4.79% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 42; restored best epoch 2. Lowest power MAE in this sweep, but cost worsens by 4,474.06/day vs 30714 |
| 2026-06-21 | 41987 | d66c424 | `30714` architecture with cost-weighted false-ON BCE; alpha=1.5; 2 GPUs, global batch=64 | 77.32% | 12.63 MW | 0.000004 MW (0.0000001%) | +4.59% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 43; restored best epoch 3. Larger alpha does not recover cost; feasibility remains numerically perfect |
| 2026-06-21 | 42010 | 435bcda + local changes | `30714` architecture with transition BCE; transition weight=0.5; 2 GPUs, global batch=64 | 77.84% | 12.86 MW | 0.000004 MW (0.0000001%) | +4.38% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 41; restored best epoch 1. Feasibility preserved, but cost worsens by 610.38/day vs 30714 |
| 2026-06-21 | 42043 | 435bcda + local changes | `30714` architecture with transition BCE; transition weight=1.0; 2 GPUs, global batch=64 | 78.21% | 12.83 MW | 0.000004 MW (0.0000001%) | +4.30% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 50; restored best epoch 10. Best moderate-weight run; improves cost by 148.63/day vs 30714 |
| 2026-06-22 | 42059 | 435bcda + local changes | `30714` architecture with transition BCE; transition weight=1.5; 2 GPUs, global batch=64 | 78.05% | 12.86 MW | 0.000004 MW (0.0000001%) | +4.35% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 48; restored best epoch 8. Cost worsens by 397.69/day vs 30714 |
| 2026-06-22 | 42060 | 435bcda + local changes | `30714` architecture with transition BCE; transition weight=2.0; 2 GPUs, global batch=64 | 77.12% | 12.59 MW | 0.000004 MW (0.0000001%) | +4.75% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 41; restored best epoch 1. Power MAE improves, but status structure and cost degrade sharply |
| 2026-06-22 | 42061 | 435bcda + local changes | `30714` architecture with transition BCE; transition weight=5.0; 2 GPUs, global batch=64 | 78.28% | 12.38 MW | 0.000004 MW (0.0000001%) | +4.27% | 0.0005 MW | 0.00% | 0.000002 / 0.000002 MW | Early stopped at epoch 81; restored best epoch 41. Best transition run: cost improves by 426.75/day and power MAE improves by 0.54 MW vs 30714 |

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
   `sbatch slurm/run_rnncell_strict_allocation.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_repair.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_lookahead_repair.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_cost_aware.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_cost_aware_margin.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_startup_repair.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_ramp_position.sh` or
   `sbatch slurm/run_rnncell_strict_allocation_multistep_ramp_position.sh` or
   `sbatch slurm/run_rnncell_strict_econ.sh`.
3. Inspect `outputs/rnncell_strict_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_repair_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_lookahead_repair_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_cost_aware_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_cost_aware_margin_<margin>_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_startup_repair_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_ramp_position_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_allocation_multistep_ramp_position_<job_id>/evaluation.json` or
   `outputs/rnncell_strict_econ_costw<weight>_<gpu_count>gpu_<job_id>/evaluation.json`.
4. Generate a summary row with `python summarize_experiment.py <output_dir>`.
5. Add the row above with a short description and commit the updated log.
