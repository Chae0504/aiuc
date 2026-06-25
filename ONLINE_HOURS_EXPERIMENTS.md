# Online-Hours Loss Experiments

These runs keep the `30714` multi-step ramp-position architecture and change
only the status imitation loss. No new physical layer is added.

The motivation came from the transition-loss replay: the best transition run
did not actually reduce transition event MAE. Its cost improvement came mostly
from fewer false-ON generator-hours and fewer predicted online-hours. This
branch therefore targets the online-hour surplus directly.

```text
H(u) = sum_t sum_i u_i,t

L_online =
  mean(|H(u_hat) - H(u)| / (T * G))

L_status = BCE(u, u_hat) + lambda_online * L_online
```

Here `T=24` and `G=54`. The loss is global at the scenario level: it pushes the
total number of online generator-hours toward the Gurobi label, but it does not
decide which specific generator should be turned off.

## Results

The strict Gurobi reference average daily cost is `939,381.69`. Job `30714` is
the physical-feasibility baseline with AI average daily cost `979,891.13`.

| Weight | Job | Status Acc. | Power MAE | Mismatch MAE | Cost Gap | AI Daily Cost | Delta vs 30714 | Best Phase 2 Epoch | Inference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `30714` | 78.10% | 12.92 MW | 0.000004 MW | +4.31% | 979,891.13 | 0.00 | 2 | N/A |
| 0.05 | `42765` | 77.00% | 13.03 MW | 0.000004 MW | +4.64% | 982,938.50 | +3,047.38 | 27 | 7.20 ms/sample |
| 0.10 | `58498` | 77.57% | 12.95 MW | 0.000004 MW | +4.46% | 981,247.88 | +1,356.75 | 26 | 7.25 ms/sample |
| 0.20 | `58816` | 78.33% | 12.74 MW | 0.000004 MW | +4.23% | 979,108.38 | -782.75 | 1 | 7.23 ms/sample |
| 0.30 | `59479` | 78.07% | 12.70 MW | 0.000004 MW | +4.33% | 980,033.75 | +142.63 | 2 | 7.35 ms/sample |
| 0.50 | `59629` | 77.90% | 12.82 MW | 0.000004 MW | +4.37% | 980,455.81 | +564.69 | 4 | 7.31 ms/sample |
| 1.00 | `59673` | 76.72% | 13.24 MW | 0.000004 MW | +4.72% | 983,765.00 | +3,873.88 | 1 | 7.13 ms/sample |

All completed online-hours runs preserve the solved physical behavior:

- mismatch max: `0.0005 MW`
- mismatch over 10 MW: `0.00%`
- shortage/excess MAE: about `0.000002 / 0.000002 MW`
- ghost/capacity/ramp/startup-cap/shutdown-cap violations: zero
- MUT/MDT violations: zero

Job `58817` attempted the same branch on the `g2` partition, but failed before
training because TensorFlow found `0` visible GPUs despite the Slurm GPU
allocation. It is excluded from the result table.

## Replay Diagnostics

Full diagnostics are in
[`LEARNING_OBJECTIVE_REPLAY_METRICS.csv`](LEARNING_OBJECTIVE_REPLAY_METRICS.csv).

| Weight | Job | Cost Delta vs 30714 | False ON / day | Pred. online-hours / day | Online-hour Delta / day | Startup count / day | Transition Event MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `30714` | 0.00 | 275.94 | 699.31 | +268.06 | 10.48 | 0.02649 |
| 0.05 | `42765` | +3,047.38 | 291.05 | 715.31 | +284.06 | 11.06 | 0.02673 |
| 0.10 | `58498` | +1,356.75 | 283.67 | 707.96 | +276.71 | 10.67 | 0.02630 |
| 0.20 | `58816` | -782.75 | 273.27 | 697.00 | +265.75 | 10.59 | 0.02633 |
| 0.30 | `59479` | +142.63 | 275.07 | 697.12 | +265.87 | 10.39 | 0.02636 |
| 0.50 | `59629` | +564.69 | 278.45 | 701.69 | +270.44 | 10.54 | 0.02628 |
| 1.00 | `59673` | +3,873.88 | 296.06 | 721.64 | +290.39 | 11.99 | 0.02660 |

## Interpretation

Weight `0.20` / job `58816` is the best online-hours result and the best
learning-objective branch so far:

- cost improves by `782.75` per day versus `30714`;
- cost improves by `356.00` per day versus the best transition run `42061`;
- power MAE improves from `12.92 MW` to `12.74 MW`;
- status accuracy improves from `78.10%` to `78.33%`;
- physical feasibility remains unchanged at numerical tolerance.

This supports the hypothesis from the transition replay: the useful economic
signal is not startup/shutdown timing by itself. It is reducing unnecessary
online generator-hours while staying feasible.

The result is not monotonic. Weights `0.05` and `0.10` are too weak or land in
a bad commitment structure. Weights `0.50` and `1.00` over-pressure the global
online-hour term and worsen cost/status quality. The useful region is narrow,
with `0.20` clearly best in this sweep and `0.30` still close but slightly
worse than the baseline cost.

## Problems Found

The largest remaining modeling problem is that the online-hour surplus is still
huge. Even the best run predicts `697.00` online generator-hours per day while
the Gurobi label has `431.25`, leaving a surplus of `265.75` generator-hours
per day.

The second problem is that this loss is global. It can reduce the total number
of online-hours, but it does not know whether the extra online unit is cheap,
expensive, ramp-critical, or needed later. This explains why the cost response
is scale-sensitive and non-monotonic.

The third problem is that Phase 2 often restores an early epoch. For the best
run, `58816`, the best Phase 2 epoch is `1`. That does not mean the whole model
failed to learn: Phase 1 already learned the main status structure, and the
deterministic layer enforces feasibility. But it does mean Phase 2 is not yet
providing a long, stable optimization trajectory for this objective.

## Next Fixes

The next branch should keep `30714` as the physical baseline and keep `58816`
as the current learning-objective best. The most useful next tests are:

1. Sweep locally around the useful region: `0.15`, `0.20`, `0.25`.
2. Replace the global online-hour loss with generator-wise online-hour loss:

   ```text
   L_gen_online =
     mean_i(|sum_t u_hat_i,t - sum_t u_i,t| / T)
   ```

   This keeps the direct online-hour idea but forces the model to match which
   units stay online.

3. Add a cost-weighted generator-wise online-hour loss:

   ```text
   w_i = normalized(no_load_i + Pmin_i * linear_cost_i)

   L_cost_online =
     mean_i(w_i * |sum_t u_hat_i,t - sum_t u_i,t| / T)
   ```

   This is safer than pointwise asymmetric BCE because it penalizes excess
   online duration over the whole day rather than every local false-ON bit.

4. Only after that, try a small combined objective with transition loss:

   ```text
   L_status =
     BCE + lambda_online * L_online + lambda_transition * L_transition
   ```

   Start conservatively because both losses are scale-sensitive.
