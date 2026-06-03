# Look-Ahead Safety Margin Experiments

These runs branch from job `22454`, the strict look-ahead repair with
cost-aware allocation. The margin changes the look-ahead shutdown feasibility
test from:

```text
sum_i U_i,t+k >= D_t+k
```

to:

```text
sum_i U_i,t+k >= D_t+k + margin
```

The goal was to reduce rare shortage tails: `mismatch_max_mw`,
`mismatch_over_10mw_percent`, and `shortage_mae_mw`.

## Results

The strict Gurobi reference average daily cost is 939,381.69.

| Date | Job ID | Margin | Status Acc. | Power MAE | Mismatch MAE (% demand) | Cost Diff. | Mismatch Max | Mismatch >10MW | Shortage / Excess MAE | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-03 | 22454 | 0 MW | 77.54% | 12.92 MW | 0.016 MW (0.0004%) | +4.40% | 378.04 MW | 0.018% | 0.016 / 0.000002 MW | Baseline cost-aware allocation result |
| 2026-06-03 | 22455 | 10 MW | 77.54% | 12.92 MW | 0.016 MW (0.0004%) | +4.40% | 378.04 MW | 0.018% | 0.016 / 0.000002 MW | Same metrics as 22454 |
| 2026-06-03 | 22461 | 25 MW | 77.54% | 12.92 MW | 0.016 MW (0.0004%) | +4.40% | 378.04 MW | 0.018% | 0.016 / 0.000002 MW | Same metrics as 22454 |
| 2026-06-03 | 22468 | 50 MW | 77.54% | 12.92 MW | 0.016 MW (0.0005%) | +4.40% | 394.30 MW | 0.018% | 0.016 / 0.000036 MW | Slightly worse tail max and excess |

## Interpretation

The safety margin did not improve the rare shortage tail. Jobs `22455` and
`22461` exactly matched `22454`; job `22468` slightly worsened the maximum
mismatch. The max-mismatch replay showed that the tail event is not caused by
unsafe shutdown at the previous hour. It is caused by insufficient proactive
startup and ramp positioning before a very steep demand jump.

Therefore the next branch should keep `22454` as the baseline and add a
look-ahead startup repair rather than increasing shutdown reserve margin.
