# Asymmetric BCE Experiments

These runs keep the `30714` multi-step ramp-position architecture and change
only the status imitation loss. The status loss is a cost-weighted BCE where
false-ON errors are penalized by generator commitment cost:

```text
w_i = 1 + alpha * (C_i^NL + C_i^L P_i^min) / max_j(C_j^NL + C_j^L P_j^min)
L_status = mean BCE(u_i,t, u_hat_i,t) * [u_i,t + (1 - u_i,t) w_i]
```

So the method does not penalize all online decisions. It penalizes predicting
an expensive unit ON when the Gurobi label says it should be OFF.

## Results

The strict Gurobi reference average daily cost is `939,381.69`. Job `30714` is
the physical-feasibility baseline with AI average daily cost `979,891.13`.

| Alpha | Job | Status Acc. | Power MAE | Mismatch MAE | Cost Gap | AI Daily Cost | Delta vs 30714 | Best Phase 2 Epoch | Inference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `30714` | 78.10% | 12.92 MW | 0.000004 MW | +4.31% | 979,891.13 | 0.00 | 2 | N/A |
| 0.5 | `39104` | 77.90% | 12.61 MW | 0.000004 MW | +4.38% | 980,481.44 | +590.31 | 1 | 7.21 ms/sample |
| 1.0 | `40234` | 76.45% | 12.33 MW | 0.000004 MW | +4.79% | 984,365.19 | +4,474.06 | 2 | 7.24 ms/sample |
| 1.5 | `41987` | 77.32% | 12.63 MW | 0.000004 MW | +4.59% | 982,516.31 | +2,625.19 | 3 | 7.27 ms/sample |

All three asymmetric BCE runs preserve the solved physical behavior:

- mismatch max: `0.0005 MW`
- mismatch over 10 MW: `0.00%`
- ghost/capacity/ramp/startup-cap/shutdown-cap violations: zero
- MUT/MDT violations: zero

## Replay Diagnostics

The saved models were replayed on the same test split to recover commitment
diagnostics that were not logged during the original runs. Full results are in
[`LEARNING_OBJECTIVE_REPLAY_METRICS.csv`](LEARNING_OBJECTIVE_REPLAY_METRICS.csv).

| Run | False ON / day | False OFF / day | Pred. online-hours / day | Startup count / day | Transition event MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `30714` baseline | 275.94 | 7.88 | 699.31 | 10.48 | 0.02649 |
| `39104` alpha 0.5 | 277.30 | 9.17 | 699.38 | 10.16 | 0.02657 |
| `40234` alpha 1.0 | 294.24 | 11.02 | 714.47 | 10.00 | 0.02649 |
| `41987` alpha 1.5 | 284.82 | 9.10 | 706.97 | 10.39 | 0.02653 |

## Interpretation

This was a useful negative result.

The method successfully leaves the deterministic feasibility layers intact, so
it does not reintroduce balance or hard-constraint violations. It also improves
power MAE versus `30714`, especially at alpha `1.0`.

But it does not improve the actual UC objective. The best asymmetric BCE case,
alpha `0.5`, is still `590.31` per day more expensive than `30714` and
`545.06` per day more expensive than the controlled cost-proxy weight-5 run
`35926`. Larger alpha values reduce status accuracy and worsen cost.

The replay shows the more direct failure mode. The asymmetric BCE did not
actually reduce false-ON behavior. Alpha `1.0`, the worst cost case, increased
false-ON events by about `18.30` generator-hours/day and increased predicted
online-hours by about `15.16` versus `30714`. Therefore the weighted OFF-label
BCE did not translate into fewer unnecessary commitments.

The likely reason is that false-ON weighting is too local and competes with the
rest of the trajectory objective. It can penalize an OFF-label unit, but it does
not know whether the replacement schedule requires more expensive production,
creates a worse ramp position, or forces future commitment changes. The final
UC cost depends on the joint commitment and dispatch trajectory, not only on
generator-wise false-ON labels.

## Conclusion

Do not use asymmetric BCE as the next baseline. Keep `30714` as the physical
baseline and `35926` as the best current economic-loss diagnostic. The next
economic method should include the linear production term or use a
feasibility-preserving cost-aware commitment/decommitment step, rather than
increasing false-ON BCE pressure.
