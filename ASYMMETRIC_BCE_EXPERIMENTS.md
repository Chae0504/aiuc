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

## Interpretation

This was a useful negative result.

The method successfully leaves the deterministic feasibility layers intact, so
it does not reintroduce balance or hard-constraint violations. It also improves
power MAE versus `30714`, especially at alpha `1.0`.

But it does not improve the actual UC objective. The best asymmetric BCE case,
alpha `0.5`, is still `590.31` per day more expensive than `30714` and
`545.06` per day more expensive than the controlled cost-proxy weight-5 run
`35926`. Larger alpha values reduce status accuracy and worsen cost.

The likely reason is that false-ON weighting is too local. It can discourage
expensive OFF-label units, but it does not know whether the removed commitment
must be replaced by another unit with higher linear dispatch cost, worse ramp
position, or worse future feasibility. The final UC cost depends on the joint
commitment and dispatch trajectory, not only on generator-wise false-ON labels.

## Conclusion

Do not use asymmetric BCE as the next baseline. Keep `30714` as the physical
baseline and `35926` as the best current economic-loss diagnostic. The next
economic method should include the linear production term or use a
feasibility-preserving cost-aware commitment/decommitment step, rather than
increasing false-ON BCE pressure.
