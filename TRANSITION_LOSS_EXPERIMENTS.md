# Transition Loss Experiments

These runs keep the `30714` multi-step ramp-position architecture and change
only the status imitation loss:

```text
startup_i,t  = max(u_i,t - u_i,t-1, 0)
shutdown_i,t = max(u_i,t-1 - u_i,t, 0)

L_transition =
  mean(|startup_hat_i,t - startup_i,t|
     + |shutdown_hat_i,t - shutdown_i,t|)

L_status = BCE(u, u_hat) + lambda_transition * L_transition
```

No physical layer was added. The purpose is to make the neural part learn
Gurobi-like startup/shutdown timing rather than relying on another repair
algorithm.

## Results

The strict Gurobi reference average daily cost is `939,381.69`. Job `30714` is
the physical-feasibility baseline with AI average daily cost `979,891.13`.

| Weight | Job | Status Acc. | Power MAE | Mismatch MAE | Cost Gap | AI Daily Cost | Delta vs 30714 | Best Phase 2 Epoch | Inference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `30714` | 78.10% | 12.92 MW | 0.000004 MW | +4.31% | 979,891.13 | 0.00 | 2 | N/A |
| 0.5 | `42010` | 77.84% | 12.86 MW | 0.000004 MW | +4.38% | 980,501.50 | +610.38 | 1 | 7.16 ms/sample |
| 1.0 | `42043` | 78.21% | 12.83 MW | 0.000004 MW | +4.30% | 979,742.50 | -148.63 | 10 | 7.21 ms/sample |
| 1.5 | `42059` | 78.05% | 12.86 MW | 0.000004 MW | +4.35% | 980,288.81 | +397.69 | 8 | 7.38 ms/sample |
| 2.0 | `42060` | 77.12% | 12.59 MW | 0.000004 MW | +4.75% | 984,017.50 | +4,126.38 | 1 | 7.33 ms/sample |
| 5.0 | `42061` | 78.28% | 12.38 MW | 0.000004 MW | +4.27% | 979,464.38 | -426.75 | 41 | 7.19 ms/sample |
| 10.0 | `42062` | 74.27% | 14.12 MW | 0.000004 MW | +4.93% | 985,679.69 | +5,788.56 | 2 | 7.22 ms/sample |

All transition-loss runs preserve the solved physical behavior:

- mismatch max: `0.0005 MW`
- mismatch over 10 MW: `0.00%`
- ghost/capacity/ramp/startup-cap/shutdown-cap violations: zero
- MUT/MDT violations: zero

## Replay Diagnostics

The saved models were replayed on the same test split to recover transition and
commitment diagnostics. Full results are in
[`LEARNING_OBJECTIVE_REPLAY_METRICS.csv`](LEARNING_OBJECTIVE_REPLAY_METRICS.csv).

| Weight | Job | Cost Delta vs 30714 | False ON / day | Pred. online-hours / day | Startup count / day | Transition Event MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `30714` | 0.00 | 275.94 | 699.31 | 10.48 | 0.02649 |
| 0.5 | `42010` | +610.38 | 278.05 | 700.12 | 10.40 | 0.02696 |
| 1.0 | `42043` | -148.63 | 274.97 | 698.84 | 10.69 | 0.02679 |
| 1.5 | `42059` | +397.69 | 275.18 | 697.16 | 10.12 | 0.02665 |
| 2.0 | `42060` | +4,126.38 | 284.73 | 704.19 | 9.64 | 0.02632 |
| 5.0 | `42061` | -426.75 | 271.14 | 691.97 | 10.65 | 0.02721 |
| 10.0 | `42062` | +5,788.56 | 319.81 | 737.44 | 14.22 | 0.03161 |

## Interpretation

This is the first learning-objective change after `30714` that improves the
recorded cost without adding another physical layer. The best run is weight
`5.0` / job `42061`:

- cost improves by `426.75` per day versus `30714`;
- power MAE improves from `12.92 MW` to `12.38 MW`;
- status accuracy improves from `78.10%` to `78.28%`;
- numerical balance and hard feasibility remain unchanged.

Weight `1.0` / job `42043` is also meaningful: it improves cost by `148.63` per
day and reaches the best moderate-weight validation behavior. The result is not
monotonic, though. Weight `2.0` is clearly bad despite good power MAE, with a
cost penalty of `4,126.38` per day versus `30714`. Weight `10.0` is worse still:
it increases false-ON events by `43.87` generator-hours/day and online-hours by
`38.13` versus `30714`.

## Problems Found

The original evaluation problem was that jobs `42010` through `42062` did not
log transition-specific metrics. The saved models can be replayed, so the
metrics are recoverable, but it costs another inference pass per model. Future
evaluations now record:

- predicted and true online generator-hours per day;
- predicted and true startup/shutdown counts per day;
- startup, shutdown, and combined transition event MAE.

The replay revealed an important modeling problem: the best cost run does not
win by directly lowering transition event MAE. Job `42061` has a slightly worse
transition event MAE than `30714`, but it reduces false-ON events by `4.80` per
day and predicted online-hours by `7.35` per day. So the current transition loss
acts as an indirect commitment-shaping signal, not as a clean transition metric
optimizer.

The second problem is objective coupling. A lower power MAE does not guarantee
lower cost: job `42060` has strong power MAE but much worse cost. This confirms
that transition imitation alone can move the model into a different feasible
commitment structure whose dispatch is closer in MAE but economically worse.

The third problem is scale sensitivity. The useful weights are not monotonic:
`1.0` and `5.0` help, while `0.5`, `1.5`, `2.0`, and `10.0` hurt. The next
sweep should therefore focus locally around the two promising regions:

- moderate: `0.8`, `1.0`, `1.2`;
- high: `4.0`, `5.0`, `6.0`.

## Conclusion

Transition loss is more promising than the asymmetric BCE sweep because it
improves cost while preserving feasibility and without adding another physical
layer. The current best transition model is job `42061`; `30714` remains the
conservative physical baseline, but `42061` is the best learning-objective
branch so far.
