# Cost Proxy Experiments

This document records the controlled Phase 2 commitment-cost proxy sweep from
the `30714` multi-step ramp-position architecture.

All three runs used:

- two RTX 4090 GPUs with `MirroredStrategy`
- global batch size `128` and per-replica batch size `64`
- status BCE weight `1`
- power MAE weight `1`
- balance weight `0`
- byte-identical Phase 1 training histories

The strict Gurobi reference average daily cost is `939,381.69`.

## Results

| Proxy Weight | Job | Status Acc. | Power MAE | Cost Gap | Daily Cost | Best Epoch | Best Val. Proxy | Proxy Change | Inference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `35385` | 77.84% | 13.04 MW | +4.36% | 980,369.19 | 5 | 0.007860 | -1.36% | 7.19 ms/sample |
| 5 | `35926` | 77.99% | 12.95 MW | +4.32% | 979,936.38 | 5 | 0.007829 | -2.53% | 7.32 ms/sample |
| 10 | `36329` | 77.89% | 12.96 MW | +4.33% | 980,048.50 | 5 | 0.007819 | -2.70% | 7.16 ms/sample |

All three runs retained:

- mismatch max: `0.000488 MW`
- mismatch over 10 MW: `0.00%`
- zero evaluated capacity, ramp, startup-cap, shutdown-cap, ghost-power, and
  minimum-time violations

## Cost Decomposition

Saved-model replay on the common 5,000-sample test split produced:

| Component | 30714 | Weight 1 | Weight 5 | Weight 10 |
| --- | ---: | ---: | ---: | ---: |
| Linear production cost | 961,456.94 | 961,783.81 | 961,450.00 | 961,554.94 |
| No-load cost | 17,030.38 | 17,121.40 | 17,041.68 | 17,035.53 |
| Startup cost | 1,403.34 | 1,460.81 | 1,454.23 | 1,453.07 |
| Online generator-hours/day | 699.31 | 705.34 | 703.07 | 704.13 |
| Startups/day | 10.479 | 11.433 | 11.291 | 11.292 |

Weight `5` is the best tested proxy scale. It reduces official daily cost by
`432.81` relative to weight `1`, but remains `45.25` above `30714`.

Increasing the weight from `5` to `10` lowers no-load and startup cost by only
about `7.31` per day, while linear production cost rises by about `104.94`.
The resulting official total cost therefore worsens by `112.13`.

## Interpretation

The proxy is behaving according to its definition: larger weights generally
reduce the normalized no-load plus startup objective. However, that objective
is not the complete UC cost.

The missing term is linear production cost:

\[
C^{linear} = \sum_{t,i} c_i p_{i,t}.
\]

A stronger commitment-only proxy can favor schedules with slightly cheaper
online/startup structure while moving dispatch and ramp positioning toward a
more expensive production mix. This is visible in the weight `10` result.

The startup term may also be misaligned with imitation of the Gurobi schedule.
The current models average roughly `10.5-11.4` startups per day, while the
strict Gurobi labels average about `15.15`. The remaining cost gap is therefore
not caused by excessive startup count alone.

The next economic objective should include linear production cost rather than
further increasing the current proxy weight. Weight `5` is the sensible
reference setting for that next branch.

## Two-GPU Timing

The controlled two-GPU runs used global batch `128` and took a median
`86-87 seconds` per training epoch. The one-GPU `30714` run used global batch
`64` and took a median `117 seconds`.

This is about a `26%` epoch-time reduction, or roughly `1.34x` throughput.
Because batch size also changed, this is not a pure GPU-count benchmark.
Inference remained around `36 seconds` for 5,000 samples and did not improve
over the one-GPU measurement; distributed inference overhead dominates at this
model size.
