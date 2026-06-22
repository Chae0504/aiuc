# Strict Experiment Flow

This document summarizes the strict-dataset experiment flow from job `22368`
to the transition-loss sweep ending at job `42061`, excluding the
no-allocation strict baselines `22376` and `22377`.

The main question was: how can the model reduce power-balance mismatch while
keeping strict UC feasibility and avoiding unnecessary cost increase?

The strict Gurobi reference average daily cost is `939,381.69`.

## Flow Summary

| Step | Job | Result That Drove the Next Step | New Objective | Method Introduced | Did It Improve? | Lesson |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `22368` | Proportional allocation reached `59.18 MW` mismatch MAE and `+5.56%` cost gap. This was much better than no-allocation baselines, but still too much mismatch for a strict UC surrogate. | Reduce mismatch more aggressively without breaking hard clipping constraints. | Ramp-aware proportional allocation after the RNN cell. | Partially. Status accuracy was high at `82.81%`, but mismatch remained too large. | Allocation is necessary, but allocation alone cannot fix bad commitment decisions. |
| 2 | `22441` | MUT/MDT-aware repair reduced mismatch MAE to `2.04 MW`, but cost gap rose to `+11.65%` and tail mismatch max was `859.03 MW`. | Keep the large mean-mismatch gain while reducing rare but severe tail failures. | MUT/MDT-aware commitment repair. | Yes for average mismatch, no for tail risk and cost. | Hard commitment repair can solve many balance cases, but local repair can still miss future demand jumps. |
| 3 | `22453` | Look-ahead repair reduced shortage MAE to almost zero (`0.008 MW`), but residual mismatch became mostly excess (`2.49 MW`) and cost stayed high at `+8.68%`. | Avoid conservative over-generation and reduce cost while preserving the shortage improvement. | Look-ahead commitment repair with future shutdown feasibility checks. | Mixed. Worst mismatch improved from `859.03 MW` to `376.09 MW`, but average mismatch and cost were not yet ideal. | Future-aware shutdown decisions help shortage risk, but dispatch/allocation still needs economic structure. |
| 4 | `22454` | Cost-aware allocation reduced mismatch MAE to `0.016 MW`, power MAE to `12.92 MW`, and cost gap to `+4.40%`. However, one rare max-shortage case remained at `378.04 MW`. | Keep the strong average and cost performance, then identify and reduce rare max-mismatch events. | Cost-aware allocation branch from the look-ahead repair model. | Strongly yes. This became the best cost-aware baseline. | Once balance is mostly fixed, the remaining problem moves from average mismatch to rare ramp/startup tail events. |
| 5 | `22455`, `22461`, `22468` | The max case in `22454` looked like a possible reserve issue, so safety margins were tested. Margins of `10 MW` and `25 MW` matched `22454`; `50 MW` worsened max mismatch to `394.30 MW`. | Test whether the rare shortage tail was caused by insufficient shutdown headroom. | Look-ahead shutdown safety margin. | No. It did not improve the tail. | The tail was not primarily a shutdown-margin problem. The model needed earlier startup and ramp positioning. |
| 6 | `22496` | Startup repair reduced the max mismatch from `378.04 MW` to `269.35 MW`, while keeping mismatch MAE at `0.015 MW`. Cost gap slightly worsened from `+4.40%` to `+4.60%`. | Reduce late-startup shortage tails while retaining the cost-aware allocation gains. | Look-ahead startup repair branch from `22454`. | Yes for worst-case shortage, slightly worse for cost and over-10MW frequency. | Startup repair solved the late-startup part of the tail. The remaining max case is now mainly ramp-positioning, not commitment availability. |
| 7 | `28562` | The `22496` max replay showed that all units were already online, but previous-hour dispatch was too low for the next ramp. | Keep cost-aware dispatch, but pre-position current output so the next hour is reachable under ramp limits. | One-step ramp-position-aware allocation after cost-aware dispatch. | Yes. Mismatch max fell to `117.57 MW`, mismatch MAE fell to `0.001 MW`, and cost gap improved to `+4.25%`. | The remaining tail was mostly a dispatch-positioning issue. Reallocating current output without changing total generation can improve both reliability and cost. |
| 8 | `30714` | The `28562` replay showed that the one-step repair ran out of receiver capacity at hour 19 before the hour-20 demand jump. | Pre-position output earlier than one hour ahead while preserving current balance and hard constraints. | Multi-step ramp-position-aware allocation. | Yes for feasibility. Mismatch max fell to `0.0005 MW` and all evaluated hard violations stayed zero. Cost gap rose slightly to `+4.31%`. | Physical feasibility is now essentially solved on the test set; the next question is economic optimality and closer Gurobi-like structure. |
| 9 | `33220` | `30714` solved physical feasibility, but retained a `+4.31%` cost gap and restored Phase 2 best epoch 2. | Redirect Phase 2 from balance toward cheaper commitment patterns. | No-load plus startup cost proxy; Phase 2 status/power/balance/cost weights=`0.5/1/0/0.05`. | No economically. Feasibility stayed perfect and power MAE improved slightly, but cost gap worsened to `+4.39%`. | The proxy term was tiny relative to BCE/power loss. Because BCE was also halved, this run evaluates the combined weighting rather than isolating the proxy itself. |
| 10 | `35385`, `35926`, `36329` | The first proxy run was confounded by a reduced BCE weight and an extremely small proxy contribution. | Isolate proxy scale while restoring BCE/power weights to `1/1` and keeping balance at `0`. | Controlled commitment-cost proxy sweep with weights `1`, `5`, and `10`; identical Phase 1 and 2-GPU batch configuration. | Partially. Weight 5 nearly matches 30714 cost while preserving feasibility, but remains 45.25/day higher. Weight 10 worsens total cost. | The proxy can shape commitment cost, but omitting linear production cost prevents monotonic improvement in the true UC objective. |
| 11 | `39104`, `40234`, `41987` | Proxy-only pressure did not reliably reduce true cost, and unnecessary false-ON commitments were still a suspect. | Test whether status imitation can directly discourage expensive false-ON commitments while preserving 30714 feasibility. | Cost-weighted asymmetric BCE for false-ON labels; alpha sweep `0.5`, `1.0`, `1.5`; 2 GPUs, global batch `64`. | No economically. Feasibility stayed perfect and power MAE improved, but best cost gap worsened to `+4.38%`. | Generator-wise false-ON weighting is too local; true cost depends on the replacement commitment, linear dispatch, and future ramp trajectory. |
| 12 | `42010`, `42043`, `42059`, `42060`, `42061` | Asymmetric BCE was too local, but the learning objective still needed to target Gurobi-like commitment structure. | Learn startup/shutdown timing directly without adding another physical layer. | Transition BCE: status BCE plus startup/shutdown event MAE; weights `0.5`, `1.0`, `1.5`, `2.0`, `5.0`. | Yes at selected scales. Weight 5 improves cost by `426.75` per day vs `30714`, preserves feasibility, and improves power MAE. | Transition imitation is promising, but scale-sensitive; future evaluation must log transition-specific metrics. |

## Narrative

### 1. From Allocation to Commitment Repair

Job `22368` showed that proportional allocation was a major structural
improvement over direct clipped RNN outputs. It produced a feasible-looking
dispatch shape and kept cost relatively close to the strict Gurobi reference.
However, `59.18 MW` mismatch MAE was still too large.

This led to the first important conclusion: the model was not only failing at
continuous power allocation. Some hours needed better binary commitment
decisions before allocation could succeed.

### 2. From Mean Mismatch to Tail Risk

Job `22441` introduced MUT/MDT-aware commitment repair. This was a major jump:
mismatch MAE fell from `59.18 MW` to `2.04 MW`.

But the result also exposed a new failure mode. The average mismatch looked
good, while `mismatch_max` was still `859.03 MW`. That means the model had
learned to fix ordinary hours, but rare high-ramp hours could still fail badly.

The research objective therefore changed from "reduce average mismatch" to
"reduce tail shortage without damaging feasibility."

### 3. From Repair to Look-Ahead Repair

Job `22453` added look-ahead shutdown logic. This nearly eliminated shortage
MAE, but it did so conservatively: the remaining mismatch was mostly excess
generation, and the cost gap remained high.

This was useful because it separated two issues:

- look-ahead repair can prevent many shortage cases;
- but if allocation ignores cost, it can keep too much expensive capacity or
  produce unnecessary output.

So the next target became economic dispatch inside the allocation layer.

### 4. Cost-Aware Allocation as the Best Average Model

Job `22454` added cost-aware allocation and produced the clearest improvement:
mismatch MAE became almost zero, power MAE improved, and cost gap dropped to
`+4.40%`.

At this point, the main model problem changed again. The average behavior was
now strong enough that the important question became rare-event reliability:
why did a `378.04 MW` max shortage still appear?

The max-mismatch replay showed that the problem was not average allocation.
It came from a steep demand jump where the model did not prepare enough
startup/ramp capability in the previous hour.

### 5. Why Safety Margin Was Not the Answer

The margin experiments tested whether adding extra shutdown headroom would
reduce this rare tail. It did not. Jobs `22455` and `22461` were essentially
identical to `22454`, and job `22468` slightly worsened the max mismatch.

That negative result was important. It ruled out a simple explanation: the
tail was not caused by a too-small shutdown reserve margin. The failure was
more specific: the model needed proactive startup and ramp positioning before
large future demand jumps.

### 6. Startup Repair Reduced the Worst Case

Job `22496` added look-ahead startup repair from the `22454` branch. This
reduced the max mismatch from `378.04 MW` to `269.35 MW` while preserving the
near-zero average mismatch.

The remaining max-mismatch analysis showed that all units were online at the
critical hour, so late startup was no longer the main issue. The remaining
shortage came from ramp positioning: some units were online, but their previous
hour dispatch was too low to ramp enough into the demand spike.

### 7. Ramp-Position Allocation Improved Both Tail and Cost

Job `28562` added one-step ramp-position-aware allocation on top of startup
repair and cost-aware dispatch. Instead of only meeting demand at hour `t`, it
reallocated current output so that hour `t+1` was more reachable under ramp
limits.

This directly targeted the remaining `22496` failure mode. The result was the
strongest strict-data run so far:

- max mismatch: `269.35 MW` to `117.57 MW`
- mismatch MAE: `0.015 MW` to `0.001 MW`
- cost gap: `+4.60%` to `+4.25%`
- mismatch over 10 MW: `0.019%` to `0.00083%`

The important point is that the method did not simply add reserve by
over-generating. It moved generation from units that could donate output
without hurting next-hour reachability to units whose current output improved
future ramp capability.

### 8. Multi-Step Ramp Positioning Closed the Balance Tail

Job `30714` extended the ramp-position layer from one-step to multi-step
look-ahead. This was motivated by the `28562` max case: at hour 19 the model
could no longer shift enough output to make hour 20 reachable, because all
useful receiver capacity was already exhausted. The model needed to begin
positioning earlier.

The result essentially closed the remaining balance tail:

- max mismatch: `117.57 MW` to `0.0005 MW`
- mismatch over 10 MW: `0.00083%` to `0.00%`
- hard violations: all zero

The trade-off is economic rather than physical. Cost gap moved from `+4.25%`
to `+4.31%`, and Phase 2 restored best epoch 2, suggesting that the heuristic
layer now dominates feasibility and the neural training objective is mostly
tuning commitment/output similarity rather than learning balance feasibility.

### 9. The First Commitment Cost Proxy Was Too Weak

Job `33220` retained the full `30714` architecture and replaced the Phase 2
balance objective with a normalized no-load and startup cost proxy. The status
BCE weight was also reduced from `1.0` to `0.5`.

Physical performance was preserved:

- mismatch MAE: `0.000004 MW`
- mismatch max: `0.0005 MW`
- mismatch over 10 MW: `0.00%`
- all evaluated hard violations: zero

However, the economic target did not improve. Average daily cost increased
from `979,891.13` to `980,649.38`, so the Gurobi cost gap worsened from
`+4.31%` to `+4.39%`.

A saved-model replay decomposed the change relative to `30714`:

| Component | 30714 | 33220 | Change |
| --- | ---: | ---: | ---: |
| Linear production cost | 961,456.94 | 962,109.44 | +652.50 |
| No-load cost | 17,030.38 | 17,137.53 | +107.15 |
| Startup cost | 1,403.34 | 1,404.23 | +0.89 |
| Online generator-hours/day | 699.31 | 703.69 | +4.38 |
| Startups/day | 10.479 | 10.731 | +0.252 |

The result moved in the opposite direction from the proxy's intent. With the
current normalizer and weight, the proxy contributes only about `0.0004` to
total loss, which is tiny compared with the BCE and power terms. Halving the
BCE weight likely changed the learned commitment structure more strongly than
the proxy discouraged expensive commitments. Because both weights changed in
the same run, this experiment does not isolate their individual effects.

### 10. Controlled Proxy Scaling Found a Middle Optimum

Jobs `35385`, `35926`, and `36329` used identical Phase 1 histories, two GPUs,
global batch `128`, and Phase 2 BCE/power/balance weights of `1/1/0`. Only the
cost-proxy weight changed.

Weight `5` was the best setting:

- weight 1: cost gap `+4.36%`
- weight 5: cost gap `+4.32%`
- weight 10: cost gap `+4.33%`

All three preserved numerical balance and zero evaluated hard violations.
The best validation proxy decreased more as weight increased, from a `1.36%`
reduction at weight 1 to `2.70%` at weight 10. Nevertheless, true cost was not
monotonic.

From weight 5 to weight 10, no-load plus startup cost decreased by about
`7.31` per day, but linear production cost increased by about `104.94`.
This demonstrates that the current proxy is internally effective but
incomplete. The next objective should include linear production cost rather
than applying still more pressure to no-load and startup terms.

### 11. Asymmetric BCE Was a Useful Negative Diagnostic

Jobs `39104`, `40234`, and `41987` kept the full `30714` architecture and
changed only the status imitation term. The BCE loss was weighted more heavily
when the Gurobi label was OFF and the predicted unit was ON, with generator
weights proportional to no-load plus minimum-output linear cost.

This tested a specific hypothesis: if the model is more strongly punished for
turning on expensive OFF-label units, it might reduce unnecessary commitments
and become cheaper without changing the deterministic feasibility layers.

The physical result was exactly as desired:

- mismatch MAE stayed around `0.000004 MW`;
- max mismatch stayed `0.0005 MW`;
- mismatch over 10 MW stayed `0.00%`;
- all evaluated hard violations stayed zero.

The economic result was negative:

- alpha `0.5` / job `39104`: cost gap `+4.38%`;
- alpha `1.0` / job `40234`: cost gap `+4.79%`;
- alpha `1.5` / job `41987`: cost gap `+4.59%`.

Alpha `0.5` was the best setting, but it was still `590.31` per day more
expensive than `30714`. Alpha `1.0` achieved the lowest power MAE in the sweep
(`12.33 MW`), but its status accuracy fell to `76.45%` and cost worsened by
`4,474.06` per day relative to `30714`.

The lesson is that false-ON weighting is not enough to optimize UC cost. It can
discourage one generator from being online, but it does not know whether the
replacement schedule requires more expensive production, creates a worse ramp
position, or forces future commitment changes. The next economic step should
optimize the joint commitment-dispatch cost, especially the linear production
term, or use a feasibility-preserving economic projection/decommitment layer.

### 12. Transition Loss Revived the Learning Objective

Jobs `42010`, `42043`, `42059`, `42060`, and `42061` kept the full `30714`
physical architecture and added a startup/shutdown transition imitation term to
the status loss. This avoided adding another physical layer while giving the
neural model a more structural target than pointwise BCE.

The result is the first post-30714 learning objective that clearly improves
cost while preserving the already solved feasibility behavior:

- weight `1.0` / job `42043`: cost improves by `148.63` per day versus `30714`;
- weight `5.0` / job `42061`: cost improves by `426.75` per day versus `30714`;
- all transition runs keep mismatch max at `0.0005 MW` and evaluated hard
  violations at zero.

The best run, `42061`, also improves power MAE from `12.92 MW` to `12.38 MW`
and status accuracy from `78.10%` to `78.28%`. This is a better signal than the
asymmetric BCE result because it improves cost, power similarity, and status
similarity at the same time.

The problem is scale sensitivity. Weight `2.0` gives good power MAE
(`12.59 MW`) but worsens cost by `4,126.38` per day versus `30714`. This shows
that matching transition shape can still land in an economically worse feasible
commitment if the weight pushes the model into a bad local structure.

Another problem is measurement. The original evaluator did not log startup and
shutdown event errors, so these five runs cannot prove directly that transition
timing improved. The evaluator now records online-hours, startup/shutdown
counts, and transition event MAE for future runs.

## Current Best Interpretation

Job `30714` is now the strongest strict-data result for physical feasibility:

- mismatch MAE: `0.000004 MW`
- max mismatch: `0.0005 MW`
- mismatch over 10 MW: `0.00%`
- hard constraint violations: zero

Job `28562` remains slightly better economically (`+4.25%` cost gap vs
`+4.31%`) and has slightly lower power MAE, but it retains a `117.57 MW` rare
shortage. The current frontier is therefore:

- `30714` for strict physical feasibility;
- `28562` for slightly better cost with one rare balance tail.

Job `33220` does not replace either frontier model. It verifies that a
differentiable commitment-cost signal can be added without breaking physical
feasibility, but the first scaling and loss balance are not economically
effective.

The controlled sweep confirms weight `5` as the best current economic-loss
variant, but `30714` remains slightly cheaper and remains the physical
baseline.

The asymmetric BCE sweep confirms that generator-wise false-ON penalties are
also insufficient as a standalone economic objective. It preserves feasibility
but does not beat either `30714` or the controlled proxy weight-5 result.

The transition-loss sweep is more promising: `42061` is cheaper than `30714`
without adding another physical layer. Treat `30714` as the conservative
physical baseline and `42061` as the best current learning-objective branch.

## Next Research Direction

The next target should be economic optimality and Gurobi-like structure, not
more balance repair.

The model now produces feasible dispatch through deterministic repair layers.
The remaining research question is how to make those feasible dispatches closer
to the reference UC optimum:

- reduce unnecessary commitments;
- reduce cost while preserving feasibility;
- improve status/power similarity to the Gurobi reference;
- and measure inference-time overhead from the heuristic layers.

The next likely methods are:

- cost-aware commitment pruning after feasibility repair;
- stronger imitation learning for commitment patterns;
- a full differentiable cost objective including linear production cost;
- or a small feasibility-preserving economic projection layer.
