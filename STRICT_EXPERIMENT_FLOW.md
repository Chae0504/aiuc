# Strict Experiment Flow

This document summarizes the strict-dataset experiment flow from job `22368`
to job `28562`, excluding the no-allocation strict baselines `22376` and
`22377`.

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

## Current Best Interpretation

Job `28562` is now the strongest strict-data result on the main metrics:

- mismatch MAE: `0.001 MW`
- cost gap: `+4.25%`
- power MAE: `12.81 MW`
- max mismatch: `117.57 MW`

The earlier trade-off between `22454` and `22496` is no longer the best
summary. Ramp-position-aware allocation kept the reliability benefit of
startup repair while also improving cost and average power accuracy.

## Next Research Direction

The next target should be the remaining `117.57 MW` tail case, not generic
hyperparameter tuning.

The current ramp-position layer only looks one hour ahead and only performs a
demand-neutral reallocation. That is why the next research question is whether
the residual tail needs:

- multi-step ramp positioning, not only `t+1`;
- a small operating-reserve margin in the ramp-position check;
- or a targeted repair for the new max-mismatch replay.

The one-step condition currently targeted is:

```text
sum_i min(Pmax_i, p_i,t + RU_i) * u_i,t+1 >= D_t+1
```

The next extension would generalize it to:

```text
sum_i U_i,t+k(p_t) >= D_t+k,  k = 1, ..., K
```

with a cost-aware rule for how much current output may be shifted to satisfy
future ramp reachability.
