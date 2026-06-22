# Strict Model Summary

This file keeps the strict-data model branches readable. `30714` is the
current physical-feasibility baseline.

## Main Branches

| Short Name | Script | Train File | Main Idea | Representative Job | Role |
| --- | --- | --- | --- | --- | --- |
| `strict_clip` | `baselines/strict_clipping/run_rnncell_strict.sh` | `baselines/strict_clipping/train_rnncell_strict.py` | Strict lower/upper clipping without allocation | `22376`, `22377` | Negative baseline; physical clipping alone leaves large mismatch |
| `strict_alloc` | `slurm/run_rnncell_strict_allocation.sh` | `train_rnncell_strict_allocation.py` | Proportional allocation after strict RNN cell | `22368` | First useful allocation baseline |
| `repair` | `slurm/run_rnncell_strict_allocation_repair.sh` | `train_rnncell_strict_allocation_repair.py` | MUT/MDT-aware commitment repair | `22441` | Large average mismatch reduction, but high tail/cost |
| `lookahead` | `slurm/run_rnncell_strict_allocation_lookahead_repair.sh` | `train_rnncell_strict_allocation_lookahead_repair.py` | Future-aware shutdown repair | `22453` | Reduces shortage risk but can create excess/cost |
| `cost_aware` | `slurm/run_rnncell_strict_allocation_cost_aware.sh` | `train_rnncell_strict_allocation_cost_aware.py` | Cost-aware allocation with look-ahead repair | `22454` | Best early economic/average-mismatch model |
| `startup` | `slurm/run_rnncell_strict_allocation_startup_repair.sh` | `train_rnncell_strict_allocation_startup_repair.py` | Look-ahead startup repair | `22496` | Reduces late-startup tail cases |
| `ramp_pos` | `slurm/run_rnncell_strict_allocation_ramp_position.sh` | `train_rnncell_strict_allocation_ramp_position.py` | One-step ramp-position-aware allocation | `28562` | Better tail and cost than startup branch, but one rare shortage remains |
| `multiramp` | `slurm/run_rnncell_strict_allocation_multistep_ramp_position.sh` | `train_rnncell_strict_allocation_multistep_ramp_position.py` | Multi-step ramp-position-aware allocation | `30714` | Current physical-feasibility baseline |
| `strict_econ` | `slurm/run_rnncell_strict_econ.sh` | `train_rnncell_strict_econ.py` | Same architecture as `multiramp`, but Phase 2 optimizes a commitment cost proxy | `35385`, `35926`, `36329` | Weight 5 is the best controlled proxy scale, but does not beat 30714 |
| `asym_bce` | `slurm/run_rnncell_strict_asym_bce_2gpu.sh` | `train_rnncell_strict_asym_bce.py` | Same architecture as `multiramp`, but false-ON status BCE is weighted by commitment cost | `39104`, `40234`, `41987` | Negative diagnostic; preserves feasibility but worsens cost versus 30714 |
| `transition` | `slurm/run_rnncell_strict_transition_2gpu.sh` | `train_rnncell_strict_transition.py` | Same architecture as `multiramp`, but status loss includes startup/shutdown transition imitation | `42010`, `42043`, `42059`, `42060`, `42061`, `42062` | Best learning-objective branch so far; weight 5 improves cost without adding a physical layer |

## Current Baseline

Use `multiramp` / job `30714` as the strict physical-feasibility baseline:

- mismatch MAE: `0.000004 MW`
- mismatch max: `0.0005 MW`
- mismatch over 10 MW: `0.00%`
- evaluated hard violations: zero
- cost gap: `+4.31%`

This means the next meaningful target is not additional balance repair. The
next target is cheaper, more Gurobi-like feasible schedules.

## Economic-Loss Result: `strict_econ`

`strict_econ` keeps the `30714` architecture and changes only the Phase 2
training objective:

- status BCE weight: `1`
- power MAE weight: `1`
- balance mismatch weight: `0`
- commitment cost proxy weight: controlled by `COST_PROXY_WEIGHT`

The purpose is to stop spending Phase 2 effort on a balance loss that is already
solved by the deterministic allocation/repair layers, and instead push the
neural commitment pattern toward cheaper no-load/startup behavior.

Job `33220` showed that this first weighting was too weak to achieve that
purpose:

- cost gap: `+4.31%` to `+4.39%`
- status accuracy: `78.10%` to `77.79%`
- power MAE: `12.92 MW` to `12.88 MW`
- physical violations and balance: unchanged at numerical tolerance
- online generator-hours: `+4.38` per day versus `30714`
- startups: `+0.252` per day versus `30714`

The normalized cost term contributed only about `0.0004` to the total loss at
weight `0.05`, while halving the status BCE weight had a much larger effect.
Therefore `30714` remains the baseline and `33220` is retained as a negative
weighting result.

New controlled runs keep BCE and power weights fixed and vary only the proxy
scale. The default proxy weight is `5.0`:

```bash
sbatch slurm/run_rnncell_strict_econ.sh
COST_PROXY_WEIGHT=1 sbatch slurm/run_rnncell_strict_econ.sh
COST_PROXY_WEIGHT=10 sbatch slurm/run_rnncell_strict_econ.sh
```

Each Phase 1/2 CSV records both `normalized_commitment_cost_proxy` and
`weighted_commitment_cost_proxy`.

For two GPUs on `ego-g01`:

```bash
COST_PROXY_WEIGHT=5 sbatch slurm/run_rnncell_strict_econ_2gpu.sh
```

The controlled `1/5/10` sweep found weight `5` to be best:

- weight 1: cost gap `+4.36%`
- weight 5: cost gap `+4.32%`
- weight 10: cost gap `+4.33%`

Weight `10` reduces the commitment proxy slightly more but increases linear
production cost. The next branch should add linear production cost to the
economic objective instead of increasing this proxy further. See
[`COST_PROXY_EXPERIMENTS.md`](COST_PROXY_EXPERIMENTS.md).

## Asymmetric BCE Result: `asym_bce`

`asym_bce` also keeps the `30714` architecture, but changes the status
imitation loss instead of adding a separate cost proxy. False-ON errors are
weighted by generator commitment cost.

The `alpha=0.5/1.0/1.5` sweep preserved physical feasibility, but did not
improve cost:

- alpha 0.5 / job `39104`: cost gap `+4.38%`
- alpha 1.0 / job `40234`: cost gap `+4.79%`
- alpha 1.5 / job `41987`: cost gap `+4.59%`

Alpha `0.5` is the best asymmetric-BCE setting, but it remains `590.31` per day
more expensive than `30714`. See
[`ASYMMETRIC_BCE_EXPERIMENTS.md`](ASYMMETRIC_BCE_EXPERIMENTS.md).

## Transition-Loss Result: `transition`

`transition` keeps the `30714` physical architecture and adds startup/shutdown
timing imitation to the status loss. This is the first post-30714 learning
objective that improves cost without adding another physical layer.

The `0.5/1.0/1.5/2.0/5.0/10.0` sweep found two useful regions:

- weight 1.0 / job `42043`: cost gap `+4.30%`, `148.63` per day cheaper than `30714`
- weight 5.0 / job `42061`: cost gap `+4.27%`, `426.75` per day cheaper than `30714`

Weight `2.0` is a cautionary case: power MAE improves, but cost worsens sharply
to `+4.75%`. Weight `10.0` is worse at `+4.93%`, with false-ON events up
`43.87` per day versus `30714`. The objective is useful but scale-sensitive. See
[`TRANSITION_LOSS_EXPERIMENTS.md`](TRANSITION_LOSS_EXPERIMENTS.md).
