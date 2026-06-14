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
| `strict_econ` | `slurm/run_rnncell_strict_econ.sh` | `train_rnncell_strict_econ.py` | Same architecture as `multiramp`, but Phase 2 optimizes a commitment cost proxy | `33220` | First economic-loss test; feasibility preserved but cost did not improve |

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

- status BCE weight: `0.5`
- power MAE weight: `1`
- balance mismatch weight: `0`
- commitment cost proxy weight: `0.05`

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
