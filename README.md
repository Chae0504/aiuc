# AIUC

Physics-informed recurrent neural network experiments for unit commitment.

## SLURM Training

Legacy RNN-cell workflows are retained under `legacy/` for historical
comparison. Submit the original modular RNN-cell training job with:

```bash
sbatch legacy/run_rnncell.sh
```

Each SLURM job writes artifacts to its own directory:

```text
outputs/rnncell_<job_id>/
```

Important files:

```text
run_configuration.json  # hyperparameters, normalizers, and Git commit
phase1_training.csv     # epoch metrics for status-focused training
phase2_training.csv     # epoch metrics for hybrid fine-tuning
evaluation.json         # final test metrics
phase1_best.keras       # best Phase 1 checkpoint
phase2_best.keras       # best Phase 2 checkpoint
```

Monitor Phase 2 training:

```bash
tail -F outputs/rnncell_<job_id>/phase2_training.csv
```

The training CSV files include `out_power_mismatch_mae_mw` and
`out_power_normalized_mismatch` so the power-balance trade-off can be monitored
at each epoch.

## Ramp-Aware Allocation Variant

The legacy baseline model remains in `legacy/rnncell_model.py`. To train the separate variant
that proportionally allocates residual demand over online generators within
their ramp-aware headroom, submit:

```bash
sbatch legacy/run_rnncell_allocation.sh
```

Allocation experiments write artifacts to a separate directory:

```text
outputs/rnncell_allocation_<job_id>/
```

Generate their experiment-log row with:

```bash
python summarize_experiment.py outputs/rnncell_allocation_<job_id>
```

The allocation layer stays inside the recurrent cell so the adjusted generator
outputs are used as the previous-hour state for the next ramp calculation.
Residual demand that cannot be covered by the online generators remains in the
balance loss and can still guide the commitment decision.

## Strict UC Dataset Generation

The original `DG/uc_new_data.npz` is preserved for comparison. Generate a separate
50,000-sample dataset with hard power-balance, capacity, startup, shutdown,
ramp, and minimum-time constraints on this server:

```bash
module load julia/1.11.3 gurobi/13.0.1
julia --project=DG -e 'using Pkg; Pkg.instantiate()'
./DG/run_strict_uc_generation.sh
```

The generator reads `DG/generator_specs.csv` and writes:

```text
DG/uc_new_data_strict.npz
```

Generation is reproducible and configurable through environment variables:

```bash
NUM_SAMPLES=10 RANDOM_SEED=42 OUTPUT_PATH=uc_strict_smoke.npz \
  ./DG/run_strict_uc_generation.sh
```

Long runs save a resumable checkpoint every 1,000 accepted samples. Restart the
same command to resume from `<output>.partial.npz` and `<output>.partial.rng`.

Run the 50,000-sample generation independently of the current terminal:

```bash
systemd-run --user --unit=aiuc-strict-uc-generation \
  --property=WorkingDirectory="$PWD" \
  /usr/bin/bash -lc './DG/run_strict_uc_generation.sh > DG/strict_uc_generation.out 2>&1'
```

Monitor the detached service and its progress log:

```bash
systemctl --user status aiuc-strict-uc-generation.service --no-pager
tail -F DG/strict_uc_generation.out
```

Validate a generated dataset before training:

```bash
python DG/validate_strict_uc_dataset.py DG/uc_new_data_strict.npz
```

Visualize the hourly mean demand and 10 reproducibly sampled profiles:

```bash
python DG/visualize_strict_demand.py
```

The strict hourly model uses `SUcap` and `SDcap`. The available generator CSV
does not uniquely define the intermediate output trajectory for multi-hour
startup and shutdown processes, so those trajectories are outside this dataset
variant.

## Strict Ramp-Aware Allocation Training

After `DG/uc_new_data_strict.npz` has been generated and validated, train the
separate strict allocation model:

```bash
sbatch run_rnncell_strict_allocation.sh
```

Use this runner for new strict-data experiments. The older
`legacy/run_rnncell.sh` and `legacy/run_rnncell_allocation.sh` workflows remain available only for legacy
comparison runs.

This variant keeps the previous allocation baseline intact. It uses the signed
initial duration from `IniState`, applies `SUcap` to `OFF -> ON`, blocks
`ON -> OFF` while the previous output exceeds `SDcap`, enforces stay-online
ramps, and blocks unverifiable terminal transitions in the same way as the
strict dataset generator. Artifacts are written to:

```text
outputs/rnncell_strict_allocation_<job_id>/
```

Generate an experiment-log row with:

```bash
python summarize_experiment.py outputs/rnncell_strict_allocation_<job_id>
```

## MUT/MDT-Aware Commitment Repair

Train the separate strict allocation variant that repairs commitment only
through transitions allowed by minimum-time, shutdown-cap, and terminal
constraints:

```bash
sbatch run_rnncell_strict_allocation_repair.sh
```

The repair first keeps or starts eligible generators until online upper bounds
can cover demand. It then removes a safe low-score prefix only when the
remaining upper bounds can still cover demand. This vectorized prefix rule is
deliberately conservative. If allowed transitions are insufficient, the model
preserves the hard constraints and leaves the remaining mismatch.
This is an hourly greedy repair layer, not a replacement for horizon-wide UC
cost optimization.

Artifacts are written to:

```text
outputs/rnncell_strict_allocation_repair_<job_id>/
```

## Look-Ahead Commitment Repair

To compare the hourly repair baseline with a future-aware shutdown rule, submit:

```bash
sbatch run_rnncell_strict_allocation_lookahead_repair.sh
```

This separate variant exposes the next `max(MDT)=15` demand values to the hard
repair layer. Before removing a low-score generator, it verifies that the
ramp-aware upper bounds of the remaining commitment can cover the look-ahead
demand window. A removed generator is counted again only from its earliest
MDT-feasible restart, with `SUcap` and subsequent ramps applied. The check is
vectorized over generators and future hours. AI-proposed `ON -> OFF`
transitions are also reconsidered by the same check before they become final.

The rule targets tail shortage after premature shutdown. It remains a
conservative feasibility heuristic, not a horizon-wide economic UC optimizer.
Artifacts are written to:

```text
outputs/rnncell_strict_allocation_lookahead_repair_<job_id>/
```

## Cost-Aware Allocation

To keep the look-ahead commitment repair while dispatching online units in
economic merit order, submit:

```bash
sbatch run_rnncell_strict_allocation_cost_aware.sh
```

This branch preserves the repaired commitment and replaces proportional
headroom allocation with a cost-aware allocation over the same ramp-aware
`L_i,t` and `U_i,t` bounds. It starts online units at their lower bounds, then
fills residual demand from the lowest `SlopeVarCost` headroom first. The layer
targets the excess-generation and cost increase observed in the conservative
look-ahead repair branch.

Artifacts are written to:

```text
outputs/rnncell_strict_allocation_cost_aware_<job_id>/
```

To test a reserve buffer in the look-ahead shutdown check, submit:

```bash
sbatch run_rnncell_strict_allocation_cost_aware_margin.sh
```

The default margin is `25 MW`. Override it per submission with:

```bash
LOOKAHEAD_SAFETY_MARGIN_MW=10 sbatch run_rnncell_strict_allocation_cost_aware_margin.sh
```

The margin changes the look-ahead feasibility test from `capacity >= demand`
to `capacity >= demand + margin`, targeting rare shortage tails such as
`mismatch_max_mw` and `mismatch_over_10mw_percent`.

To instead start eligible offline units early when future ramp-aware capacity
is short, submit:

```bash
sbatch run_rnncell_strict_allocation_startup_repair.sh
```

This branch keeps the cost-aware allocation from `22454`, but adds a proactive
startup repair before the shutdown-removal pass. It targets rare cases where
the model has the correct commitment at the peak hour but starts too many units
one hour late, leaving them limited by `SUcap` and ramp-up constraints.

Artifacts are written to:

```text
outputs/rnncell_strict_allocation_startup_repair_<job_id>/
```

## Strict-Clipping Baseline

To measure the allocation layer's contribution, the strict-clipping baseline is
kept under `baselines/strict_clipping/`. It uses the same corrected transition
clipping without proportional allocation:

```bash
sbatch baselines/strict_clipping/run_rnncell_strict.sh
```

Its artifacts are written to:

```text
outputs/rnncell_strict_<job_id>/
```

## Experiment Log

Models, datasets, and raw logs remain local because they are large. Record the
compact result summary in [EXPERIMENTS.md](EXPERIMENTS.md) after each run.

Generate a Markdown row from an output directory:

```bash
python summarize_experiment.py outputs/rnncell_<job_id>
```
