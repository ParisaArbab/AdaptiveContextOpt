# Frozen EGAV validation, 10 additional development bugs

The five original pilot bugs helped design EGAV, so their accuracy gain is not an
independent test. This launcher evaluates **new batches 01 and 02** from the
already frozen GRAFT 100-bug cohort, with the *same* EGAV selection weights,
two-Full cap, Qwen2.5:7b model, 24-candidate pool limit and 60,000 logical
token budget. It does not tune the weights on the new bugs.

## Exact independent bug IDs

- Batch 01: Chart-1, Closure-128, Lang-35, Math-57, Time-14.
- Batch 02: Chart-12, Closure-32, Lang-6, Math-56, Time-6.

These IDs come from the checked-in `experiment/config.json` in each batch.
The program reads the IDs from the config and validates five non-overlapping
bugs per batch. **Do not** use `run_aiscalar_sample.sh` for these batches:
its `study` command still defaults to the five original pilot bugs.

## AIScalar commands

```bash
conda activate rave-fl
cd ~/RAVE_FL_Research/MethodDiscovery
git fetch origin refs/heads/research/egav-validation-20261119:refs/remotes/origin/research/egav-validation-20261119
git merge --ff-only refs/remotes/origin/research/egav-validation-20261119

# Preflight, no model calls. Requires existing immutable AIScalar checkouts.
python research_method_agent/egav_validation.py preflight

# Collect independent real observations for five bugs at a time.
python research_method_agent/egav_validation.py run-01 --allow-inference \
  2>&1 | tee egav_dev_batch01.log

# After batch 01 succeeds:
python research_method_agent/egav_validation.py run-02 --allow-inference \
  2>&1 | tee egav_dev_batch02.log

# One paired report across 10 distinct bugs, no new Qwen calls.
python research_method_agent/egav_validation.py report
python research_method_agent/egav_validation.py status
```

If the collector reaches its call/token cap, `run-01` or `run-02`
can resume the existing observations with a higher explicit cap, after
reviewing the cost. Defaults are 450 new calls and 800,000 physical tokens
*per batch*, including attempts already saved.

The launcher checks Git blob SHA hashes of
`research_method_agent/research_method_agent.py` and
`research_method_agent/egav_policies.py` before running, protecting
the **frozen policy and evaluation implementation** from accidental changes.
Changes to these files require a new study version rather than silently
mixing methods. Do not change model, pool size, budget, or bug cohort
mid-collection.

## Outputs and interpretation

Each batch writes `results/egav_validation_10bugs/batch_XX/`, including
manifest, provider observations, and `evaluation/results.json`.
The combined report, after both batches complete, is saved as:

- `results/egav_validation_10bugs/combined_summary.json`
- `results/egav_validation_10bugs/combined_report.md`

The aggregation compares arms **only on complete paired bugs**. Read the
paired count before interpreting the means. Physical token accounting
includes all separately collected counterfactual calls, whereas logical
policy tokens describe the replayed policy only.

This is new **development** evidence, not an untouched final test, and
modified-class file proxy MRR is not method-level bug localization.
After examining the 10-bug findings, select any future policy changes on
development data only, then lock a new version for a separate held-out
test group of at least 30 bugs.
