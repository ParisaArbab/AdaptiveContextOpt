# GRAFT-FL Phase 1: fusion and graph-action ablations

**Prepared November 13, 2026.** Experimental changes are isolated, opt-in and uncalibrated. The original policies and committed AIScalar runs remain unchanged.

## Implemented in code

1. `rave_fl/eval/fusion.py`: new `suspicion_rank_blend`, fixed 0.70 valid L1 suspicion plus 0.30 normalized original ordinal rank. The RRF, prior-aware, consistent-score and historical fusion algorithms are unchanged. Invalid/missing observations fall back to original rank. The output run metadata records these fixed weights.
2. `rave_fl/graft/controller.py`: new `l1_graph_suspicion` policy. Initial sketch observations and evidence are unchanged. It selects the eligible graph target with highest **observed** suspicion, breaking ties by lower uncertainty, lower predicted token cost and original rank. No gold labels enter this policy. The existing `l1_graph` and `graft_adaptive` policies retain their historical selection decisions.
3. `rave_fl/graft/runner.py`: registers the opt-in graph policy.
4. `scripts/audit_graft_fusion_offline.py`: reconstructs historical saved RRF rankings, then compares frozen L1 re-rankings. It uses ground truth only after ranking for offline metrics. **Zero inference calls.**
5. Added regression tests for original RRF, deterministic and invalid-score handling, graph action selection and zero-call offline evaluation.

## Time-4 offline development result

On saved Qwen2.5 7B Time-4 L1 responses, initial rank 10 and valid suspicion 0.9 for `org.joda.time.Partial.with(DateTimeFieldType,int)`:

| Fusion | Correct method rank | Top-5 | MRR |
|---|---:|---:|---:|
| RRF, historical | 6 | 0 | 0.1667 |
| Consistent score, already supported | 1 | 1 | 1.0000 |
| Prior aware, already supported | 1 | 1 | 1.0000 |
| Suspicion rank blend, new | 1 | 1 | 1.0000 |

Saved machine-readable output: `reports/graft_time4_frozen_fusion_replay_20261113.json`.

**This is NOT fresh-model validation.** No new model calls were made to obtain these ranks. Since Time-4 is a development bug, these figures cannot be used to choose and claim a generalizable policy.

## AIScalar verification, no real provider calls

From the existing branch/worktree `~/RAVE_FL_Research/GRAFT_FL_New_Run`:

```bash
conda activate rave-fl
git fetch origin graft-fl-aiscalar-20261113
git merge --ff-only FETCH_HEAD
python -m pytest -q tests/test_graft_fusion_ablation.py tests/test_graft_graph_suspicion.py tests/test_graft_controller.py tests/test_graft_pipeline.py
python -m scripts.audit_graft_fusion_offline \
  results/graft_time4_graph_ablation_20261113_143016/run \
  --output results/graft_time4_fusion_replay_local_20261113.json
```

Stop if tests fail. This sequence makes **zero real Qwen calls**.

## Next phases, not yet implemented or validated

- Run paired, predeclared fusion and graph-selection comparisons on independent bugs. Keep the original baselines. Log physical tokens separately from logical per-policy totals.
- Implement AST-backed retrieval of specifically requested missing code such as `indexOf()`, with explicit unresolved-edge outcomes.
- Study a conservative evidence-usefulness stopping rule without suppressing promising or uncertain candidates.
- Calibrate weights on separate training/calibration bugs and report held-out test data. Do not use Time-4 to tune a winning weight.

**Current status:** changes committed to GitHub. Tests on AIScalar have not yet been run for this Phase 1 revision; no new live Qwen outcomes are claimed.
