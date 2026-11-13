# Phase 1 AIScalar verification, 2026-11-13

**Status:** User-reported real AIScalar test execution completed successfully. This file records tests and offline re-ranking only, not new Qwen inference.

## Environment and source

- Branch: `graft-fl-aiscalar-20261113`
- Tested checkout after fast-forward from `56a09f1` to `fa0342c`
- Environment: `rave-fl` on AIScalar
- Command: `python -m pytest -q tests/test_graft_fusion_ablation.py tests/test_graft_graph_suspicion.py tests/test_graft_controller.py tests/test_graft_pipeline.py`
- Result: **31 passed in 1.14 seconds**

## Saved-response replay

Command:

```bash
python -m scripts.audit_graft_fusion_offline \
  results/graft_time4_graph_ablation_20261113_143016/run \
  --output results/graft_time4_fusion_replay_local_20261113.json
```

Output:

```text
Time-4: 23 candidates, RRF reproduced=True
rrf                      rank=6 Top5=0 MRR=0.16666666666666666
prior_aware              rank=1 Top5=1 MRR=1.0
consistent_score         rank=1 Top5=1 MRR=1.0
suspicion_rank_blend     rank=1 Top5=1 MRR=1.0
Report saved: results/graft_time4_fusion_replay_local_20261113.json
```

No model inference was done during these tests or replay.

## Interpretation

The alternative score functions place the one true buggy method first **on the previously saved Time-4 L1 responses**. These are retrospective calculations using a development example, not new Qwen outcomes and not independent held-out test results. The graph-selection policy passed fixture tests but was **not yet run on real Qwen** at this stage. The suspicion blend has not shown an advantage over the pre-existing `prior_aware` and `consistent_score` methods on Time-4.

The raw offline replay output exists on AIScalar at `results/graft_time4_fusion_replay_local_20261113.json`. It was not uploaded in this GitHub update; the original saved-response replay JSON is under `reports/graft_time4_frozen_fusion_replay_20261113.json`.

## Next

Use a fresh, bounded, multi-arm experimental run comparing new fusion and graph selection against frozen RRF and legacy graph policies. Archive all raw provider responses, config, metrics and failures; compare real provider usage without double-counting reused calls.
