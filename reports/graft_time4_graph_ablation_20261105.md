# Time-4 graph-specific ablation, November 5, 2026

**Status:** Completed. **Scope:** Exploratory, single bug, not independent held-out evaluation.

User-reported execution output from AIScalar run:
`results/graft_time4_graph_ablation_20261105_170855/run`

- Qwen2.5 7B, Defects4J Time-4, 23 candidate methods, 98 graph nodes.
- 24 tests passed in 1.13 s, preflight ready, all policy arms completed.
- Four policies share one provider journal and model requests.
- Actual physical provider total: 28 new calls and 14,825 input+output tokens, no unknown provider usage.

| Policy | Rank | Top-5 | MRR | Logical tokens | Observed stages |
|---|---:|---:|---:|---:|---|
| Fixed L1 | 6 | 0 | 0.1667 | 10,150 | Fixed baseline |
| L1 anchor (failure-name only) | 6 | 0 | 0.1667 | 10,150 | 23 sketches |
| L1 graph (failure-name plus graph) | 6 | 0 | 0.1667 | 12,404 | 23 sketches, 2 graph inspections |
| GRAFT adaptive | 6 | 0 | 0.1667 | 12,571 | 23 sketches, 2 graph inspections, 1 Full inspection |

## Interpretation

Graph refinement did **not** improve the correct-method rank on this run. Relative to L1 anchor, L1 graph used 2,254 more logical tokens (~22.2%). GRAFT adaptive used 2,421 more logical tokens (~23.9%) than L1 anchor, without improving rank. Do not confuse these per-policy logical figures with physical provider spending, because identical initial requests are reused through the journal.

This run did not reproduce the anchored rank-5 result reported in the earlier separate Time-4 comparison. The cause is not yet established. It could involve variation in Qwen output, evidence/configuration differences, or another factor, and must be diagnosed from both runs' immutable inputs and saved paid responses. Avoid attributing the difference to a specific cause without the comparison.

Both Time-4 analyses are exploratory. This bug was selected by gold-aware offline screening; it must not be presented as held-out evidence.

## Recommended next checks (offline, zero inference)

1. Compare `manifest.json` (input hashes, source revision, provider metadata, full configuration) between the earlier anchored Time-4 run and this ablation.
2. Compare exact sketch `raw_response` / paid request hashes for matching candidate IDs.
3. Report graph evidence selection and stage histories for the correct method. Check whether it received graph or Full inspection, and whether scores were applied to the ranking.
4. If rankings differ because of model output variation, rerun at least several paired seeds with a stated call/token cap; do not cherry-pick a favorable run.
5. Repeat on additional predeclared bugs before drawing general conclusions.

**Artifact note:** The full output directory is currently on AIScalar. This document is a transcript-based summary; it does not mean the raw artifact has been uploaded to GitHub.
