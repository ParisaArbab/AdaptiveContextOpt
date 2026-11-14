# GRAFT-FL fusion and graph-selection improvement plan

**Date:** 2026-11-13  
**Status:** Offline diagnosis and predeclared design proposal. No new LLM calls, no new measured policy performance.

## New offline finding

Read the saved Time-4 `l1_anchor` per-method observations from:
`results/graft_time4_graph_ablation_20261113_143016/run/instances/Time-4_0d6a675374/l1_anchor__rrf_948fc16d89.json`.

- Gold method for evaluation only: `org.joda.time.Partial.with(DateTimeFieldType,int)`.
- Gold method's valid Qwen L1 suspicion: **0.9** with uncertainty **0.1**.
- All other 22 candidates have suspicion **at most 0.3**.
- Current RRF fused score for gold: **0.030679156908665108**, final rank **6**.
- Among the five candidates above the gold, the first had suspicion 0.3, the rest were 0.1.
- **Retrospective counterfactual:** Sorting candidates by descending stored L1 suspicion, with original order for ties, puts the gold method at **rank 1**. This is not a new experimental run, a generalization result, or a claim of valid independent performance.

The current RRF fusion weights relative *ranks*:
`1/(k + original_rank) + 1/(k + 1 + count(other_suspicion > own_suspicion))`
for `k = 60`. This can retain substantial original-rank advantage even when a method has a much higher raw suspicion score.

## Selection failure confirmed separately

The saved `l1_graph` route chose methods via original candidate rank rather than its computed utility score. Both graph inspections returned suspicion 0.0 plus inadequate evidence, and the unsupported-downgrade guard correctly declined to apply them. The actual gold method received no graph or Full stage. GRAFT adaptive used an uncalibrated cost/uncertainty utility that also missed the gold.

## Controlled improvement sequence

1. **Offline fusion ablation first.** Compare frozen RRF, suspicion-first and predeclared blended approaches using saved responses. Keep all labels strictly offline, and choose one fusion configuration on separate calibration bugs, not Time-4.
2. **New graph-action policy as its own named treatment.** Rank eligible graph requests by a gold-free, auditable score using suspicion, genuine unresolved evidence (ignore strings like `none`), distance to the top-k cutoff, graph support, and predicted cost. Keep old `l1_graph` as the unmodified control.
3. **Missing-evidence-guided retrieval.** When a sketch identifies a missing code call or concept such as `indexOf()`, use AST-backed source lookup and real graph edges to retrieve those fragments. If resolution fails, record it explicitly rather than inventing links.
4. **Budget-sensitive stop decision.** Abstain from additional graph requests when the relevant evidence is missing or the expected improvement is low; report time and physical provider-token consumption.
5. **Paired held-out verification.** Freeze parameters before fresh runs. Use independent Defects4J bugs, report MRR, Top-1/5, rank, invalid responses, exact-cache reuse and physical provider tokens.

### Scientific limits

The Time-4 case was chosen after examining gold-aware offline results, so it is **development data**. A single successful counterfactual cannot justify selecting score weights or reporting an accuracy gain for the new policy. All new treatment arms must use identical candidate lists and evidence without accessing gold at inference time.
