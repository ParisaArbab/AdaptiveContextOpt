# GRAFT-FL Phase 1 live audit, Time-4, November 13 2026

## Experimental status
Source: `results/graft_phase1_live_20261113_174425` on branch `graft-fl-aiscalar-20261113`. AIScalar output: **33 passed** in 1.15 seconds, **10 completed arms**, **30 physical new Qwen provider calls**, **16,970 provider tokens**, **usage_unknown=false**. User pushed experiment commit `cbdcc30`. Per-policy logical tokens include reuse and must **not** be summed as provider cost.

## Main result

| Graph policy | RRF gold rank | Suspicion-rank blend gold rank |
|---|---:|---:|
| fixed_l1 | 6 | 1 |
| l1_anchor | 6 | 1 |
| l1_graph | 6 | 1 |
| l1_graph_suspicion | 6 | 1 |
| graft_adaptive | 6 | 1 |

**Conclusion:** The rank improvement is entirely attributable to the new fusion formula on this development bug. The new graph-selection policy did not improve the measured rank for either fusion.

## Graph selection actually improved targeting

In contrast to the original `l1_graph` policy, `l1_graph_suspicion` **did select the gold method** for its first graph call:

`org.joda.time.Partial.with(DateTimeFieldType,int)`.

The initial L1 observation for this method had suspicion **0.9**, uncertainty **0.1**. Its L1 explanation was `Method does not handle null DateTimeFieldType properly`, with requested missing evidence `null check in indexOf() method`.

**Important evidence caveat:** The actual supplied source for `Partial.with(...)` contains an explicit `if (fieldType == null) throw new IllegalArgumentException(...)` check. Therefore the model's L1 explanation is not reliable proof of the bug's mechanism, even though the method is the gold bug label. The observed test calls `with(DateTimeFieldType.clockhourOfDay(), 6)` and then `fail()`, which is a different scenario from a null argument.

## Precise next failure: graph packaging and retrieval

For the correct method, the graph request contained:
- the failing test fragment;
- the `Partial.with(...)` method signature, the null guard, and `int index = indexOf(fieldType)`;
- the **signature** of `AbstractPartial.indexOf(DateTimeFieldType)`, but **not its body**;
- unrelated excerpts from `BasicDayOfMonthDateTimeField.getMaximumValue(...)`.

The graph omitted the much larger `if (index == -1) { ... }` control-flow block, and the actual body of `AbstractPartial.indexOf(...)`. Both bodies existed in the source/graph; they were not included in the selected evidence. The selection diagnostics list many `budget_rejected_ids`, while `semantic_completeness_guaranteed=false`, despite the syntactic support-closure check passing.

Qwen's paid graph response for the correct method was suspicion **0**, uncertainty **1**, explanation `No direct evidence links to fault` and missing evidence `AST not fully provided`. The **unsupported_graph_downgrade guard correctly marked this result not applied to ranking**, retaining the stronger original L1 observation. The second graph call inspected `Partial.getFieldType(int)`, and similarly reported insufficient evidence.

## Why this matters

**Candidate selection is no longer the primary failure for `l1_graph_suspicion`.** It found and inspected the correct method. The evidence package did not give Qwen enough method/dependency code to assess the failing test. The guard prevented a lower-quality graph response from reducing the final ranking.

The current result does not establish that a complete graph would produce a better answer, that Qwen's 0.9 suspicion was correct, or that the suspicion-rank blend generalizes beyond Time-4. All are hypotheses needing independent bugs.

## Next controlled research tasks

1. Add a **new opt-in, target-focused graph retrieval policy**, retaining all existing baselines. Start with mandatory failure and target signature, then prioritize target method control-flow/condition fragments and the **actual body** of named missing-evidence callees such as `indexOf()`. Use only observed input evidence and real graph/source nodes, never gold labels.
2. When budget is too small to include the referenced method or decisive target branch, record `evidence_incomplete` explicitly and avoid treating the resulting model answer as a full graph verdict. Do not hallucinate or fabricate execution edges.
3. Offline-test the rendered graph package on Time-4 **without any Qwen calls**. Assert that relevant target body and `indexOf` body are present within the same 4,096-unit input budget, or honestly report infeasibility.
4. After code and tests pass, run identical-budget, paired arms on independent bugs. Report validity, rank/Top-1/Top-5/MRR, physical provider tokens, and refusal/incompletion rates, not only chosen showcase bug.

**Research integrity:** Time-4 remains development data, not held-out verification. The observed physical count of 30 calls / 16,970 tokens is shared across all arms and is not a per-arm cost estimate.
