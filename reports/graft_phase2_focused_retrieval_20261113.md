# GRAFT-FL Phase 2, source-focused graph retrieval

**Date:** 2026-11-13  
**Branch:** `graft-fl-aiscalar-20261113`  
**Status:** Code and offline/fake tests committed; AIScalar checks and independent real Qwen runs **not yet executed** for Phase 2.

## Why Phase 2 is needed

From the saved Time-4 Phase-1 experiment, the suspicion-first graph policy successfully inspected the correct method `org.joda.time.Partial.with(DateTimeFieldType,int)`, but its graph prompt provided only a method signature and a few fragments. It omitted the central `if (index == -1)` branch and the body of `AbstractPartial.indexOf(DateTimeFieldType)`. Qwen returned uncertainty 1 and reported incomplete AST evidence. The graph-downgrade guard rejected that response. The observed rank-1 gain came from **fusion**, not graph retrieval.

## Implementation

- `rave_fl/graft/selection.py`: new opt-in `graft-focused-v1` compact renderer, grouping exact immutable buggy-revision code fragments by function, file path and line ranges. Reuses the same graph integrity checks and the same provider message/system prompt. The historical `graft-evidence-v1` format is not changed.
- `rave_fl/graft/focused.py`: new gold-free `focus_select()`. It extracts actual method-call names from an L1 `missing_evidence` string, verifies calls in the target source, resolves only uniquely named available candidate methods, adds the **callee's method signature and all available AST code units** as one bundle, and prioritizes target code that uses the returned variable. Unknown or ambiguous requested calls are reported as `focused_evidence_incomplete`, so no misleading graph verdict is purchased.
- `rave_fl/graft/controller.py`: new `l1_graph_focused` option, using the same highest-L1-suspicion selection as `l1_graph_suspicion`, with the new source-focused evidence package. The historical policies remain unchanged.
- `rave_fl/graft/runner.py` and report script: register the new policy.
- `scripts/audit_graft_focused_offline.py`: zero-provider-call audit of exact returned source fragments, selected target and evidence budget.
- `scripts/prepare_graft_focused_independent.py`: paired exploratory configuration for Time-1, Lang-1 and Math-1, which were explored previously and **are not a clean held-out set**.
- `tests/test_graft_focused.py`: Time-4 source-based budget regression, failure on unresolved code hints, reproducibility, frozen BM25 schema, and synthetic pipeline tests.

## Time-4 budget pre-check

The saved Time-4 graph has the following exact source units available:

| Source | Lines | Target evidence |
|---|---|---|
| Partial.with | 412-474 | Target signature |
| Partial.with | 427-429 | Null guard |
| Partial.with | 430 | Call to `indexOf(fieldType)` |
| Partial.with | 431-466 | Large `if (index == -1)` branch |
| AbstractPartial.indexOf | 163-176 | Callee signature |
| AbstractPartial.indexOf | 170-174 | Callee loop and comparison |
| AbstractPartial.indexOf | 175 | Callee return |

A byte-count calculation equivalent to the new grouped JSON format gives **about 4,053 units / 4,096 limit** with these fragments (including the unchanged system message and 256-unit proxy overhead). This is a **preliminary code-based estimate**, not a verified AIScalar Python test or new Qwen evaluation. The new unit test asserts exact feasibility and original-source equality.

Even if this fits, some other target method statements are intentionally omitted. The diagnostics include `omitted_target_code_ids` and `semantic_completeness_guaranteed=false`. This does not claim a complete method or runtime dependency proof.

## Step 1: Run the zero-inference tests on AIScalar

```bash
cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
conda activate rave-fl
git fetch origin graft-fl-aiscalar-20261113
git merge --ff-only FETCH_HEAD
bash scripts/aiscalar_graft_focus_offline.sh
```

The script runs the focused and historical regression tests and audits the saved Time-4 graph using exactly 4,096 input units. Only if all tests and budget assertions pass does it commit and push its report with the configured GitHub author email. It sends **zero real Qwen calls**.

## Step 2: Predeclared additional bugs, AFTER successful offline validation

```bash
bash scripts/aiscalar_graft_focus_independent.sh
```

This runner checks independent-of-Time-4 exploratory bugs `Time-1`, `Lang-1` and `Math-1` and refuses real inference if preflight is not ready for all three. It compares `l1_anchor`, `l1_graph_suspicion`, and `l1_graph_focused` with the same RRF or fixed new fusion; includes a fixed-L1 reference; and uses an explicit limit of 115 new calls and 180,000 physical provider tokens.

This is **not** held-out generalization evidence because these three bugs were studied before. For publication-level claims, select and freeze a separate unseen test cohort before studying its outcomes.

## Limitations and integrity

- A source-level invocation is not proof of runtime execution. Ambiguous names remain unresolved rather than inventing call-graph edges.
- The compact evidence format changes packaging overhead, so compare the measured real provider tokens across paired policies, not only logical tokens or ranks.
- A graph response is still just a Qwen judgment. Retain original provenance, incomplete-coverage diagnostics and the unsupported-downgrade guard.
- The code, tests and runners are committed, but **Phase 2 has not been run on AIScalar yet**. Do not claim an observed Phase-2 accuracy gain.
