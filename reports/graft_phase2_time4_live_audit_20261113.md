# GRAFT-FL Phase 2 live results audit: Time-4

**Date:** 2026-11-13  
**Source run:** `results/graft_phase2_time4_live_20261113_202949`  
**Original experimental commit:** `2dd113d`  
**Status:** Completed live development experiment; this audit uses saved data and adds **zero** Qwen calls.

## Verification and accounting

- 41 local tests passed in 1.12 seconds.
- Time-4 preflight was ready, run completed successfully.
- Shared physical provider usage: **27 new calls**, **13,664 known input+output provider tokens**, **usage_unknown=false**.
- These provider totals refer to the *entire shared* run, **not** the independent cost of any policy; cache reuse prevents simple per-arm attribution.

## Accuracy and per-policy logical tokens

| Policy | RRF gold rank | 70/30 suspicion blend gold rank | Logical tokens |
|---|---:|---:|---:|
| fixed_l1 | 6 | 1 | 10,150 |
| l1_anchor | 6 | 1 | 10,150 |
| l1_graph_suspicion | 6 | 1 | 12,348 |
| l1_graph_focused | 6 | 1 | 11,466 |

`l1_graph_focused` saved **882 logical tokens** relative to the original suspicion-based graph policy, or **7.14%**, with two graph calls each. The difference is **not** a measured standalone physical provider saving and the ranks did **not** improve.

## Exact Qwen judgments for gold method

**Method:** `org.joda.time.Partial.with(DateTimeFieldType,int)`

- Shared initial L1 sketch: suspicion 0.9, uncertainty 0.1, reason `Method does not handle null DateTimeFieldType properly.`, missing evidence `null check in indexOf() method`.
- Original `l1_graph_suspicion` graph verdict: suspicion **0**, uncertainty **1**, reason `No direct evidence links to fault.`, missing evidence `Stack trace missing`.
- New `l1_graph_focused` graph verdict: suspicion **0.2**, uncertainty **0.8**, reason `Index check might fail`, missing evidence `No failing case found`.
- **Both graph verdicts were recorded but not applied** due to `unsupported_graph_downgrade`. Consequently the high original L1 suspicion remains the final effective score.

This establishes a difference in the model's *responses*, not improved rank or validated belief calibration. The stronger initial L1 reason is not supported as a null-handling explanation by the source: `Partial.with` has an explicit `fieldType == null` guard.

## Focused evidence quality

The new `focused_compact_v1` package for the gold method was **ready**, exactly **4096/4096 byte-proxy units**, and included:
- failing test excerpt `testWith3()` invoking `test.with(DateTimeFieldType.clockhourOfDay(), 6)` and then `fail()`;
- target method signature and null guard;
- `int index = indexOf(fieldType)`;
- the major `if (index == -1)` branch;
- full available source units of `AbstractPartial.indexOf(DateTimeFieldType)`.

Its diagnostics correctly recorded `semantic_completeness_guaranteed=false`. It omitted three target nodes: the subsequent `if (value == getValue(index))` branch, `newValues = getField(index).set(this, index, newValues, value)`, and `return new Partial(this, newValues)`. The target's other post-branch line `int[] newValues = getValues()` was included.

The original graph package for this method used **4066/4096 units**, but prioritized other method fragments and lacked the full relevant callee/source bodies. The new selector improved source specificity under the same budget, but did not include the entire target code.

## Next hypothesis and controlled follow-up

The observed failure excerpt shows `fail()` after `with(...)`, consistent with the test expecting an exception. It does **not** establish the full intended exception or exact faulty operation, because the displayed trigger excerpt is abbreviated. The earlier L1 null-handling explanation is suspect for this test.

Do **not** tune retrieval by consulting the gold method or fixed Time-4 outcome. Instead:

1. Extract expectation structure from the **observed failure and real test source**, when available. Examples: `fail()` in a try block, expected exception in a catch clause, assertion expected vs actual.
2. Rank branch fragments based on the test's API call and reachable *source-level* branches. Include post-branch setter/return paths if the evidence points there, rather than always selecting the first large `if`.
3. Preserve explicit `omitted_target_code_ids`, uncertainty, unresolved call targets and guard decisions.
4. Run budget-matched ablations on independently prepared bugs once genuine failing evidence, FlexFL candidates and bug-revision checkouts are available. The earlier Time-1/Lang-1/Math-1 cohort cannot currently run because its inputs are missing.

**Scientific conclusion:** Improved source specificity yielded a more specific but still uncertain Qwen answer, 7.14% fewer *logical* tokens for the graph policy, and **no observed rank improvement** on the Time-4 development bug. Fusion, not graph retrieval, remains responsible for the Top-1 result in this experiment.
