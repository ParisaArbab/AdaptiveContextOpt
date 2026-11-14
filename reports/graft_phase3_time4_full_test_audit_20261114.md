# GRAFT-FL: full official trigger-test Time-4 live audit

**Audit date:** 2026-11-14  
**Run:** `results/graft_phase3_time4_full_test_20261113_224352`  
**Run commit:** `939488c`  
**Research branch:** `graft-fl-aiscalar-20261113`  
**Cohort:** Time-4, development case repeatedly inspected; **not held out**.

## Verification and input provenance

- New regression suite **9 passed**, runner regression suite **35 passed**.
- The source-preparation code extracted **1,040 bytes** of combined evidence, comprising the authentic full official Defects4J `TestPartial_Basics.testWith3()` triggering method and the old abbreviated trigger description.
- Trigger-test **source** was pinned to the fixed-revision *test file* at `3ba9ba799b3261b7332a467a88be142c83b298fd`; the production checkout was pinned to original buggy revision `bcb044669b4d1f8d334861ccbd169924d6ef3b54`.
- The resulting provenance records SHA-256 of the test source, extracted method, and combined evidence. **No new failing-test execution took place**, and the evidence remains classified `trigger_description` rather than `observed_failure`.
- Actual physical Qwen usage across **all arms combined**: **27 new calls**, **18,913 provider input/output tokens**, zero unknown/reserved tokens. Per-arm logical tokens include reused requests and are **not** standalone physical costs.

## Reported ranks, one development bug

| Policy | RRF gold method rank | Suspicion-rank blend gold method rank | Logical tokens | Arm completion |
|---|---:|---:|---:|---|
| fixed_l1 | 4 | 1 | 15,855 | **incomplete** |
| l1_anchor | 4 | 1 | 15,855 | **incomplete** |
| l1_graph_focused | 4 | 1 | 17,347 | completed |
| l1_graph_test_aware_nostop | 4 | 1 | 17,421 | completed |
| l1_graph_test_aware | 4 | 1 | 17,421 | completed |

Relative to the previous abbreviated-trigger experiment, RRF's **displayed** gold rank moved from 6 to 4 for **both** graph and L1-only policies. The model and prompt input changed, so this is a change in the input condition, **not evidence that graph refinement specifically helped**. With suspicion-rank blend, the displayed gold rank remained 1 for all.

**Critical completion caveat:** The execution summary records the whole Time-4 instance **incomplete**, with 8 completed and 4 incomplete policy/fusion arms. The headline summary treats incomplete arms as noncompleted rather than full successful results, even though rank diagnostics are available.

## Exact cause of baseline incompletion

I inspected `run/instances/Time-4_0d6a675374/` and counted the final state values:

- `fixed_l1`: **23** candidates, **22** valid initial L1 observations, **1 invalid**.
- `l1_anchor`: **23** candidates, **22** valid initial L1 observations, **1 invalid**.
- Invalid L1 candidate: `org.joda.time.field.AbstractPartialFieldProperty.compareTo(ReadablePartial)`.
- `fixed_l1` stop reason: `full_repair_limit`, `l1_anchor`: `l1_anchor_only`. These labels are stopping reasons, **not actual program fixes**.
- `l1_graph_focused`: **23** valid final observations after two graph calls, status completed.
- The graph arms inspected the invalid candidate after the primary `Partial.with` inspection, restoring a valid structured observation. This can improve **response coverage**, though not the first relevant rank.
- `complete()` in `rave_fl/graft/runner.py` explicitly requires all 23 `observation.valid` values; thus the incomplete baseline statuses are explained by response validity, not by missing final ranking files.

**Do not call L1-only a fully completed baseline.** Report its rank as a partial-run diagnostic until a preregistered, consistent invalid-response handling rule is implemented and independently evaluated.

## Full-test graph inspection, still no gold-rank improvement

Graph actions in `l1_graph_test_aware`:

1. `org.joda.time.Partial.with(DateTimeFieldType,int)`:
   - L1: suspicion **0.6**, uncertainty **0.4**, reason suggesting a missing null check; requested `null check implementation`.
   - Graph: suspicion **0**, uncertainty **1**, reason **"Test expected exception, no fault."**, missing evidence **"Last line not shown fully"**.
   - Graph response **not applied**: `unsupported_graph_downgrade`; initial L1 score preserved.
   - Graph test-signal: `possible_expected_exception`, `catch_present`, catch type `IllegalArgumentException`, provenance `supplied_text:trigger_description`, `full_test_verified=false`.
   - Graph evidence package: **2,921/4,096** proxy units; `semantic_completeness_guaranteed=false`.
2. `org.joda.time.field.AbstractPartialFieldProperty.compareTo(ReadablePartial)`: graph suspicion **0**, uncertainty **1**, reason **"Stack trace missing; cannot verify."**; this provides a valid structured observation for the previously invalid L1 candidate.

`l1_graph_focused` also made two graph calls and returned high-uncertainty verdicts, including **"Test passes now; no defect"** on the gold method. This sentence is a model explanation, **not a verified test-execution outcome**. The input was a test description and the script performed **no tests**. Do not repeat this explanation as a fact.

The test-aware **with-stop** and **no-stop** policies had identical ranks and each made two graph calls; no observed stopping benefit in Time-4.

## Interpretation and next step

**Supported conclusions:** The authentic complete trigger changes the input condition and yields RRF gold rank 4 instead of 6 in the displayed diagnostics; the graph policies completed all 23 response states, while L1-only arms had 22/23 valid. The graph policies did **not** improve gold rank within either fusion. The test-aware selector did **not** reduce graph calls and the model remained uncertain.

**Unsupported conclusions:** Graph-specific improvement in localization accuracy, a calibrated improvement in uncertainty, correct bug-reasoning explanations, a general benefit of complete tests, or held-out performance.

**Next priority:** Validate authentic data preparation for the preregistered Time-1, Lang-1, Math-1 exploratory cohort, and run paired policy comparisons with identical real evidence/candidate pools. First run `bash scripts/aiscalar_graft_phase3_prepare.sh inspect`. The previous environment returned `defects4j=null`, so collection requires a valid official Defects4J executable and genuine checkout/test data. Do not invent missing runtime evidence or substitute different bugs.

Also evaluate an explicit cross-policy strategy for invalid L1 responses, reporting **valid-observation coverage** as a separate metric. Do not retroactively mark incomplete arms completed or claim rank improvements from uncompleted policies.

This audit was written from the user's AIScalar log and committed experiment artifacts, with **zero** new model calls.
