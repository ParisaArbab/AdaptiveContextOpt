# GRAFT-FL three-bug preflight blocker, November 13 2026

## Observed AIScalar diagnosis

The user executed the current independent-bug evidence diagnostics on the AIScalar `rave-fl` environment and reported `3 passed` for the evidence-diagnosis regression tests.

`python -m scripts.diagnose_graft_independent_evidence --manifest configs/rave_independent_smoke.json --output results/graft_independent_evidence_diagnosis.json` reported:

| Bug | FlexFL bug report | FlexFL trigger test | Source checkout |
|---|---|---|---|
| Time-1 | NO FILE | NO FILE | nonexistent |
| Lang-1 | NO FILE | NO FILE | nonexistent |
| Math-1 | NO FILE | NO FILE | nonexistent |

A previous `graft_focused_independent_20261113_201417` preflight stopped with `dataset_error=Empty debugging evidence from flexfl_original`, `preparation_error=Empty debugging evidence from flexfl_original`, `ready=false`, `candidate_preparations=[]`, exit code 3. Neither preflight nor diagnostic calls Qwen; no independent-bug accuracy results were obtained.

## Exact scope of problem

The manifest is `configs/rave_independent_smoke.json`. Its `cases` reference `references/FlexFL_Time25` and `data/rave_fl_checkouts/<bug>` for the three named bugs. These specific paths contain neither the required test text nor the buggy checkouts on the user's current AIScalar working tree. This does not prove that all genuine benchmark data is absent elsewhere on the host.

The `rave_fl.eval.adapters.normalize` fallback calls `rave_fl.compare_expanded.load_context`, which concatenates `wp1.flexfl_data.read_bug_report` and `read_trigger_test`. If both return empty content, it raises `ValueError("Empty debugging evidence from flexfl_original")`. This failure is outside the focused retrieval code.

`defects4j` executable was also not found in the observed preflight; that is a separate environment limitation if real test runs are needed.

## Safe recovery workflow

1. Search existing authorized AIScalar research worktrees and archived logs for the exact named **buggy revisions**, original FlexFL bug reports and trigger tests. Verify that retrieved inputs are real observed/trigger text and record their path, SHA256 and evidence type.
2. If the records are genuinely unavailable, obtain the official Defects4J bug checkouts and reproducible failing test output under a separate **new** data-preparation workspace. Do not change or overwrite existing checkouts. Installing/running Defects4J may be needed to generate authentic observations; it was not present in PATH during preflight.
3. Create a **new versioned manifest** pointing to verified inputs, not modifying the frozen 2026-10-22 manifest. Keep the bug IDs and original selection rule unchanged. Declare where any tests actually ran. Distinguish bug reports/trigger descriptions from observed runtime failures.
4. Require ready=true and three independent candidate preparation successes before any real Qwen run. If still unavailable, report incomplete input availability rather than silently replacing bugs, candidate data, or source revision.

## Work that can proceed now

Time-4 already had valid preparation in a committed Phase-1 live experiment. A separate opt-in Phase-2 development smoke has been added as `scripts/prepare_graft_focused_time4_live.py` and `scripts/aiscalar_graft_focus_time4_live.sh`, with unchanged old controls, the same 4,096 byte-proxy evidence budget and bounded provider caps. It compares new compact targeted evidence against the old selection on an **already studied development bug**, not on a fresh independent or held-out test.

**Research integrity:** No claim about Phase-2 live accuracy or independent generalization is justified until real runs complete. Do not fabricate observations or convert missing data into synthetic evidence.
