# GRAFT-FL Phase 3: test-aware evidence and conservative stopping

**Prepared:** November 2026  
**GitHub branch:** `graft-fl-aiscalar-20261113`  
**Status:** Implementation, tests and runners committed. Phase-3 AIScalar tests and live Qwen inference **not yet executed**.

## Motivation, directly observed in Phase 2

On Defects4J Time-4, `l1_graph_focused` retrieved the target code and `AbstractPartial.indexOf` but still produced uncertainty 0.8 and suspicion 0.2. The protective graph guard retained initial L1 suspicion 0.9. Both original RRF and new fusion maintained the same final buggy method rank as without graph. A narrow test excerpt shows:

```java
Partial test = createHourMinPartial();
try {
    test.with(DateTimeFieldType.clockhourOfDay(), 6);
    fail();
```

There is **no visible catch block** in the saved excerpt. This is **trigger-description evidence, not a complete observed test execution**. A `fail()` after the API call suggests an expected exception, but the excerpt does not establish which exception is expected. The initial Qwen explanation about missing null handling was contradicted by the target's actual null guard. This motivates test-aware evidence selection without treating either model explanation as ground truth.

## New opt-in policies

- `l1_graph_test_aware_nostop`: prioritizes AST-backed target code units based on supplied failure clues: likely exception, assertion, or unknown. It may prioritize `throw`, `validate`, `set`, return statements, and relevant branches. L1 missing-method suggestions are secondary: full callees are included only when their real source units fit. All missing units and unresolved callees remain in diagnostics.
- `l1_graph_test_aware`: identical evidence selection and ranking to `_nostop`, plus an **uncalibrated, conservative** graph gate. It skips one candidate's graph action only when the current candidate is top-ranked, L1 suspicion >= 0.85, uncertainty <= 0.20, valid, and there is **no open missing-evidence question**.
- Historical `l1_anchor`, `l1_graph_suspicion`, `l1_graph_focused`, `graft_adaptive`, RRF, and existing fusion remain unchanged.

The policy **does not** use gold labels, corrected source, post-fix tests, or actual runtime reachability. It does not fetch unseen test files. It uses only the supplied failure node, its provenance, the immutable buggy-revision source nodes, and already-observed L1 output.

## New prompt schema and provenance

Only `test_aware_compact_v1` uses `graft-test-aware-v1` prompt JSON. The prompt includes exact target/source units, line ranges, and a `test_signal` field derived deterministically from the mandatory failure text. A partial `try {...; fail();` is labelled:
- `kind: possible_expected_exception`
- `certainty: incomplete_test_excerpt`
- `full_test_verified: false`
- `evidence_origin: supplied_text:trigger_description`

No fabricated catch blocks, test coverage, executable call edges, or claims of guaranteed semantic completeness.

## Offline verification plan, zero provider calls

Run the named script:

```bash
cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
conda activate rave-fl
git fetch origin graft-fl-aiscalar-20261113
git merge --ff-only FETCH_HEAD
bash scripts/aiscalar_graft_test_aware_offline.sh
```

It runs targeted unit and integration tests, then reconstructs a new bounded prompt from the **saved** Phase-2 graph and initial L1 observations. It records each policy's selected source lines, budget use, uncertain test-signal classification, whether the stop gate triggers, and all omitted source fragments. It **cannot** measure new Qwen accuracy because it makes no new model calls.

A code-based pre-check suggests exception and setter/return paths may fit inside 4096 byte-proxy units when the optional `indexOf` callee body is omitted, but this is **not yet a verified Python test result**. The same evidence proxy is used for all compared treatments. Exact model tokenizer counts are not claimed.

## Live development comparison, only after offline tests pass

```bash
bash scripts/aiscalar_graft_test_aware_live.sh
```

The controlled run compares:
1. `fixed_l1` + `l1_anchor`, no graph calls;
2. `l1_graph_focused`, previous Phase-2 method;
3. `l1_graph_test_aware_nostop`, new retrieval without stopping;
4. `l1_graph_test_aware`, identical retrieval with stopping.

Each graph policy is evaluated with both `rrf` and `suspicion_rank_blend`. Provider calls share exact requests in one run; logical per-arm totals must not be added into one bill. Hard caps: 45 new calls and 60,000 provider tokens. Stops before inference if tests, saved-data validation, or preflight fails. All outcomes including invalid responses must be kept.

### Evaluation metrics

For each treatment: first gold rank, Top-1, Top-5, MRR, graph target, selected fragment IDs, missing-evidence completeness, graph responses, guard decisions and logical tokens. Report global **physical** new calls/provider tokens separately and distinguish runtime from proxy measures.

**Scientific limit:** Time-4 is development data studied repeatedly. No method improvement can be claimed as generalizable without independent prepared bugs and, ideally, a frozen held-out cohort. Time-1/Lang-1/Math-1 remain blocked by missing real debugging evidence and missing buggy checkouts.
