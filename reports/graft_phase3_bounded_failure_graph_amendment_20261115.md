# Phase 3 real Qwen three-bug graph feasibility amendment, 2026-11-15

## Observed result from the completed 90k logical-policy run

Experiment results (archived commit 904b976 or later):

    results/graft_phase3_real_bugs_20261115_132745/

- 3 authentic Defects4J buggy revisions, Time-1, Lang-1, Math-1
- 24 BM25 method candidates per bug, real observed failing-test evidence
- All 36 arms completed, 29 new Qwen2.5:7b calls, 69,492 actual
  provider input+output tokens, zero pending or unknown token usage
- File-level modified-class proxy rank 1 and MRR 1.0 across all saved
  policies/fusions; no exact method-level gold available
- **Zero real graph calls**, as reported by the independent completion gate
- Every graph-focused graph package rejected as
  focused_mandatory_exceeds_budget and every no-stop test-aware package as
  test_aware_mandatory_exceeds_budget

Root cause: the *mandatory complete failure node alone* is 5,548 to 8,096
characters, while the entire graph evidence message including the system
prompt, target signature, sources and template overhead is limited to 4,096
UTF-8 bytes. A larger logical policy token budget cannot resolve that.

The old run demonstrates successful L1 file ranking, not a graph accuracy gain.

## Separate, explicit amended graph prompt condition

The next exploratory run declares:

    graft.failure_digest_mode = "bounded_excerpt"

Only the graph prompt's mandatory failure node presentation is changed. The
immutable full failure text remains in the evidence graph, the manifest,
archive, and every original L1 prompt. For graph LLM calls, an explicit v2
renderer instead selects exact numbered lines from the observed failure:
official bug and test identifiers, test source SHA, real first/last test
statements and assertions, reported exception(s) and first observed
project/test stack frames. Selected text is never paraphrased and excluded
lines are explicitly disclosed.

Each v2 graph message records the full failure's SHA-256 hash, number of
original lines/bytes, a fixed selection rule, and the exact line numbers/text
that were kept. Maximum original selected-line content is 1,650 UTF-8 bytes.
This leaves room for the method signature and real AST-backed source units
inside the existing 4,096-unit evidence budget. Completeness cannot be
claimed for omitted evidence or unresolved callees.

V1 focused and test-aware selectors are still unchanged and are the default
outside this separate condition. Graph retrieval, candidate pool, gold, model,
batch size, graph evidence budget, verification strict admission, and fixed
physical provider limits are not relaxed.

The new v2 graph prompt is **not directly equivalent** to the previous
full-failure prompt, and results must be labeled as a new condition. It is
legitimate to compare its four policies within the new run because they see
the same full L1 baseline and verified failing test. Evaluation gold is only
used offline after choices are made.

## Required pre-inference safeguard

Before new Qwen inference, the launcher:

1. verifies previous source run, exact request identities, provider digest,
   counters, and current prepared official bug inputs
2. seeds a fresh cache from the November 15 run's sibling cache directory,
   which holds the earlier 43 exact Qwen responses plus the 29 new L1 calls
3. runs the candidate preflight for all three bugs
4. reuses the **previous online L1 observations**, without reading gold, to
   simulate v2 graph packing for every one of the 24 candidates per bug
5. requires at least one ready graph package in each bug's online L1 top 5
   for BOTH focused and test-aware-no-stop selectors, with 4,096-unit bounds
6. only after passing these checks starts bounded Qwen calls for unseen
   graph requests; otherwise stops before inference and prints a diagnosis

New cached responses are reported as cache reuse and never counted as
fresh physical provider calls. The original archives remain read-only.

### AIScalar verification and launch

    cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
    conda activate rave-fl
    git fetch origin graft-fl-aiscalar-20261113
    git merge --ff-only FETCH_HEAD
    python -m pytest -q \
      tests/test_graft_phase3_bounded_failure.py \
      tests/test_graft_phase3_cache_seed.py

    bash scripts/aiscalar_graft_phase3_finish.sh

The launcher prints the log. The resulting experiment stores:
graph_readiness.json, cache_seed_provenance.json, paired_audit.json and
completion_gate.json. A successful "Completed real graph comparison: True"
must be confirmed from the latter and actual graph_calls, not inferred from
the command returning successfully.

All code preparation and offline audit performed here adds **zero new Qwen
calls**. This document asserts no unobserved new real graph accuracy result.
