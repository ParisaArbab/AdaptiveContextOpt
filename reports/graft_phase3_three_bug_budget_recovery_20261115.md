# Phase 3 three-bug recovery: preserve physical Qwen calls and finish L1 before comparing graph evidence

Date: 2026-11-15
Research branch: graft-fl-aiscalar-20261113
Original run: results/graft_phase3_real_bugs_20261113_222808/run
Archived experiment commit: aaffaee

## Observed and verified result (old run)

The official Defects4J Time-1, Lang-1, and Math-1 preparation succeeded, with real
observed buggy failing tests and a fixed 24-method BM25 candidate pool per bug.
The 59 pre-inference tests and all three preflights succeeded. Qwen2.5:7b
produced 43 physical model calls and 98,430 provider input+output tokens,
with zero unknown reservations and no simulated calls. All 30 non-prior arms
were incomplete, and all six prior arms completed. These results were archived
to GitHub; they are preliminary development diagnostics, not held-out evidence.

All three test-aware policies in the RRF arms stopped with
"incomplete_l1_budget_or_context" before any graph calls:

| Bug | L1 logical tokens before stop | L1 candidates | Graph calls |
| --- | ---: | ---: | ---: |
| Lang-1 | 33,351 | 24 | 0 |
| Math-1 | 32,619 | 24 | 0 |
| Time-1 | 32,460 | 24 | 0 |

The configured 40,000 logical tokens per policy cannot admit another
8,192-token context plus 160-token output reservation once the L1 spend is
greater than 31,648. This blocked candidate 5 of 24 (or later) before graph
expansion. The observed RRF file ranks Lang=1, Math=2, Time=1 and rank blend
ranks all 1 are *partial*, and should not be called completed graph wins.
The modified-class gold is only file-level proxy truth, not exact method truth.

## Recovery design

1. Archive the old run unchanged, including cache, journal, accounting, raw
   ledgers, candidate evidence, reports, and full configuration.
2. For the next exploratory feasibility run, increase **per-policy logical
   token budget** from 40,000 to 90,000. This is a declared correction based
   on incomplete L1 admission, not tuning on ranking quality or gold results.
3. Keep the **new-run physical** provider limits unchanged:
   240,000 input+output provider tokens total, 80,000 per bug, 135 new calls.
   These limits are PER RUN, not cumulative with the archived prior 98,430.
4. Optionally seed the new output's independent cache with validated prior
   Qwen response JSON. Verify the original dataset SHA-256, frozen request
   options, model digest, cache filenames, response counters, payload model,
   and exact response SHA-256. Store again checks each request's exact identity
   before accepting cached data. Do not copy the old SQLite journal, overwrite
   the old cache, or claim copied responses are new physical calls.
5. Preserve the new run's own provenance. No prior policy rankings are used as
   new policy decisions. All policies still evaluate the same original
   candidate pool, and the model may be called for any not-yet-seen request.
6. A completed run still requires valid L1 observations and successful
   admissible steps; the larger budget is NOT a guarantee of completion or
   graph benefit. If source or model changes, exact cache hits must fail.

## AIScalar command

    cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
    conda activate rave-fl
    git fetch origin graft-fl-aiscalar-20261113
    git merge --ff-only FETCH_HEAD
    python -m pytest -q tests/test_graft_phase3_cache_seed.py tests/test_graft_phase3_real_bugs.py

    bash scripts/aiscalar_graft_phase3_multibug.sh \
      data/graft_phase3_official_20261113_215102/manifest.json \
      --seed-run results/graft_phase3_real_bugs_20261113_222808/run

Expected before inference: exact response files copied, zero new provider calls,
three bug preflights "ready", correct physical caps and 90,000-token logical cap.
Audit each new cache hit and physical token counter after the run. Never sum
logical policy token counts as physical Qwen usage. Any comparison with
the old 40,000-budget run must be labeled a feasibility/config change.

No new Qwen calls were executed by preparing this fix on GitHub.
