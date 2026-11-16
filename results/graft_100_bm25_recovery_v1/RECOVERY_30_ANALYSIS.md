# GRAFT-FL corrected-BM25 recovery: paired analysis of 30 bugs

**Date:** 2026-11-16  
**Source:** `results/graft_100_bm25_recovery_v1/batches/{batch_02,batch_06,batch_07,batch_11,batch_14,batch_19}/experiment/paired_audit.json`; `results/graft_100_bm25_retry_2_v1/experiment/paired_audit.json`.  
**Selection rule:** Original recovery results for 28 individually complete bugs, except take **Math-69** and **Time-22** only from the isolated, fully passed two-bug retry. Each bug contributes exactly once per policy and fusion. Original Batch 06's failed accounting is **not** rewritten or treated as completed.  
**Gold:** official Defects4J modified-classes file-level proxy, **not method-level ground truth**.

## Main finding
All four policies produce **identical file-level rankings on all 30 bugs for both fusion methods**. Two graph variants each execute an average of 2 graph calls per bug, but change no measured file ranking. Graph evidence has **zero observed improvement** over L1 in this cohort. Fusion changes rankings independently of the graph selection policy.

## Paired accuracy (30 bugs; no missing policy/fusion records)

| Fusion | Policy | MRR | Top-1 | Top-5 | Mean logical tokens per bug | Mean graph calls per bug |
|---|---|---:|---:|---:|---:|---:|
| RRF | L1 anchor | 0.56361 | 12/30 (40.00%) | 24/30 (80.00%) | 40,763.87 | 0 |
| RRF | Graph-focused | 0.56361 | 12/30 (40.00%) | 24/30 (80.00%) | 42,688.20 | 2 |
| RRF | Test-aware, no stop | 0.56361 | 12/30 (40.00%) | 24/30 (80.00%) | 42,755.47 | 2 |
| RRF | Test-aware, with stop | 0.56361 | 12/30 (40.00%) | 24/30 (80.00%) | 42,742.47 | 2 |
| Suspicion-rank blend | L1 anchor | 0.66389 | 16/30 (53.33%) | 25/30 (83.33%) | 40,763.87 | 0 |
| Suspicion-rank blend | Graph-focused | 0.66389 | 16/30 (53.33%) | 25/30 (83.33%) | 42,688.20 | 2 |
| Suspicion-rank blend | Test-aware, no stop | 0.66389 | 16/30 (53.33%) | 25/30 (83.33%) | 42,755.47 | 2 |
| Suspicion-rank blend | Test-aware, with stop | 0.66389 | 16/30 (53.33%) | 25/30 (83.33%) | 42,738.37 | 2 |

Blend versus RRF: **+0.10028 absolute MRR**, **+4 Top-1 hits** (12→16), **+1 Top-5 hit** (24→25), on the same bugs. Treat as descriptive, not evidence of statistical significance. Gold changed-file coverage in candidate pool averages **0.86667** across bugs.

## Which bugs changed under fusion (same for every policy)

| Bug | RRF first correct-file rank | Blend rank | Change |
|---|---:|---:|---|
| Math-53 | 3 | 1 | Better |
| Closure-85 | 8 | 2 | Better |
| Chart-5 | 3 | 1 | Better |
| Closure-161 | 6 | 1 | Better |
| Time-19 | 3 | 1 | Better |
| Time-13 | 5 | 6 | Worse |
| Math-58 | 2 | 3 | Worse |

Five bugs improved and two worsened under fusion. The remaining 23 did not change their first correct-file rank.

## Where graph failed to help
No bugs improved or worsened relative to L1 from adding graph evidence, for either fusion (30 equal results out of 30). In particular, **Closure-32, Closure-106, Math-96, Closure-96** have MRR 0 for both fusions and all graph policies. These are relevant failure cases for future candidate-recall and graph-target diagnostics; the present files alone do not prove why those bugs failed.

## Context cost and stop rule
Compared with L1 anchor, graph-focused uses **+1,924.33 mean logical tokens per bug** (+4.72%); test-aware without stopping uses **+1,991.60** (+4.89%). Test-aware with stop uses **+1,978.60** (+4.85%) under RRF, and **+1,974.50** (+4.84%) under blend. These are logical token counters, **not separately billed provider tokens**.

The stop gate triggered in **11 of 60 bug-fusion pairs** and reported **903 total logical tokens saved** across those pairs, but **zero graph model calls saved**. Some individual stop comparisons had negative savings; this is the net reported total, not a guarantee of reduced physical cost.

## Physical Qwen usage and accounting caveats

| Archive | New calls | Known provider input+output tokens | Unknown reservation |
|---|---:|---:|---:|
| Batch 02 | 140 | 206,473 | 0 |
| Batch 06, original partial | 98 | 159,718 | 8,352 |
| Batch 07 | 141 | 212,387 | 0 |
| Batch 11 | 141 | 212,091 | 0 |
| Batch 14 | 141 | 258,623 | 0 |
| Batch 19 | 142 | 232,057 | 0 |
| Math-69 / Time-22 retry | 56 | 83,987 | 0 |
| **Total executed** | **859** | **1,365,336 known** | **8,352 unresolved reservation** |

Physical provider usage counts are **shared across policies/fusions** (via cache and run structure). Do not assign the physical totals to individual policies or interpret mean logical tokens as provider billing. The Batch 06 incomplete attempt remains part of the *run-level* consumed resources, even though its two unfinished bugs are taken from the isolated retry for *accuracy*.

## Scientific interpretation and next steps
1. **Observed:** corrected BM25 + suspicion-rank blend substantially improves this cohort's file-level rankings versus RRF, while the current graph-selection interventions do not change file-level accuracy.
2. **Unresolved mechanism:** check whether selected graph evidence is consumed by the decision/ranking stage, whether the model outputs differ, and whether BM25 candidate recall or graph edges limit impact.
3. **Cohort limits:** 30 recovery bugs, not 100 uniformly re-evaluated with corrected BM25; these are selectively recovered batches, not an independently sampled performance benchmark. Do not merge old 58-bug metrics with this corrected-BM25 subset as if method configurations were identical.
4. **Next evaluation:** run the same corrected implementation on a fixed 100-bug cohort with independently verified gold and compare paired results, calls, context, wall-time, and output quality. Report confidence intervals and failure categories.

This report is an offline calculation from the archived paired audits and requires **zero new model calls**.
