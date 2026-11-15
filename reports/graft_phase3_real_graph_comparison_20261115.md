# GRAFT-FL Phase 3, completed real three-bug graph comparison

Run: `results/graft_phase3_real_bugs_20261115_134708/`
Date: 2026-11-15
Official bug IDs: Lang-1, Math-1, Time-1
Research branch: `graft-fl-aiscalar-20261113`
Evidence: archived real Defects4J buggy revisions and observed failing-test output, with immutable full evidence retained in graph and baseline.
LLM: Qwen2.5:7b through Ollama, strict inference accounting.
Graph-prompt condition: explicit opt-in **bounded observed-failure excerpt v2**, with exact line numbers, source SHA-256 and documented omissions. The original full-failure graph-prompt condition is not directly comparable.
Gold semantics: **official changed-source-file proxy**, not exact faulting method labels.

## Main answer

**Completed real graph comparison: TRUE. Graph improves file localization on these three bugs: NOT OBSERVED.**

The independent completion gate confirmed all six bug/fusion comparison pairs complete, all 36 configured arms complete, and real graph evidence calls in both graph-focused and test-aware-no-stop arms. The entire project did not finish in this run: it was a three-bug development experiment, not a held-out generalization study.

- 72 previously saved exact Qwen request responses were seeded from the archived previous run. Cache reuse made zero new physical calls for those responses.
- **12 new real Qwen inference calls**, **13,758 new provider input+output tokens**, zero simulated calls, zero unknown usage or pending reservations.
- All 36 arms complete (including prior arms). Graph-focused and test-aware/no-stop arms each made two logical graph calls per bug/fusion; some request responses were shared by exact cache matching.
- Graph-focused, test-aware without stopping, and test-aware with stopping all had file-level changed-file proxy MRR **1.0**, equal to L1-only, for all three bugs and both RRF and suspicion-rank-blend fusion.
- The 6 audited pairs all show `file_mrr_test_aware_minus_focused = 0`, `file_mrr_stop_minus_nostop = 0`, and `graph_calls_saved_by_stop = 0`.
- The test-aware stopping gate **did not activate** in any bug/fusion pair.

## File-level results, each fusion showed the same ranks

| Bug | L1-only file rank / MRR | Graph-focused file rank / MRR | Test-aware file rank / MRR | Focused graph calls per arm | Test-aware graph calls per arm |
| --- | --- | --- | --- | ---: | ---: |
| Lang-1 | 1 / 1.0 | 1 / 1.0 | 1 / 1.0 | 2 | 2 |
| Math-1 | 1 / 1.0 | 1 / 1.0 | 1 / 1.0 | 2 | 2 |
| Time-1 | 1 / 1.0 | 1 / 1.0 | 1 / 1.0 | 2 | 2 |

File-level gold coverage is not perfect: Time-1 has only one of two official changed files represented in the candidate pool (coverage 0.5). The file-rank metric reports the **first** changed source file and is not a guarantee that the true faulty method is at rank 1.

## Additional logical verification budget used by graph

Logical units are per-arm prompt accounting, **not new physical provider tokens**. The same L1 observations are reused across policies and fusion methods; never sum the table below as physical expenditure.

| Bug | L1-only logical units | Graph-focused logical units | Extra focused units | Test-aware logical units | Extra test-aware units |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lang-1 | 60,552 | 62,900 | 2,348 | 62,970 | 2,418 |
| Math-1 | 64,320 | 66,701 | 2,381 | 66,740 | 2,420 |
| Time-1 | 43,050 | 45,138 | 2,088 | 45,153 | 2,103 |

Both policies incurred additional logical cost with **zero observed file-rank gain** on this cohort.

## What changed compared with the previously incomplete experiments

- The prior 40k logical-policy experiment stopped during initial L1 screening. See `results/graft_phase3_real_bugs_20261113_222808/`.
- The next 90k logical-policy run completed L1 but made zero graph calls because a mandatory full failure evidence block exceeded the unchanged 4,096-unit graph evidence budget. See `results/graft_phase3_real_bugs_20261115_132745/`.
- The opt-in bounded failure evidence view retained source fidelity and left the 4,096-unit evidence budget unchanged. Its offline readiness gate verified that real packages fit before inference; then the real model performed graph calls. See `results/graft_phase3_real_bugs_20261115_134708/graph_readiness.json`.

## Reliability notes

The graph LLM's written explanation is not ground truth. In the Time-1 graph response it says a test "passes"; **this graph stage did not run tests**, so that sentence must be treated as an unsupported model-generated judgment. The graph responses also often gave low suspicion and high uncertainty to source methods, but there was no change to the evaluated file rank. Never cite these natural-language judgments as actual test outcomes or proof that the bug is fixed.

This small development cohort has a **ceiling effect**: all three L1 baselines already achieve file MRR 1.0. It cannot demonstrate a Top-1 accuracy improvement. The results instead indicate that graph expansion made the method more expensive without improving this proxy metric, and that the stopping rule did not save calls on these cases. This is neither a proof that graph methods never help nor evidence that they do.

## Next scientifically useful comparison

Freeze a *new*, larger, independent set of real bugs **before seeing their gold rankings**. Include cases where the L1 baseline has a lower rank, a real missing dependency, or uncertainty that motivates graph expansion. Continue to compare the same candidates and failure evidence across L1-only, graph-focused, test-aware-no-stop, and test-aware-stop. Record method-level accuracy only if reliable method-level gold is available, plus file MRR/Top-1/Top-5, graph package completeness, stop decisions, physical calls, prompt units, latency and all invalid outputs. No accuracy gain or token savings should be asserted until confirmed on such cases.

## Direct artifacts

- `results/graft_phase3_real_bugs_20261115_134708/completion_gate.json`
- `results/graft_phase3_real_bugs_20261115_134708/paired_audit.json`
- `results/graft_phase3_real_bugs_20261115_134708/graph_readiness.json`
- `results/graft_phase3_real_bugs_20261115_134708/run/accounting.json`
- `results/graft_phase3_real_bugs_20261115_134708/run/raw_ledgers/requests.jsonl`
- `results/graft_phase3_real_bugs_20261115_134708/run/instances/*/*.json`

This report is an audit of archived outputs. Writing it made **zero new Qwen calls**.
