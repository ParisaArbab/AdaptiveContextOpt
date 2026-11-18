# Method Discovery Agent: Detailed Review of Five Real Defects4J Bugs

Source: user-uploaded `method_discovery_analysis.zip` (`results.json`, `grid.json`, `report.md`, `summary.csv`, `sweep_report.md`, `tradeoff.svg`). No new inference was executed for this review.

## Main results

| Arm | Mean file MRR | Delta MRR vs L1 | Top-5 | Mean logical tokens | Delta tokens | Paired wins/ties/losses |
|---|---:|---:|---:|---:|---:|---|
| l1 | 0.4900 | +0.0000 | 60% | 33,036 | +0 | [0, 5, 0] |
| risk_full | 0.5567 | +0.0667 | 80% | 35,644 | +2,607 | [1, 4, 0] |
| code_cards | 0.5222 | +0.0322 | 60% | 29,251 | -3,785 | [2, 3, 0] |
| combined | 0.5067 | +0.0167 | 80% | 38,788 | +5,752 | [1, 4, 0] |
| dependency_repair | 0.4967 | +0.0067 | 80% | 36,148 | +3,112 | [1, 4, 0] |
| esr_proxy | 0.5000 | +0.0100 | 60% | 35,607 | +2,571 | [1, 3, 1] |
| cascade | 0.4900 | +0.0000 | 60% | 34,632 | +1,596 | [0, 5, 0] |
| adaptive_k | 0.4067 | -0.0833 | 80% | 4,308 | -28,728 | [1, 3, 1] |

All above results use five matched bugs and **file-level modified-class proxy labels**. They are observation-replay results using real Qwen replies; logical tokens are not the wall-clock or the physical tokens spent collecting all independent responses.

The non-dominated choices in MRR-versus-token space are `adaptive_k` (very cheap but lower MRR), `code_cards` (cost-accuracy tradeoff), and `risk_full` (highest MRR).

## Per-bug file ranks

| Bug | L1 | Risk Full | Code cards | Combined | Dependency repair | ESR proxy | Adaptive-k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chart-17 | 1 | 1 | 1 | 1 | 1 | 1 | 2 |
| Closure-165 | 5 | 5 | 3 | 5 | 5 | 6 | 5 |
| Lang-28 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Math-95 | 6 | 2 | 6 | 4 | 5 | 4 | 4 |
| Time-26 | 12 | 12 | 9 | 12 | 12 | 12 | 12 |

## Mechanism-level findings

1. **Risk-based Full** improved only **Math-95**, file rank 6 to 2 (+0.3333 MRR on that bug); the other four bugs had exactly unchanged ranks. Both Full checks matter in this pilot: the first candidate on Math-95 was `ChiSquareTestImpl.chiSquareTest` (Full suspicion 0.30), and the second was `FDistributionImpl` constructor (Full suspicion 0.85). A one-Full budget produced zero average MRR gain; two Full checks produced +0.0667 average.
2. **Code cards** improved **Closure-165** from rank 5 to rank 3 and **Time-26** from rank 12 to rank 9, without worsening any file ranks. Their Top-5 outcomes stayed unchanged, so the method helps ranking position, not Top-5 recall on this cohort. It used fewer measured logical tokens on each of the five bugs.
3. **Dependency repair** improved only **Math-95** rank 6 to 5. The combined DARC variant improved only Math-95 rank 6 to 4 and cost substantially more. The graph-derived `MAY_CALL` links are hypotheses, not proven causal dependencies.
4. **Adaptive-k** used only three L1 observations per bug and saved considerable cost, but demoted Chart-17 rank 1 to 2, reducing MRR by 0.5 on that bug. The replay implementation leaves uninspected candidates ranked by baseline prior rather than excluding them, so it is a selective-screening experiment, not exact top-k truncation.
5. **ESR proxy** improved Math-95 rank 6 to 4 but hurt Closure-165 rank 5 to 6. It is an uncertainty-times-influence heuristic, not the exact-verifier procedure from the OECP paper.
6. **Cascade** left all five file ranks unchanged and used more logical tokens than L1; the low-level cheap screening is already available, so conditional Full verification did not save tokens relative to L1 here.

## Statistical evidence

`risk_full` has a five-bug bootstrap 95% interval for mean delta-MRR of **[0, 0.20]**. `code_cards` has **[0, 0.08556]**. Both contain zero. Neither is a confirmed improvement. A development/held-out split and more independent bugs are necessary.

## Budget sweep

The grid tried 24 parameter combinations (4 budgets × 3 verification caps × 2 buffers) but only **12 configurations** produced any five-bug paired summaries. Budgets 16k and 24k were below the 30k-36k L1 logical cost and could not support a complete L1 baseline. The sweep reported **46 exploratory (configuration, arm) entries**, many duplicates across buffer settings.
At budgets of at least 40k, `risk_full` with two Full checks reached +0.0667 mean delta MRR (approximately +2,607 logical tokens). One Full check produced no MRR gain; three Full checks cost more without further gain on the full five-bug sample at 60k. `code_cards` averaged -3,785 tokens and +0.0322 MRR across all supported settings.

## Known limits

- Sample size is five bugs from the predeclared development cohort, one per project, and only file-level modified-class proxies are available. Do not claim method-level fault localization.
- The policy simulator uses independently collected real responses, not live policy execution with measured end-to-end latency.
- The agent uses simple prior/LLM-score fusion and a distinct L1 prompt from archived GRAFT; direct numeric comparisons to previous GRAFT results are invalid.
- The collector identified an interrupted atomic write; the physical token accounting is an **incomplete lower bound**. Individual logged logical call counts remain measured for replay.
- `conformal_k` is not calibrated because there are zero method-gold calibration bugs; `llmlingua2` has no real observations and cannot be evaluated.
- The `code_cards` action changes code text and the `Evidence form: card` prompt label. It is not a controlled isolate of compression alone. Some code is omitted by simple keyword selection and may remove crucial syntax or dependencies.

## Next experiments: freeze a focused pilot

**Priority 1: risk-aware Full selection near the Top-5 boundary.** Use the failure stack frame, AST call links and base rank to rank candidate Full actions, then verify one targeted candidate at a time. Compare (a) highest uncertainty, (b) highest prior, (c) boundary-aware, (d) failure-anchored, with 1 and 2 Full caps. The Math-95 action sequence makes targeting especially important.
**Priority 2: code cards plus selected Full verification.** Build an actual new `code_cards_risk_full` policy, run card screening on all 24 candidates, choose Full based only on observed card uncertainty plus source-grounded graph signals, and rerank. Do not assume existing L1 risk-selected Full observations cover the card-selected candidates. First check coverage of the saved Full lookup, then collect new responses for missing actions.
**Priority 3: matched unchanged evaluation.** Freeze criteria using new development bugs (10 to 15), then test on a separate held-out group (at least 30 complete bugs, ideally 80). Do not tune on the held-out set; record per-bug paired changes, file Top-1/Top-5/MRR, logical/physical tokens, candidate-pool coverage and actual latency.

## Source integrity checks

- Source results manifest SHA256: `19068806e0e368f9782d029ea49a802d4f48bbf9ea3ff3cb71fb84f2b35a5098`
- Source observation SHA256: `51bb52a5dcaaa85b51c794701bf3188bc9899a563ffaf355f01adf7ebbab3d5a`
- Real provenance: `live_ollama`
- Physical accounting complete: `False`
- Source rows: 50; completed: 40; incomplete: 10.