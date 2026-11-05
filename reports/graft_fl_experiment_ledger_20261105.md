# GRAFT-FL experiment ledger, November 5, 2026

GRAFT-FL is being developed on [`graft-time25-qwen7b-budget40k-results`](https://github.com/ParisaArbab/AdaptiveContextOpt/tree/graft-time25-qwen7b-budget40k-results), separate from the production `main` pipeline. This ledger summarizes completed Qwen2.5 7B runs and does **not** claim statistical or held-out generalization.

## Time-25 development bug

- Original prior rank: 22.
- Guarded GRAFT adaptive without failure-name anchor: rank 10.
- Failure-name-anchored GRAFT adaptive: rank 2, MRR 0.5000.
- Anchored L1 graph: rank 3, MRR 0.3333.
- This bug was used to design the heuristic, so it is development evidence only.

## Three additional exploratory defects

| Bug | Prior rank | GRAFT adaptive control rank | GRAFT adaptive anchored rank |
|---|---:|---:|---:|
| Lang-1 | 2 | 2 | 2 |
| Math-1 | 1 | 1 | 1 |
| Time-1 | 1 | 1 | 1 |

Both runs finished all three bugs. Across these three bugs, adaptive MRR was 0.8333 in both arms, Top-5 was 3/3 in both arms, and the recorded whole-run provider accounting was 84 calls / 62,141 tokens for control versus 84 calls / 62,229 tokens for anchored. Failure-name anchoring produced no rank gain in this easy set.

## Time-4 exploratory hard case

Gold-aware offline screening of twelve previously declared bugs identified Time-4 as the only accessible case with the correct method originally outside the Top-5 (rank 10). Two separate Qwen runs completed:

| Policy | Control correct-method rank | Failure-name anchored rank |
|---|---:|---:|
| Prior | 10 | 10 |
| Fixed L1 | 6 | 5 |
| Highest suspicion | 10 | 10 |
| L1 graph | 6 | 5 |
| GRAFT adaptive | 6 | 5 |

Time-4 adaptive MRR improved from 0.1667 to 0.2000, and Top-5 changed from 0 to 1. Each whole run required 29 real provider calls. Control used 15,662 provider tokens; anchored used 15,614. This tiny token difference is **not** meaningful compression evidence.

Because Fixed L1 achieved the same rank change, this Time-4 comparison does not independently establish that the graph contributes beyond failure-name matching.

## Research limitations and evidence

- Qwen2.5 7B was run with batch size 1. Earlier batch-size-2 requests returned one response for a two-method batch and generated incomplete arms.
- The Time-1, Math-1, Lang-1, and Time-4 runs used externally supplied FlexFL trigger descriptions, not independently executed failing tests.
- The Time-4 bug was selected by an offline screen using gold rank. The selection is explicitly exploratory, not held-out.
- A full repository-based fault-localization benchmark, graph-specific ablation, and independent unseen-bug validation are still needed.
- The code, configs, source-span tests, trace diagnostics and full three-bug provider results are preserved on the research branch.
- The **raw Time-4 AIScalar run artifacts are not yet archived to GitHub**. See its preliminary [Time-4 report](https://github.com/ParisaArbab/AdaptiveContextOpt/blob/graft-time25-qwen7b-budget40k-results/reports/graft_time4_qwen7b_exploratory_20261105.md).

This file exists on `main` for research traceability. It does not merge or activate experimental GRAFT-FL code.
