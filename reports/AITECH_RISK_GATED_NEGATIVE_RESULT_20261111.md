# AITECH novelty search and risk-gated pilot, 2026-11-11

## Research question

Can preserving failing-test anchors and gating optional code dependencies by
confidence and token opportunity cost improve the debugging evidence shown
to an LLM, without expensive unconditional graph expansion?

This is a **candidate novelty**, not a proven original contribution.

## Relevant published and preprint work

- Graph-based bug localization with heterogeneous interprocedural edges and
  Bug Anchor Diffusion (TIC 2026),
  https://ieeexplore.ieee.org/abstract/document/11703025/
- Retrieval-Oriented Code Representations in Agentic Bug Localization (2026),
  https://arxiv.org/abs/2607.11046
- BLAgent, agentic file-level RAG and bounded reranking (2026),
  https://arxiv.org/abs/2605.17965
- Cost-aware evidence access in RAG (EMNLP Findings 2026),
  https://aclanthology.org/2026.findings-emnlp.413/
- CogniGent, graph-based agentic reasoning and hypothesis testing (2026),
  https://arxiv.org/abs/2601.12522

**Novelty caution:** graph retrieval, cost-sensitive context selection, and
agentic bug-localization are all established concepts. Our precise potential
contribution must therefore involve a validated interaction such as
source-provenance confidence, failure-anchor safety, cost-of-displacement,
and explicit abstention when graph evidence cannot help.

## Implemented risk-gated selector

`aitech/risk_gated_context.py` preserves lexically matched failure anchors
and then greedily selects remaining code snippets using their marginal
relevance+support value divided by their stored L1 sketch-token cost.
It accepts dependency edges only above a confidence threshold and reports
unresolved dependencies. Confidence values are *uncalibrated heuristics*.

`aitech/risk_gated_pilot.py` compares rank-fill, anchor-first,
always-expand and risk-gated on three archived real Defects4J bugs:
Lang-1, Time-1 and Time-25. The first two are development cases.
Time-25 was also studied earlier, so it is an exploratory transfer case,
**not a previously unseen holdout**.

All four arms use the same 200, 300, 450 and 600 stored L1 sketch-token
budgets. No model, patch data or gold scoring was used.

## Actual paired CPU results

GitHub Actions unit-test run `43072185499` completed successfully and
uploaded `aitech-risk-gated-43072185499` with JSON and Markdown results.

- 12 matched bug-by-budget conditions.
- Risk-gated and anchor-first selected different sets in **5/12** cases.
- **0** evidence units were promoted because of graph support.
- **0** anchor-retention gains versus anchor-first and **0** losses.
- **4/12** conditions had zero usable same-owner arity graph edges
  (all Time-1 budgets).
- Available same-owner arity graph edges: Lang-1 7, Time-1 0, Time-25 4.

| Bug | Budget | Context changed? | Graph promotions | Anchor delta | Token delta |
|---|---:|---|---:|---:|---:|
| Lang-1 | 200 | No | 0 | 0 | 0 |
| Lang-1 | 300 | No | 0 | 0 | 0 |
| Lang-1 | 450 | Yes | 0 | 0 | -9 |
| Lang-1 | 600 | Yes | 0 | 0 | -9 |
| Time-1 | 200 | No | 0 | 0 | 0 |
| Time-1 | 300 | Yes | 0 | 0 | +6 |
| Time-1 | 450 | Yes | 0 | 0 | -29 |
| Time-1 | 600 | No | 0 | 0 | 0 |
| Time-25 | 200 | No | 0 | 0 | 0 |
| Time-25 | 300 | No | 0 | 0 | 0 |
| Time-25 | 450 | Yes | 0 | 0 | -27 |
| Time-25 | 600 | No | 0 | 0 | 0 |

These results are an important **negative result**: we cannot attribute any
of the changes to graph expansion. Different packing order alone does not
prove better code evidence or better debugging accuracy.

## What to do next

1. Improve graph *coverage*, especially class-qualified and cross-class calls,
   without inventing resolved Java bindings.
2. Require source provenance and explicit uncertainty labels for every edge.
3. Compare evidence sets and support quality at matched token budgets.
4. Obtain an independent, frozen set of real buggy repositories and patch
   labels for file/function Hit@k and rank evaluation.
5. Only then run the *same* LLM with identical prompts, input tokenizer,
   context budgets and seeds to measure actual localization MRR/Hit@k.
6. Treat negative results as valid results; never claim an LLM gain when
   no LLM was run.

## Research execution mode

The recurring ChatGPT automation is paused at the user's request.
The daily cron trigger in the AITECH research GitHub workflow was also
removed. Work runs interactively or on code pushes, not on a schedule.
