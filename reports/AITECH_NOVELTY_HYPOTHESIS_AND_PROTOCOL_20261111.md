# AITECH Novelty Assessment and Experimental Protocol

**Updated:** 2026-11-11
**Status:** research hypothesis, NOT an established novel method or demonstrated AI accuracy gain.

## 1. Problem and falsifiable hypothesis

An AI debugging agent sees a failed test, stack trace, source snippets, and
interprocedural relationships. Each additional snippet costs prompt tokens,
retrieval time, and possibly model confusion.

**Hypothesis H1:** A calibrated decision policy that predicts *marginal
localization benefit* from acquiring or upgrading each evidence item can
reduce harmful context expansions compared with relevance-only ranking,
fixed summaries, and unconditional graph expansion, at the same TOTAL model
input-token and inference-compute budget.

The policy may take actions:
- abstain from retrieving anything more;
- preserve an existing method signature;
- upgrade to structured code sketch;
- upgrade to more complete code body;
- retrieve a class/method through a *verified* dependency binding.

The policy should base choices on available pre-gold observations:
failure type; presence and strength of file/function clues; semantic relevance;
cost to include or upgrade; method rank; source/edge provenance;
ambiguity/unresolved bindings; already selected context and remaining budget.

**Important:** The existing pilot used uncalibrated weights. No current result
proves that a learned risk gate improves localization.

## 2. Closely related methods, why simple RAG+graph is insufficient

| Work | Established result or mechanism | Remaining AITECH question |
|---|---|---|
| FaR-Loc (2025) | Functionality extraction, dense retrieval, LLM reranking for method localization | Whether dynamic acquisition/abstention reduces harmful evidence |
| CosFL (2026) | LLM query refinement, multi-granularity code search and voting | Calibrated marginal value of *upgrading* retrieved code under token limit |
| BLAgent (2026) | Path+AST chunks, two-phase agent reranking and bounded candidate reasoning | When to refuse optional dependency/source expansion |
| Retrieval-Oriented Code Representations (2026) | Code representation cost/accuracy tradeoffs, role-aware summaries | Per-bug adaptive selection instead of a fixed representation |
| Role of FL Context for LLM Repair (2026) | More FL context can hurt; file-level choice especially important | Predict the harmful expansion BEFORE spending tokens |
| From Finding to Fixing (2026) | FL accuracy does not consistently translate into successful agentic repairs | End-to-end localization/repair *net benefit* accounting for tool cost |

Primary links:
- https://arxiv.org/abs/2509.20552
- https://doi.org/10.1145/3757915
- https://arxiv.org/abs/2605.17965
- https://arxiv.org/abs/2607.11046
- https://arxiv.org/abs/2604.05481
- https://arxiv.org/abs/2610.25826

The exact novelty claim requires a more exhaustive literature review and a
measured improvement on unseen bugs. The table is NOT proof of novelty.

## 3. Actual AITECH evidence to date

Three previously examined Defects4J bugs (Lang-1, Time-1, Time-25), archived
candidate pools, stored L0/L1 token proxies, and official posthoc patched-method
labels. The bugs are *development-only* and do not constitute a clean unseen test.

1. Forced graph closure can remove failure-named methods when the bundle is
   too large (Time-1). Zero dangling links does not imply retained root cause.
2. Risk-gated and failure-conditioned lexical graph methods produced no proven
   graph-based advantage over relevance-only selection. The dependency bonus
   did not change the selected evidence in the latest 12 budget/bug conditions.
3. L0 identity coverage must not be equated with L1 detailed source coverage.
4. New marginal-utility multi-resolution pilot: compared with the single-L1
   control across 12 bug-budget cases, detailed changed-method coverage
   improved once (Time-1 at budget 200) and decreased four times
   (Time-25 at all budgets).
5. Against the *best existing tiered AITECH control*, the new candidate
   **won 0, tied 8, lost 4**. Reject it as the primary research contribution.
6. Gold-aware oracle ceilings show: Lang-1 at 600 tokens could fit one
   changed-method sketch but the compared selectors missed it; Time-25
   changed-method sketch fit in every budget but the new tiered selector
   downgraded it to identity-only.
7. All these are pre-LLM evidence selection or posthoc coverage measures.
   None demonstrates a new LLM bug localization gain.

The current runner for an *actual* free CPU 0.5B code-model comparison is
`aitech/cpu_llm_localization.py`; the one-time workflow is
`.github/workflows/aitech-cpu-llm-paired-v2.yml`.
Do not report LLM results until its generated model output is verified.

## 4. Necessary experiment design for a publishable contribution

- **Data:** Defects4J projects with fixed buggy revisions and source hashes;
  separate independent train, calibration, validation, and final test projects.
  Avoid reusing Lang/Time bugs as an unseen final benchmark.
- **Outcome label:** Did adding/replacing evidence improve model method/file
  rank or patch success at equal prompt and inference budget? Log harmed cases.
- **Learning:** Learn calibrated benefit/harm probabilities from frozen
  context/action/outcome tuples; compare calibrated model, fixed heuristic,
  and no-graph controls. Use a small contextual decision policy only after
  sufficient independent outcomes.
- **Metrics:** Changed-method L1 coverage, Hit@1/5, MRR, accepted prediction
  validity, model input/output tokens, latency, localization/repair successes,
  miscalibration/Brier score, and harmful-expansion rate.
- **Paired tests:** Same model, tokenizer, temperature, seeds, candidate pool,
  prompts and total input-token ceiling. A no-retrieval control must exist.
- **Strong baselines:** rank fill; anchor first; best AITECH tiered method;
  role-aware summaries; semantic retrieval; FaR-Loc/CosFL/BLAgent-style
  candidate reranking when feasible; full graph closure; risk-gated retrieval.
- **No leakage:** Freeze policy and selections before reading patch labels.
  Do not tune on the final test. Report the fraction of bugs where correct
  source was absent from the initial candidate pool.

## 5. Decision rule to attempt next

For state s and retrieval action a, predict:

    net_value(s,a) = P(improves | s,a)*gain
                     - P(harms | s,a)*harm_cost
                     - lambda*extra_model_tokens
                     - mu*extra_latency

Use the model only when calibrated marginal benefit is positive and the
source/target identity is not ambiguous. Otherwise abstain.

This is a **research proposal**, not an evaluated classifier or a
mathematically proven superior policy.

## 6. Stop/go

**NO-GO on declaring novelty now.** Advance only after obtaining paired
real LLM outcomes on multiple independent bugs showing a measurable,
statistically defensible improvement or a valuable reproducible negative
result. Do not optimize for GitHub commit quantity instead of evidence.
