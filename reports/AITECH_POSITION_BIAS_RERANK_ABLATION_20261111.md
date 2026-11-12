# AITECH Research Iteration: Fixing Candidate-Position Bias in Real LLM Fault Localization

**Date:** 2026-11-11 (America/New_York)
**Branch:** `AITECH`
**Status:** 0.5B five-arm study completed; 1.5B model study launched separately.
**Decision:** Do not promote the new semantic/stack reranker as a superior method.

## Motivation

Previous same-LLM experiments added more L1 code context on four
Defects4J bugs and did not change any predictions. The 0.5B Qwen
decoder often chose the *first* method, regardless of the added code.

The original proposed marginal-action learner cannot learn meaningful
positive or harmful labels from such neutral actions. We therefore
tested whether improving evidence ranking and controlling candidate
position can separate source relevance from decoder bias.

## Changes implemented

1. `aitech/failure_aligned_reranker.py`: deterministic pre-gold stack
   alignment and IDF-weighted failure/source lexical scoring.
2. `aitech/test_name_anchored_reranker.py`: separate convention-based
   mapping from `FooTests`, `FooTest`, `TestFoo` and `.junit.`
   test packages to likely production classes. This is **not**
   verified causality.
3. `aitech/cpu_failure_rerank_ablation.py`: five real Qwen
   inference arms with frozen pre-gold original sources, new ranked
   candidates, reverse-order controls, raw prompt/response preservation
   and posthoc changed-class Hit@1 scoring.
4. `aitech/cpu_llm_localization.py`: strict parser now accepts a
   single balanced bracketed integer such as `[1]`, but rejects
   multiple choices and unbalanced brackets.
5. Unit tests validate no gold access, deterministic ranking,
   candidate identity/order handling, prompt token ceiling and
   bracketed choice parsing.
6. `.github/workflows/aitech-cpu-failure-rerank.yml`: successful
   one-shot CPU experiment with saved raw data and posthoc scores.
7. `.github/workflows/aitech-stronger-qwen-rerank.yml`: separate
   experiment using a larger model to test model-capability effects.
   Its result is intentionally **not included** until complete.

## Completed 0.5B Qwen experiment

- Model: `Qwen/Qwen2.5-Coder-0.5B-Instruct`
- Two previously studied, pre-gold Defects4J buggy source snapshots:
  `Math-1` and `Chart-1`.
- Prompt uses the same failed tests and original 25-source candidate
  pool. Each arm displays up to 8 L0 method identities.
- Model output: a single selected candidate position, not free-form
  repository localization.
- Same greedy decoding and total prompt ceiling of 1,800 actual
  model tokens.
- Ground truth: official Defects4J **modified class Hit@1**, an
  imperfect fault-causality measure.
- Gold is read only after model replies are saved.

| Pre-gold method ordering | Math-1 class hit | Chart-1 class hit | Total |
|---|---:|---:|---:|
| Original top-8 | 1 | 1 | **2/2** |
| Trace + IDF | 1 | 0 | 1/2 |
| Trace + IDF reversed | 0 | 0 | 0/2 |
| Test-name hint + trace/IDF | 1 | 1 | **2/2** |
| Test-name hint reversed | 0 | 0 | 0/2 |

**All ten outputs parsed successfully.** Nine of ten predictions
picked the first displayed candidate. Reversing the same ranked
candidate set destroyed both class-level hits on both bugs.

The test-name hint restores the baseline but does not improve it;
the generic stack/IDF reranker is worse. Neither can be claimed as
better than the strong original baseline.

## Input cost and timing notes

The real prompt token counts across all ten conditions range from
1,068 to 1,100. Candidate-order comparisons use the same list of
method identities within each normal/reversed pair and have identical
measured total input tokens. Each Qwen CPU generation was around
4.4-4.8 seconds on this CI runner; these are individual observations,
not repeated latency samples.

Ranking and graph-tool CPU costs are not included in that generation
time. Therefore do **not** claim a reduced end-to-end cost.

## Scientific interpretation

- A stronger relevance ranker does not automatically mean better
  LLM debugging. Test-name alignment is often a useful prior,
  but cannot be treated as verified diagnosis.
- The small model appears heavily biased toward prompt position
  in this constrained method-number task.
- Training a marginal benefit/harm controller on a position-biased
  model can teach the wrong objective, namely preserving favorable
  candidate positions rather than adding useful evidence.
- Required controls for the next phase: stronger model, fixed
  candidate identities, balanced order permutations, actual
  input/output tokens, independent bug projects and posthoc labels.
- Retain original top-8 as a strong baseline; do not claim novelty,
  generalization or accuracy improvements from two bugs.

## Reproduce and inspect

Completed experiment:
https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43155635217

The archived ZIP contains:
`frozen_prompts.json`,
`raw_predictions_before_gold.json`,
`posthoc_scores.json`, runtime requirements and logs.

Stronger-model workflow:
https://github.com/ParisaArbab/AdaptiveContextOpt/actions/workflows/aitech-stronger-qwen-rerank.yml

## Research decision

**No-Go for the plain trace+IDF reranker as the proposed improvement.**
**Keep the test-name prior only as an ablation candidate.**
**Go for measuring candidate-position robustness with a stronger model
and larger, project-separated bug samples.**

A credible method improvement requires a held-out class/method Hit@k
and harm/cost advantage over the original strong baseline, not merely
moving a known changed class to position 1.
