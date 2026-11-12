# AITECH: Failure-grounded retrieval and Qwen 1.5B controlled study

Research date: November 11, 2026 (America/New_York)
Branch: `AITECH`

## Research objective

Address failures in early Qwen 0.5B source-detail experiments:
a single L0-to-L1 upgrade was often neutral and the model was sensitive
to the position of candidate methods. Test a larger CPU model,
failure-conditioned source reranking and a no-training STOP fallback.

**Scope:** previously studied Math-1 and Chart-1 development bugs.
No held-out generalization or demonstrated novel ML controller.

## Methods implemented

1. `aitech/failure_grounded_rerank.py`: deterministic BM25-like
   relevance over source snippets, plus stack-frame matching and weak
   test-name-to-production-class hints. No patch labels, embeddings,
   or learned fault probabilities are used. This is not FaR-Loc.
2. `aitech/grounded_rerank_cpu_ablation.py`: equal eight-method
   candidate pools and the same input-token ceiling across four arms:
   existing original order, new failure-grounded order, reversed new
   order as position-bias control, and new order with one L1 upgrade.
   All raw model responses are saved before Defects4J changed-class
   labels are loaded.
3. Model `Qwen/Qwen2.5-Coder-1.5B-Instruct`, CPU float32, greedy
   decoding, seed 42, up to 24 new tokens, model-tokenizer
   input ceiling 1800 for every arm.
4. `aitech/grounded_stop_guard.py`: **untrained heuristic** that
   elects STOP if the selected methods already contain a class seen
   in a production trace or named by a failing test (e.g. FooTests ->
   Foo). This is an uncertain clue, not a verified root-cause edge.
   Without such a clue it proposes L1 upgrade, if feasible.

## Verified result 1: Stack/BM25 alone did not improve class Hit@1

The first experiment (without test-class hint) on Math-1 / Chart-1:
both the original order and the proposed grounded reranking predicted
a changed class on both bugs, **2/2 Hit@1**, but the grounded shortlist
used 10 and 50 fewer model input tokens, respectively.

This does NOT establish superior retrieval; method-level answers
could differ while class-level Hit@1 ties.

Source:
https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43150737850

## Verified result 2: Test-name clue repaired one harmful upgrade

The same model and source candidate pool were compared after enabling
the weak pre-gold test-name class clue:

| Bug | Original L0 class hit | Grounded L0 hit | Initial grounded L1 hit | Revised grounded L1 hit |
|---|---:|---:|---:|---:|
| Math-1 | 1 | 1 | 1 | 1 |
| Chart-1 | 1 | 1 | 0 | 1 |

- Initial Chart-1 L1: the model predicted a method in
  `org.jfree.chart.plot.PolarPlot`, NOT the changed class.
- Revised Chart-1 L1: the model predicted a method in
  `org.jfree.chart.renderer.category.AbstractCategoryItemRenderer`,
  which IS a changed class.
- Initial / revised Chart-1 L1 actual input tokens: **1177 / 1143**.
- Revised grounded L0 tokens: Math-1 **1077**, Chart-1 **1069**.
- Revised grounded L1 tokens: Math-1 **1381**, Chart-1 **1143**.

So on these *development cases* the revised L1 arm went from 1/2
changed-class Hit@1 to 2/2. This improvement was motivated by the
observed Chart-1 failure, so the cases are NOT held-out tests and no
independent accuracy gain is claimed.

Source:
https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43151088416

## Important observation: order sensitivity

Reversing exactly the same eight grounded method IDs changed the
chosen method on both bugs. In the revised Chart-1 study, it also
changed the chosen class to `BarRenderer3D`, an unchanged class.
The model is therefore not position-invariant. Reporting only the
best candidate ordering would give a misleading result.

The STOP heuristic avoids potentially unnecessary L1 upgrades, but
has NOT been shown to improve held-out accuracy. Reading source
clues can still be misleading, and STOP may prevent a helpful upgrade.

## New independent-project test

A one-shot workflow tests `Codec-1` and `Compress-1`, previously
unused in these AITECH pilot comparisons, with identical Qwen 1.5B
conditions. It records **the guard's choice before inference or
changed-class scoring**. It then executes both the STOP and forced
upgrade options to estimate the action's actual counterfactual outcome.

Workflow:
https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43151606993

**Do not interpret the run until its raw_predictions_before_gold.json
and scored_after_gold.json artifacts are verified.**

## Limitations

- Java method extraction currently uses a regex/brace heuristic,
  not verified full AST plus compiler target identities.
- The candidate-choice task only ranks eight of 25 pre-retrieved
  source methods, not the full repository.
- Official modified classes are *patch proxies*, not causal root
  cause or line-level localization labels.
- Qwen 1.5B can remain position-biased despite higher capacity.
- No full ARISE data-flow slicing, FaR-Loc embedding retrieval or
  trained Referential Dangling detector has been integrated.
- The fallback is deterministic and uncalibrated. A learned
  positive/harm/neutral action-value model is still waiting for
  sufficient independent true paired outcomes.

## Scientific acceptance criteria

An improvement is credible only when matched-model and matched-budget
test-project evaluations demonstrate a better localization/cost frontier
versus original ordering, grounded L0, always upgrade, and always STOP.
Report all raw inputs, predictions, selected source, abstentions,
invalid outputs, CPU time, tokens, class/method hits and order stability.
Do not tune the final test-project policy after observing its gold.
