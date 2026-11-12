# AITECH: Model-Free Evidence-Confidence STOP Pilot

Date: 2026-11-11 (America/New_York)
Branch: `AITECH`
Outcome: **two development bugs, two class hits, zero LLM inference tokens**.
Status: exploratory heuristic baseline, **not** a trained ML policy,
independent validation, or proof of generalization.

## Motivation

In the measured 0.5B Qwen ablation, the LLM chose candidate 1 in
nine of ten prompts. Reversing the same candidate order changed
correct answers into incorrect ones. This means the 0.5B model did
not consistently justify its inference cost on these cases.

A necessary comparison is therefore a model-free early-stop baseline:
**Can a source-grounded, pre-gold failure/test-name hint identify a
reasonable class without invoking the LLM at all?**

## Method

- Extract the buggy-revision candidate methods and failure traces
  using `aitech/live_defects4j_snapshot.py` (source snapshots frozen
  in the earlier real Defects4J experiment).
- Score candidate methods with stack-frame agreement and IDF-weighted
  failure/source lexical similarity
  (`aitech/failure_aligned_reranker.py`).
- Parse conventional failing test class names such as
  `FooTest`, `FooTests`, `TestFoo`, or
  `pkg.junit.FooTests`. Map a name to a *candidate production class*
  using naming conventions only. This is not compiler verification
  or causal attribution (`aitech/test_name_anchored_reranker.py`).
- Return `STOP_CLASS_HINT` with the best candidate class only when
  the test-name mapping exactly matches the full source-owner identity.
- Otherwise return `ESCALATE` with NO fabricated LLM result.
- `aitech/model_free_stop_gate.py` implements this transparent
  decision. It does not output learned probabilities or claim calibrated risk.
- Freeze all decisions to JSON before loading official Defects4J
  changed-class labels; score posthoc with
  `aitech/evaluate_stop_gate.py`.

## Actual CPU study on previously examined real buggy revisions

- Data: `Math-1` and `Chart-1` pre-gold 25-method snapshots,
  created from genuine Defects4J buggy checkouts.
- Controller uses ONLY the failing test, trace and source candidate list.
- Reference metric: official `classes.modified` changed-class Hit@1.
  This is **not** verified root-cause localization.
- Main gate result, directly observed from one-shot workflow:
  `STOP 2 of 2 bugs; correct changed class 2`.
- LLM input and output tokens: **zero**, because no inference was
  performed for the STOP decisions.
- Source extraction, reranking and workflow CPU time are NOT
  zero and were not fully measured in this pilot.

| Policy | Math-1 | Chart-1 | Total class hits | LLM calls |
|---|---:|---:|---:|---:|
| Original source order, model-free first rank | 1 | 1 | 2/2 | 0 |
| Trace/IDF model-free first rank | 1 | 1 | 2/2 | 0 |
| Exact test-name-gated model-free STOP | 1 | 1 | 2/2 | 0 |
| Small Qwen with original top 8 | 1 | 1 | 2/2 | 2 |
| Small Qwen with trace/IDF rank | 1 | 0 | 1/2 | 2 |

**Observed improvement**: relative to the two-bug Qwen trace/IDF
arm, zero-inference rank achieves one extra changed-class hit and
removes all LLM input/output tokens. Relative to the strong
Qwen original top-8 arm, class Hit@1 is **tied** and only
LLM inference cost is avoided.

This is a specific two-bug exploratory result. It is **not**
evidence that model-free STOP is generally superior or safe.

## Artifacts

- Five-arm real Qwen study:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43155635217
- Model-free ranking study:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43156370842
- STOP-gate verification and output:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43156663087
- Larger 1.5B Qwen independent capability ablation:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43155966060
  (report only once verified completed)

## Methodological decision

Do NOT promote the gate as a trained adaptive ML controller.
It has no empirical calibration, false-STOP risk estimate, retrieval
CPU timing, method-level correctness guarantee or held-out evaluation.

Most important future evaluation:
1. On many project-disjoint Defects4J bugs, measure changed class,
   function and eventually true causal-root Hit@1/5 and MRR.
2. Report **coverage of STOP**, **accuracy conditional on STOP**,
   **wrong-STOP fraction**, **escalation frequency**, actual tokenizer
   input/output and end-to-end source-retrieval/inference latency.
3. Evaluate cases in which test names are misleading or do not
   correspond to defective classes, and where multiple classes share
   short names.
4. Train and calibrate the margin/risk model only on real
   independent-project before/after outcomes, including harmful
   and helpful evidence actions. The policy should abstain
   when confidence is low.
5. Require superiority to the strong original ranking and
   model-free first-rank baselines; otherwise report no novelty.

**Conclusion**: This pilot makes a concrete, falsifiable efficiency
case for a pre-LLM STOP option. It does not establish new accuracy,
causal localization or generalizable model improvement.
