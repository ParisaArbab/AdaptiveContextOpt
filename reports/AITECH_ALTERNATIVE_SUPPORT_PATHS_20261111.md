# AITECH: Sufficient Alternative Evidence Paths, Controlled Research Pilot

Date: 2026-11-11
Branch: `AITECH`

## Question

When a retained test failure has several *alternative sufficient*
support explanations, does treating all edges as mandatory waste budget?

The Referential Dangling paper motivates requiring at least one
sufficient support set, not necessarily every possible support path.
Existing AITECH `evidence.py` represented all dependencies as
simultaneously required. This pilot introduces a small exact solver
that separates OR-of-AND alternatives from all-required graph closure.

**This is an implementation and controlled ablation of an existing
theoretical idea, not a new research contribution or a reproduction of
the paper's learned BERT detector.**

## Mechanism

For evidence `failure`, suppose the possible sufficient sets are:

- `{call_bridge}`
- `{function_definition, assertion_context}`

Retaining either *entire* set is enough under the manually defined
support specification. The model does not need to include both paths.

The selector enumerates all subsets for at most 18 evidence items,
optimizes sum of non-oracle relevance at fixed proxy-token budget,
and only accepts subsets with a grounded support witness.
Unsupported circular explanations such as `a requires b` and
`b requires a` are rejected. The exhaustive method is a
small-instance exact reference, not a scalable retrieval algorithm.

## Experiment design

Five artificially priced evidence items: failed assertion (4 tokens,
relevance 10), trace bridge (4, 1), source definition (3, 0.6),
assertion context (3, 0.6), unrelated warning (4, 5).

Three policies:

1. Relevance-only greedy selection, may preserve unsupported anchors.
2. Original all-edges-required greedy closure, overrestricts
   alternative support evidence.
3. New exact one-sufficient-path selector, may choose either
   alternative under the same budget.

Budgets: 8, 10, 12, 14, 16 artificial proxy tokens.

## Controlled finding, budget 8

| Selector | Failure retained | Support complete | Interpretation |
|---|---|---|---|
| Relevance-only | Yes | No | Failed assertion can remain dangling |
| All-edges-required | No | Yes (vacuously) | Cannot afford the union of all alternatives |
| Exact sufficient-path | Yes | Yes | Retains failure plus its trace bridge |

At budget 8, the exact reference retains `failed_assertion` and
`trace_bridge`, costing 8 artificial units and relevance 11.
The original union of possible supports costs 14 including failure
and cannot fit at this budget.

This result demonstrates a **structural difference** between two
formal support models under controlled assumptions. It does not show
that either support path is actually sufficient for diagnosing a
real Java failure, nor that an LLM would localize more bugs.

## Implementation

- `aitech/alternative_support.py`: OR-of-AND support, grounded
  completeness check, exact <=18-item selection.
- `aitech/alternative_support_ablation.py`: deterministic 15-row
  baseline comparison across five budgets.
- `tests/test_aitech_alternative_support.py`: support alternatives,
  transitive obligations, circular-only references, input checks.
- `tests/test_aitech_alternative_support_ablation.py`: budget and
  outcome sanity checks.

Run:

```bash
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.alternative_support_ablation \
  --output results/aitech/alternative_support_microcase.json
```

## Limits

- All support alternatives are **manually authored** for a synthetic case.
  They are not inferred from FaR-Loc, ARISE or real failure records.
- Evidence is indivisible and its token costs are artificial.
- Exact search is exponential and restricted to tiny instances.
- High relevance does not imply bug causality or LLM usefulness.
- No classifier was trained, no actual LLM inference took place,
  no independent Defects4J accuracy was measured.
- This is **not** a novel sufficient-set concept; novelty, if any,
  must come from reliable source-grounded inference and calibrated
  cost-sensitive decisions across real projects.

## Scientific next step

Use compiler-attributed Java call targets and verified test/trace
provenance to propose *candidate* support paths, clearly label
uncertain links, and manually audit sufficiency on a small held-out
real-bug set. Compare the exact small-instance ceiling against a
scalable, uncertainty-aware policy, and measure actual matched-model
localization changes before training a marginal-harm controller.
