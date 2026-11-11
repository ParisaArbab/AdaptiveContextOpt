# AITECH Experiment: Anchor-first selection under dependency budgets

Date: November 11, 2026

## Hypothesis

A budget-aware policy which protects failing-test-named methods before
optional dependency expansion retains more diagnostic anchors than full
dependency closure, especially where closure forms expensive bundles.

## Implementation

- `aitech/anchor_budget.py` prioritizes candidate methods named in
  failing-test evidence, then adds affordable supporting methods.
- Unresolved edges remain explicit, not silently marked complete.
- `disambiguate_lexical_edges` excludes call edges whose callee name
  is overloaded in the same class. This is a conservative *lexical*
  filter, **not** Java AST or true call resolution.
- `aitech/real_bug_closure_stress.py` adds `anchor_first_audited`
  alongside existing rank fill, dependency closure and naive reservation.

## Inputs

Two archived real Defects4J *development* cases, Lang-1 and Time-1,
from `results/rave_fl_iteration_d/independent_live/{bug}/input_snapshot.json`.
Each contains 25 method candidates, failing-test context, and L1 sketch
token estimates. Input candidate rankings were fixed before this analysis.
No gold patch labels were consulted.

## Measured anchor retention

| Bug | Budget (L1 proxy) | Rank fill | Closure | Naive reserve | Anchor-first |
|---|---:|---:|---:|---:|---:|
| Lang-1 | 200 | 1/2 | 1/2 | 1/2 | 1/2 |
| Lang-1 | 300 | 1/2 | 1/2 | 1/2 | 1/2 |
| Lang-1 | 450 | 1/2 | 1/2 | 1/2 | 1/2 |
| Lang-1 | 600 | 1/2 | 1/2 | 1/2 | 1/2 |
| Time-1 | 200 | 1/2 | 0/2 | 0/2 | 1/2 |
| Time-1 | 300 | 0/2 | 0/2 | 0/2 | 1/2 |
| Time-1 | 450 | 1/2 | 0/2 | 0/2 | 1/2 |
| Time-1 | 600 | 1/2 | 0/2 | 0/2 | 1/2 |

At Time-1 budgets 200, 300, 450 and 600, Anchor-First used
177, 287, 443, and 596 proxy tokens, all within budget.

## Graph audit

Lang-1 had 8 original same-owner lexical edges, with 2 edges rejected
for ambiguous overloaded names. Time-1 had 6, with 4 rejected.
At Time-1 budget 600, Anchor-First kept one failure-related anchor
but had **two unresolved non-ambiguous lexical support edges**.

Therefore the new method improves failure-name anchor retention in this
small development case, but does not guarantee complete graph support.

## Scientific limits

- Only two previously examined bugs, no independent test set.
- Failure-name anchors are an indirect heuristic for useful evidence,
  not verified fault locations.
- Original edges came from lossy L1 source sketches; overloaded-name
  filtering is *not* fully verified Java method binding.
- The number of kept anchors is not file/function Recall@k.
- No trained ML, LLM rerun, wall-clock profiling or actual model
  tokenizer evaluation was performed.
- Comparisons use source sketch token estimates, not Ollama prompt costs.

## Reproduce

```bash
git checkout AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.real_bug_closure_stress \
  --output results/aitech/anchor_first_real_bug_stress.json
```

## Next test

Build reliable code-level call edges with Java parsing and method
signature information. Then test on an independent Defects4J project,
compare file and method Recall@k after actual LLM localization, and
optimize jointly for both anchor preservation and support completeness.
