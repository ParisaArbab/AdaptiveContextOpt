# AITECH research agent: live CPU pilot, 11 November 2026

## Research question

Does simple dependency closure preserve more useful debugging evidence than
relevance-only ranking at a matched context budget?

## Sources and execution

Two **real archived development** Defects4J failures, Lang-1 and Time-1.
Each has 25 independently ranked buggy Java methods and the saved failing-test
context at `results/rave_fl_iteration_d/independent_live/{Lang-1,Time-1}/input_snapshot.json`.

I tested three CPU selector policies at stored **L1 sketch-token proxy** budgets
200, 300, 450, 600: rank fill, closure of same-owner lexical calls, and naive
failure-anchor reservation before closure. These call edges are heuristic and
**not a validated Java call graph**. No patch gold, trained model, or LLM was
used. See `python -m aitech.real_bug_closure_stress`.

## Selected measures

| Bug | Budget | Rank fill: anchors kept | Closure: anchors kept | Rank dangling | Closure dangling |
|---|---:|---:|---:|---:|---:|
| Lang-1 | 200 | 1/2 | 1/2 | 0 | 0 |
| Lang-1 | 300 | 1/2 | 1/2 | 0 | 0 |
| Lang-1 | 450 | 1/2 | 1/2 | 0 | 0 |
| Lang-1 | 600 | 1/2 | 1/2 | 0 | 0 |
| Time-1 | 200 | 1/2 | 0/2 | 1 | 0 |
| Time-1 | 300 | 0/2 | 0/2 | 0 | 0 |
| Time-1 | 450 | 1/2 | 0/2 | 1 | 0 |
| Time-1 | 600 | 1/2 | 0/2 | 1 | 0 |

**Finding:** With Time-1, dependency closure eliminated known missing links
but also dropped a method named in the failing-test evidence. A zero
dangling count does **not** imply high failure-evidence retention. Naive
anchor reservation performed like closure because the required bundle
was too large.

## Reproducibility and limitations

```bash
git switch AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.real_bug_closure_stress \
  --output results/aitech/real_bug_closure_stress.json
```

Only two previously seen development bugs, no held-out test and no
downstream fault-localization accuracy measurement. The evidence anchor
definition is a lexical test/failure-name heuristic rather than a gold
root-cause label. Naive same-class call matching can mishandle overloaded
methods and introduce false cycles. Token values are precomputed L1 sketch
proxies, not actual LLM input token counts.

## Next independent research step

Resolve calls using Java AST and method signatures; inspect cycles, distinguish
mandatory failure anchors from optional dependency edges, and test
anchor-preserving, budget-constrained selection on different projects.
Train an adaptive policy only after independent outcome data exists.

## Related primary research

- [FaR-Loc, functionality-aware retrieval](https://arxiv.org/abs/2509.20552)
- [ARISE, repository graph and data-flow slicing](https://arxiv.org/abs/2605.03117)

Research Compass saved artifacts:
`aitech-closure-stress-20261111-results.json` and
`aitech-closure-stress-20261111-report.md`.
