# AITECH: Arity-aware Java call audit, November 11, 2026

## Research question

Can matching explicit `CALL` records by method name, owner, and argument
count reduce questionable code dependency links while protecting failure
anchors under a fixed evidence budget?

## Method

A new `aitech/java_call_audit.py` reads existing L1 method sketches.
It parses `CALL` entries, counts nested arguments, and accepts only a
unique target with the same method owner, name and parameter count.
Receiver variables, unresolved names, and same-arity overloads are
**not inferred**. This is *not* a compiler-resolved Java call graph.

The fifth condition `anchor_first_arity` retains ranked methods named
in failing-test evidence first and then adds affordable supports from
the arity-audited graph. All baseline conditions use the same archived
candidate pool and stored token-proxy budgets.

## Observed development results

| Measure | Lang-1 | Time-1 |
|---|---:|---:|
| Candidate methods | 25 | 25 |
| Explicit CALL lines examined | 98 | 131 |
| Accepted arity-aware same-owner edges | 7 | 0 |
| Ambiguous matching call entries | 0 | 0 |
| Unresolved eligible call entries | 4 | 13 |

| Bug | Budget proxy | Anchor-first arity kept | Approx. sketch tokens used | Audited missing edges |
|---|---:|---:|---:|---:|
| Lang-1 | 200 | 1/2 | 196 | 0 |
| Lang-1 | 300 | 1/2 | 299 | 0 |
| Lang-1 | 450 | 1/2 | 443 | 0 |
| Lang-1 | 600 | 1/2 | 569 | 0 |
| Time-1 | 200 | 1/2 | 177 | 0 |
| Time-1 | 300 | 1/2 | 287 | 0 |
| Time-1 | 450 | 1/2 | 443 | 0 |
| Time-1 | 600 | 1/2 | 596 | 0 |

These are a replay of the deterministic selector against real archived
source evidence, not an independent held-out localization benchmark.
The previous naive same-owner lexical graph produced eight links for Lang-1
and six for Time-1. The stricter parser retained seven and zero links.
Those differences must **not** be interpreted as improved correctness:
the missing Time-1 edges may reflect false negatives caused by incomplete
receiver/type information and constructor overloads.

## Negative findings

- No verified same-owner links were found for Time-1 from available sketches.
- Having no selected missing edges is vacuous when no edges were found.
- Method name + arity still cannot distinguish two overloads with identical
  argument counts but different parameter types.
- Anchor retention did not improve over the previous anchor-first
  strategy in this small sample.
- We have no actual downstream fault localization result for this method.

## Reproduce

```bash
git switch AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.real_bug_closure_stress \
  --output results/aitech/arity_call_stress.json
```

## Next research step

Use Java compiler AST/symbol bindings or tree-sitter plus type-aware
resolution on a pinned repository checkout. Label edges as verified,
ambiguous, or unresolved and compare graph-edge precision/recall on a
small manually audited sample. Then test candidate retrieval and
downstream localization on independent bugs at equal model-token budgets.

Sources: archived repository snapshots:
`results/rave_fl_iteration_d/independent_live/{Lang-1,Time-1}/input_snapshot.json`.
The source candidates predate this new method. No gold patch labels or
LLM inference were accessed.
