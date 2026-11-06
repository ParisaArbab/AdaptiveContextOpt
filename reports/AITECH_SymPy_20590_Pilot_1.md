# AITECH Pilot 1: Real SymPy debugging trace (offline microcase)

**Instance:** `sympy__sympy-20590` (SWE-bench Lite)
**Historical artifact:** `data/swebench_workspaces/sympy__sympy-20590/outputs/compression_tax/agent4sr_qwen_COMPRESSION_TAX.log`
**Code:** `python -m aitech.sympy_trace_pilot`

## Objective

Test a narrow mechanical property: when the same small budget is available,
can dependency-aware selection keep the connection between a code statement
and its required import while a relevance-only selector drops it?

This is not yet a full debugging benchmark run, and does not call an LLM.

## Experiment

A seven-line real-source excerpt is extracted from the archived Agent4SR trace.
It contains a test assertion, test function, object initialization, Basic
class declaration, Printable import, and two unrelated imports.

The pilot manually defines three explicit source-level dependency relations:

- test assertion needs the test function and object initialization;
- object initialization needs the Basic class declaration;
- the Basic class declaration references Printable, which is imported from
  `._print_helpers`.

**Caveat:** These are known syntactic support relationships, not proven causal
links for the bug. They are also incomplete relative to the whole repository.

Both arms use the **same** greedy selector, input items, manually assigned
relevance weights, and regex-based token proxy. The only difference is whether
the dependency closure must be selected together.

## Recorded result

| Metric | Relevance-only | Dependency-aware |
|---|---:|---:|
| Selected evidence units | 5 | 5 |
| Proxy tokens selected | 35 | 35 |
| Budget (proxy tokens) | 36 | 36 |
| Known dangling selections | 1 | 0 |
| Includes Printable import | No | Yes |
| Actual LLM localization accuracy | Not measured | Not measured |

At the same 35 proxy-token cost, dependency-aware selection substitutes the
relevant Printable import for an unrelated import. This directly verifies one
dependency-preservation behavior on source lines captured from the real SymPy
debugging trajectory. It **does not** demonstrate improved fault localization.

At a tighter 31-token budget, the dependency-aware selector can avoid a
dangling reference by omitting the test assertion entirely. Therefore,
**zero dangling links does not imply sufficient evidence**.

The historical experiment already in the repository reports that Raw Agent4SR
hit `sympy/core/_print_helpers.py::Printable` at rank 1 while compressed
Agent4SR missed it in Top-5, but this result is not attributable to AITECH.
The final Agent4LR stage showed the reverse ordering in that archived run.
These historical rankings must not be reported as new experimental improvements.

## Reproduce

```bash
git checkout AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.sympy_trace_pilot \
  --trace data/swebench_workspaces/sympy__sympy-20590/outputs/compression_tax/agent4sr_qwen_COMPRESSION_TAX.log \
  --output /tmp/aitech-sympy-trace.json
```

Primary recorded result:
`results/aitech/sympy20590_trace_microcase.json`.

## Required next test (not yet done)

1. Extract **all** relevant runtime and source evidence from the bug, not
   just seven manually selected lines.
2. Generate dependency edges using real AST/call graph/ARISE data-flow analysis
   and audit them before selection; record false and missed edges.
3. Run the **same** localization LLM on matched-budget contexts under
   relevance-only and dependency-aware selection, with gold labels hidden.
4. Compare file/function ranking, MRR, token counts using the actual LLM
   tokenizer, runtime, and costs. Report both fixes and regressions.
5. Repeat across independent bugs and repos before claiming an improvement.

**Conclusion:** AITECH passed a meaningful *evidence-link preservation*
microcase, but the important claim, better bug localization at no extra
context cost, is still untested.
