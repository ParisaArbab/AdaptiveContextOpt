# AITECH: dependency-preserving debugging context (MVP)

This is the **first small experiment**, not a reproduction of published benchmark results.
It tests whether preserving required evidence links can prevent dangling debugging
context under the *same token budget*.

## Scientific motivation

- **FaR-Loc:** Learn a functionality description from failures and retrieve
  semantically relevant methods. Paper: https://arxiv.org/abs/2509.20552
- **ARISE:** Repository graph and on-demand intra-procedural def-use slicing,
  which improves function-level localization but can consume more tokens than
  unmodified SWE-agent. Paper/code: https://github.com/FARD-Lab/ARISE
- **Referential Dangling:** Independently ranked context fragments can lose
  the support needed to interpret them. Paper/code:
  https://github.com/JusperLee/Referential-Dangling

The MVP uses **explicit dependency edges** and a deterministic greedy selector.
It does **not** yet train ML, query a graph API, perform RAG, or invoke an LLM.
These are future, separately measurable stages.

## Run

From repository root, Python 3.10+:

```bash
python -m aitech.demo
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
```

No external Python packages or API keys are required.

## What the toy example demonstrates

Three evidence units each cost 4 tokens (symbolic token counts).
Budget = 8.

- `failed_test` relevance 10, depends on `call_bridge` relevance 1.
- `unrelated` relevance 5, has no dependencies.
- **Relevance-only** keeps `failed_test` and `unrelated`; the failure
  becomes dangling because the connecting call is absent.
- **Dependency-aware** keeps `failed_test` and `call_bridge`, without
  increasing the budget.

These values are artificial and **do not establish software fault-localization gains**.
The purpose is to verify a measurable selection contract.

## Selector contract

`Evidence(id, text, tokens, relevance)` is one indivisible evidence fragment.
`dependencies[child] = (required_parent_ids,...)` means all listed parents
must be included to retain a self-contained fragment. Requirements are
transitive; cycles and unknown IDs are rejected.

`select_evidence(..., preserve_dependencies=True)` selects a dependency
closure as one bundle. A greedy utility-per-token heuristic picks bundles
until it runs out of budget. This is not an optimal solution to the
dependency-constrained knapsack problem.

`evaluate(...)` computes budget use, dangling count, and optional gold
evidence recall. Gold labels are **evaluation-only**, never available to
the selector. A low dangling count is insufficient by itself: dropping all
evidence also yields zero dangling, so always report recall and coverage.

## Next controlled study

1. Convert actual pytest, SWE-bench, and Defects4J failures into typed
   evidence with stable source IDs.
2. Build dependency edges from stack-trace frames, test-to-function calls,
   and on-demand ARISE def-use slices. Record uncertain/missing edges.
3. Compare raw, relevance-only, dependency-aware, and a learned dependency
   classifier under **matched budgets** and matched downstream LLM.
4. Track File/Function Recall@k, MRR, evidence recall, dependency violations,
   input/output token count, wall time and dollar cost.
5. Train a risk policy only after gathering independent bug-level outcomes,
   splitting by repository to prevent leakage.

## Reproducibility limits

This module deliberately does not vendor ARISE or the Referential Dangling
repo, avoiding upstream license/version confusion. Pin their upstream commits
when conducting full reproductions. Full benchmark integration and actual
trained-policy performance are **not yet verified**.
