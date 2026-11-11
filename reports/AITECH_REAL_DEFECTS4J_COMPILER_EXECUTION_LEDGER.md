# AITECH Real-Repository Compiler Experiment: Execution Ledger

**Date:** 2026-11-11
**Branch:** `AITECH` (default)

## Goal

Use a *complete buggy Defects4J checkout*, rather than archived fragments,
to create a compiler-attributed Java static call graph and compare evidence
selection policies under matched token-proxy budgets.

## Implemented and tested

1. `ResolvedCallExtractor.java`: optional compile classpath forwarded to
   `JavacTask` semantic analysis.
2. `java_compiler_graph.py`: exposes this classpath to Python.
3. `defects4j_compiler_pilot.py`: validates pre-existing buggy checkout,
   exports `dir.src.classes` and `cp.compile`, records source and metadata
   SHA-256, refuses source trees above an explicit file limit, and fails on
   compiler attribution errors.
4. `verified_graph_evidence.py`: links compiler-declared static targets
   to uniquely matched candidate evidence IDs. It rejects ambiguous
   caller identities and uncertain overload matches.
5. `compiler_graph_ablation.py`: compares rank fill, anchor-first, and
   compiler-graph-first using identical candidate pools and budgets.
6. Unit tests exercise external library classpath resolution, errors,
   ambiguous overload handling, and matched-budget selection.

## Real Defects4J experimental workflow

`.github/workflows/aitech-defects4j-compiler.yml` performs one
**separate, one-time push-triggered** experiment:
- installs prerequisites on a GitHub-hosted CPU runner;
- checks out pinned Defects4J framework commit
  `8c16da8230843cdc918eaf4ddb449637f02b83c6`;
- initializes Defects4J and checks out real `Lang-1b`;
- exports source metadata and compile classpath;
- executes compiler-backed extraction and freezes the candidate snapshot;
- saves raw logs and JSON results through a GitHub Actions artifact,
  including when setup fails.

This is not part of the daily research workflow because Defects4J
initialization downloads large external project dependencies. CI setup
may fail due to network, legacy Java build compatibility, or unavailable
dependencies, in which case its logs are the result, **not a success claim**.

## Verification status

The **unit-test workflow passed** for the compiler graph and
classpath test changes. A complete Lang-1b compiler graph **has not yet
been confirmed from the one-time GitHub Actions artifact**. Do not cite
a graph edge count, model-token saving, or fault-localization result
until that run completes and its output is inspected.

## Scientific validity

- All ranked evidence is frozen before any gold patch scoring.
- This stage does not use gold labels, make an LLM call, or generate patches.
- Java compiler targets are *static declaration bindings*, not necessarily
  runtime targets under polymorphism and reflection.
- Candidate signature normalization, especially imported simple type names,
  is a conservative approximate join, separately audited.
- Java version or missing classpath dependencies cause explicit failure,
  not a silently incomplete graph.

## Next gate

If real Lang-1b completes: audit accepted edges and method mappings.
If it fails: use the archived setup diagnostics to fix only the
environment, not the experimental selection metrics. Then run a matched
baseline versus graph-informed localization on multiple independent bugs
with the same actual LLM tokenizer, prompts, model, and inference setup.
