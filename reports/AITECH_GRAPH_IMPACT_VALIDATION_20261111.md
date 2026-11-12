# AITECH research checkpoint: Graph coverage is not graph benefit

Date: 2026-11-11
Branch: `AITECH`

## Problem discovered

The previous `validate_real_bug_graph.py` could approve structurally
well-formed experimental outputs even when **no compiler call edges mapped
into the candidate evidence pool**, or when graph-first context selection
was **identical** to anchor-first selection.

A zero-edge graph can be a valid, important negative result. However,
it cannot support a claim that graph retrieval improved selection or
fault localization. Identical selected contexts similarly give no
basis to attribute any downstream change to graph expansion.

## Implemented change

`aitech/validate_real_bug_graph.py` now distinguishes:

1. **Structurally valid:** required source and frozen-candidate hashes,
   non-negative call counts, valid graph node indices, complete and
   unique policy comparisons at matched budgets, and no gold labels
   in selection.
2. **Mapped graph available:** at least one compiler-derived call
   mapped to the frozen candidate pool.
3. **Graph changed selected evidence:** compiler-graph-first selected
   a different set of evidence IDs from anchor-first at one or more
   matched budgets.

None of these states implies a model localized a bug successfully.
They are prerequisites for interpreting a downstream comparison.

The validator rejects missing policies, duplicated methods at one
budget, out-of-range graph indices, over-budget rows and missing
provenance. It still allows structurally complete, zero-edge runs
as **negative or coverage diagnostics**, rather than fabricating graph
effects.

## New artifact

`aitech/report_compiler_graph.py` renders a Markdown report with:

- Compiler call counts and number mapped into ranked candidates
- Ambiguous short-type identities
- Graph availability and whether context selection changed
- Per-budget, per-method retained failure-name anchors
- Missing mapped supporting links
- Explicit limitations on actual localization and token costs

The one-off Lang-1b GitHub workflow now uploads the diagnostic
Markdown together with its raw JSON and setup logs, if extraction
completes.

## Verified status

GitHub unit tests passed for the new validation and reporting code.
A successful **complete Lang-1b real-checkout result** is not
confirmed in this checkpoint, and no external model was run here.

## Scientific next gate

When the real checkout artifact is available, examine the three
status values before deciding what to do:

- Invalid source/compiler output: repair setup, rerun.
- Valid but no mapped graph: measure candidate-pool and type-identity
  recall, improve coverage without looking at patch labels.
- Valid graph that does not change context: document equivalent
  selections, test other budgets or independent bugs.
- Valid graph that changes context: freeze selections and run an
  independent, matched-model fault-localization evaluation.

Real accuracy gains require a measured downstream run, not simply
compiler-based edges or green CI badges.
