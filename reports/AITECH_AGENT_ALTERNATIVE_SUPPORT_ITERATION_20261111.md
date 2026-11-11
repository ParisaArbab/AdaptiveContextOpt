# AITECH Research Agent Iteration: Alternative Support and Provenance Gates

Date: November 11, 2026
Branch: `AITECH`

## Scientific objective

Advance the paper-inspired pipeline toward a testable link between
FaR-Loc-style retrieval, ARISE-like graph tools and Referential Dangling
support completeness, WITHOUT assuming that graph adjacency is enough
to explain a software failure.

**Critical novelty limit:** OR-of-AND sufficient supporting sets already
appear in the Referential Dangling paper. Implementing them by itself is
not a new method. AITECH would need to learn when a real debugging
evidence path is sufficient and when its token cost justifies retrieval.

## Completed engineering work

- `aitech/alternative_support.py`:
  exact controlled solver for up to 18 evidence fragments, with
  alternative sufficient bundles and grounded support.
- `aitech/alternative_support_ablation.py`:
  matched artificial budgets 8, 10, 12, 14, 16 and three policies.
- `aitech/support_provenance.py`:
  audit gate that prevents unverified compiler edges from becoming
  assertions of causal sufficiency.
- `aitech/compiler_support_proposals.py`:
  connects compiled Java method edges to candidates that still require
  an explicit sufficiency audit.
- `aitech/support_annotation_schema.py`:
  validates real bug annotations and enforces project-disjoint
  train/calibration/test splits. No synthetic examples are exported as
  model-training data.
- Tests cover alternative paths, nested requirements, circular-only
  support, budget checks, no false compiler-edge promotion and
  project-level leakage.
- The push-triggered AITECH CPU workflow archives the controlled
  comparison JSON. No research cron/schedule was added.

## Controlled outcome

At illustrative budget 8, relevance-only retained a failure with
missing support; mandatory union closure dropped the failure because
it could not afford all alternative paths; exact sufficient-path
selection kept the failure plus one supporting call bridge.

The result distinguishes formal support definitions and budget effects.
It is **not** a result from a real Defects4J bug or actual model inference.

## Why this matters

We now have a safety contract that separates four facts:

1. A real Java source declaration exists.
2. Javac resolves a static call target to that declaration.
3. That call is a *candidate* explanation/support path.
4. An independently audited full path is sufficient to interpret
   the particular failure.

Only (4), with proper ground evidence, can be used to train or
evaluate a support-completeness predictor.

## Next falsifiable experiment

1. Collect 15-30 real failed-test cases from multiple independent
   Defects4J projects, preserving the buggy source revision and
   official failure output.
2. Construct retrieval candidates using actual failure evidence,
   compiler-bound methods where available, and a fixed semantic
   relevance baseline.
3. Audit a sample of alternative support paths for sufficiency
   BEFORE accessing any gold patches; preserve negatives.
4. Fit a small classifier for `P(sufficient support | failure, evidence path)`
   on project-separated training bugs and calibrate on separate projects.
5. Compare evidence packing under the same model-token budgets:
   relevance-only, forced closure, one-sufficient-path heuristic,
   calibrated path selector, and a STOP/abstain policy.
6. Freeze LLM predictions before gold scoring. Report method/file
   Hit@1/5, MRR, harmful expansion, actual model input/output tokens,
   latency and calibration metrics.
7. Evaluate on untouched projects. A credible positive finding must
   improve downstream localization or its accuracy-cost frontier,
   not just the number of retained edges.

## Honest status

- No trained support classifier or adaptive controller.
- No executed Qwen fault-localization result produced in this iteration.
- No ARISE def-use slice runtime or FaR-Loc UniXcoder/FAISS pipeline.
- No verified real-bug sufficiency annotations yet.
- The work is a foundation and a controlled pilot, not a publishable
  novel fault-localization algorithm by itself.

## Reproduction

```bash
git switch AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.alternative_support_ablation \
  --output results/aitech/alternative_support_microcase.json
```

These are run-on-push and on-demand commands only. No scheduling.
