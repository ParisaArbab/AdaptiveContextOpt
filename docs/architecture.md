# Main branch architecture

This branch implements the Defects4J method-localization pipeline used for the context-compression experiment.

```text
Defects4J bug instance
        |
        v
Preparation
  - checkout buggy version
  - Graphify code graph
  - run defects4j test once
        |
        +-------------------------+
        |                         |
        v                         v
RAW test output             LeanCTX output
(no reduction)              real shell compressor
        |                         |
        +------------+------------+
                     |
              run per model
                     |
                     v
                 Agent4SR
               top 5 methods
                     |
                     +-------------------------------+
                     |                               |
                     v                               v
          Traditional FL results                 Agent4SR
        5 SBIR + 5 Ochiai + 5 BoostN             top 5
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
                         merge up to 20 methods
                         original FlexFL order
                                     |
                                     v
                                  Agent4LR
                              rerank candidates
                                     |
                                     v
                             final ranked top 5
                                     |
                                     v
                     Top-1, Top-3, Top-5, MAP, MRR
```

## Important experiment rule

`defects4j test` is executed once for a bug. Its captured output is used directly by the RAW arm. The same captured bytes are passed to LeanCTX `ctx_compare` using `command + output`. This calls LeanCTX's production shell compressor without rerunning the test.

This prevents test nondeterminism from becoming a fake compression effect.

## What Graphify does

Graphify is a preparation layer. It parses the actual Defects4J checkout and gives Agent4SR and Agent4LR a repository structure and source lookup layer. RAW and LeanCTX use the same Graphify graph.

Graphify is not used as an extra localization model, and GraphLocator is not part of this main pipeline.

## What FlexFL contributes

The project uses the original replication for:

- Agent4SR and Agent4LR workflow design
- SBIR top 5
- Ochiai top 5
- BoostN top 5
- trigger-test text
- bug-report text when available
- Defects4J method-level ground truth
- evaluation conventions

The merge order follows the original `combine.py`: SBIR, Ochiai, BoostN, Agent4SR. The original code appends candidates without deduplication, so this implementation keeps that behavior.

## What LeanCTX does

LeanCTX is not approximated in Python. `wp1/leanctx_compressor.py` invokes the installed `lean-ctx` binary and calls `ctx_compare` with the captured command output. `ctx_compare` uses LeanCTX's production shell-compression engine.

If LeanCTX is missing or fails, the LeanCTX condition fails. There is no silent heuristic fallback.
