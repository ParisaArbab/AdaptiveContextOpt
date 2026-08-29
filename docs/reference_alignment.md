# Reference-code alignment

Three external repositories were reviewed before rebuilding `main`.

## Graphify

Reference: `Graphify-Labs/graphify`

The integration uses the real `graphify` CLI:

```bash
graphify extract <defects4j-checkout> --code-only --no-viz
```

The generated `graphify-out/graph.json` is converted to a compact index for repository paths, classes, methods, and source snippets.

## FlexFL original replication

Reference: `ParisaArbab/FlexFL_OriginalReplication`

Important behavior copied from the replication:

1. Agent4SR performs search-space reduction.
2. Traditional FL contributes top candidates from SBIR, Ochiai, and BoostN.
3. `combine.py` merges 5 SBIR + 5 Ochiai + 5 BoostN + 5 Agent4SR, at most 20 entries.
4. Agent4LR inspects only the merged candidate methods and produces a final top 5.
5. Evaluation uses Defects4J method-level ground truth and Top-1, Top-3, Top-5, MAP, and MRR.

The original Agent4SR tool idea is kept. The implementation exposes class, method, path, and code-snippet searches, but its source index now comes from Graphify and the actual Defects4J checkout.

## LeanCTX

Reference: `yvgude/lean-ctx`

The relevant upstream tool is `ctx_compare`. Its source states that shell previews call the same production compressor used by `ctx_shell`. It accepts:

```text
command=<cmd> + output=<captured text>
```

and reports original/compressed token counts plus a line diff. The benchmark reconstructs the compressed bytes from that diff. This means it does not duplicate or approximate LeanCTX's compression rules.

This design is useful for a controlled RAW vs LeanCTX experiment because the test command only needs to run once.
