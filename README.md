# AdaptiveContextOpt

Smart Context Optimization for AI Bug Localization on SWE-bench Lite.

Fault localization methods: **FlexFL** (two-stage ReAct, coarse-then-fine)
and **GraphLocator** (symptom vertex + neighbor expansion) — see
`docs/` for the source papers.

## Revised pipeline (current)

```
Graphify (structure, once per repo, fully local/offline)
    -> compressor: raw (control) | rtk (naive baseline) | lean-ctx (smart)
    -> feedback loop (agent double-checks fidelity, <=2 rounds, compressed conditions only)
    -> evaluation framework (compression_tax_analyzer.py)
```

- **Graphify** ([Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify)):
  local tree-sitter AST parsing, no LLM call, no network — builds a queryable
  structure map (files, functions, communities) *before* any log gets
  compressed, so every condition below reasons over the same map.
- **rtk**: kept as the naive baseline. `rtk_compressor.py` is an unchanged
  Python re-implementation of rtk's documented heuristics — the source of
  the original Compression Tax evidence (0.917 -> 0.833 file-level success;
  `sympy__sympy-17630`, `sympy__sympy-16792` as confirmed instances).
- **lean-ctx** ([yvgude/lean-ctx](https://github.com/yvgude/lean-ctx)):
  the smarter replacement. See `docs/leanctx_reference_notes.md` for the
  real daemon-vs-reference-mode distinction — this matters for whether a
  given result is publishable.
- **Feedback loop** (`feedback_loop.py`): after compression, the
  localization agent can flag `MISSING: L<range> <reason>` and get that
  exact raw line range revealed — capped at 2 rounds so it can't become an
  uncompressed re-run in disguise.

## Layout

```
wp1/
  fetch_instances.py        # SWE-bench Lite instances + gold-patch ground truth
  docker_harness.py         # real pytest capture (Docker, or local-fallback for offline dev)
  graphify_structure.py     # repo structure map, step 0
  rtk_compressor.py         # naive baseline (unchanged from earlier WP1)
  leanctx_compressor.py     # smart compressor (daemon mode + reference fallback)
  feedback_loop.py          # <=2-round fidelity check
  agent_localizer.py        # FlexFL+GraphLocator hybrid, model-agnostic (Claude/GPT/DeepSeek/Qwen)
  run_wp1_benchmark.py      # orchestrator: raw vs rtk vs lean-ctx
  compression_tax_analyzer.py  # 3-way scoring + T1-T5 taxonomy tagging
docs/
  leanctx_reference_notes.md
  rtk_reference_notes.md
  error_taxonomy_report.md
```

## Status

- Graphify integration: verified against a real repo (psf/requests), fully
  working offline.
- lean-ctx daemon mode: SDK verified against source; the real binary itself
  isn't installable from this sandbox yet due to a GitHub API rate limit on
  its shared IP (see `docs/leanctx_reference_notes.md`) — runs in
  `reference` mode (documented-behavior re-implementation, tagged
  provisional) until that's resolved on a normal network.
- Full pipeline wiring (Graphify -> compress -> feedback loop -> scoring):
  integration-tested end to end with the heuristic backend.
- Blocked on: Docker (for the authoritative SWE-bench eval images) and an
  `ANTHROPIC_API_KEY`/other provider key (for the real LLM localization
  backend, vs. the current key-free heuristic backend used to validate
  plumbing).

## Running

```bash
pip install -r requirements.txt

python wp1/fetch_instances.py --n 15 --seed 42 --out data/instances.json

python wp1/run_wp1_benchmark.py \
    --instances data/instances.json \
    --local-fallback \
    --backend heuristic \
    --out results/wp1_results.json

python wp1/compression_tax_analyzer.py \
    --results results/wp1_results.json \
    --out results/compression_tax_report.json
```

Swap `--backend heuristic` for `--backend claude` (requires
`ANTHROPIC_API_KEY`), `gpt`, `deepseek`, or `qwen` once credentials are
available. Drop `--local-fallback` to use the real SWE-bench Docker images
once Docker is available.
