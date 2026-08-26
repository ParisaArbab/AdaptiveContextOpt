# AdaptiveContextOpt

Smart Context Optimization for AI Bug Localization on SWE-bench Lite.

Fault localization: a real two-stage **FlexFL** pipeline (Agent4SR space
reduction -> Agent4LR refinement) combined with **GraphLocator**-style
causal-graph expansion over Graphify's actual call-graph edges — see
`wp1/agent_localizer.py` and `docs/flexfl_graphlocator_notes.md` for exactly
where and how each paper's method is applied.

## Pipeline (rtk removed — lean-ctx is now the only compression condition)

```
Graphify (structure map + real call graph, once per repo, fully local/offline)
    -> compressor: raw (control) | lean-ctx (smart)
    -> feedback loop (agent double-checks fidelity, <=2 rounds, lean-ctx only)
    -> FlexFL Stage 1: Agent4SR space reduction -> candidate list
    -> FlexFL Stage 2: Agent4LR refinement (real code snippets, re-ranks)
    -> GraphLocator: symptom vertices -> real call-graph expansion
    -> evaluation framework (compression_tax_analyzer.py)
```

- **Graphify** ([Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify)):
  local tree-sitter AST parsing — structure map AND the real `calls` edge
  graph that GraphLocator's expansion step walks.
- **lean-ctx** ([yvgude/lean-ctx](https://github.com/yvgude/lean-ctx)): the
  sole compression condition now. See `docs/leanctx_reference_notes.md` for
  the daemon-vs-reference-mode distinction.
- **rtk**: removed entirely per the revised architecture — no longer part
  of the comparison.
- **FlexFL** (Xu et al.): implemented as two explicit stages in
  `agent_localizer.py` — `stage1_space_reduction` / `AGENT4SR_SYSTEM_PROMPT`
  (space reduction, using stack-trace evidence + lexical overlap as the
  non-LLM signal, or a real ReAct loop over `StructureQueryTools` for LLM
  backends) and `AGENT4LR_SYSTEM_PROMPT` (refinement, re-ranking candidates
  against their real retrieved code snippets).
- **GraphLocator** (Liu et al.): implemented as `graphlocator_expand`,
  walking Graphify's real `calls` edges outward from symptom vertices
  (`graphify_structure.build_call_graph`) — not a community-clustering
  proxy, which an earlier version of this file mistakenly used.
- **Feedback loop** (`feedback_loop.py`): after lean-ctx compression, the
  localization agent can flag `MISSING: L<range> <reason>` and get that
  exact raw line range revealed — capped at 2 rounds.

## Layout

```
wp1/
  fetch_instances.py        # SWE-bench Lite instances + gold-patch ground truth
  docker_harness.py         # real pytest capture (Docker, or local-fallback for offline dev)
  graphify_structure.py     # structure map + real call graph (GraphLocator's substrate)
  leanctx_compressor.py     # smart compressor (daemon mode + reference fallback)
  feedback_loop.py          # <=2-round fidelity check
  agent_localizer.py        # real FlexFL (Agent4SR + Agent4LR) + GraphLocator expansion
  run_wp1_benchmark.py      # orchestrator: raw vs lean-ctx
  compression_tax_analyzer.py  # 2-way scoring + T1-T5 taxonomy tagging
docs/
  leanctx_reference_notes.md
  flexfl_graphlocator_notes.md
  error_taxonomy_report.md
```

## Status

- Graphify integration: verified against a real repo (psf/requests) —
  structure map AND call graph, both confirmed against real `graph.json` output.
- FlexFL + GraphLocator: integration-tested end to end on psf/requests —
  Stage 1 correctly recovered the exact stack-trace-evidenced method,
  GraphLocator expansion pulled in a real caller/callee via actual `calls`
  edges (not community clustering).
- lean-ctx daemon mode: SDK verified against source; real binary blocked by
  a GitHub API rate limit on this sandbox's IP (see
  `docs/leanctx_reference_notes.md`) — runs in `reference` mode
  (documented-behavior re-implementation, tagged provisional) until resolved.
- Blocked on: Docker (for the authoritative SWE-bench eval images) and an
  `ANTHROPIC_API_KEY`/other provider key (for the real LLM-driven FlexFL
  agents, vs. the current key-free heuristic backend used to validate
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
`ANTHROPIC_API_KEY`), `gpt`, `deepseek`, `qwen`, or `local` (a locally-served
open-source model on your own GPU — see `docs/gpu_setup.md`) once
credentials/infrastructure are available — this switches FlexFL's
Agent4SR/Agent4LR from the heuristic stand-ins to real LLM-driven ReAct
loops. Drop `--local-fallback` to use the real SWE-bench Docker images once
Docker is available.
