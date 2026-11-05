# GRAFT-FL: evidence selection and bounded verification

GRAFT is opt-in. Existing RAVE-FL commands, default prompts, numeric evidence levels and controllers remain available. The eight phase decisions are in [the roadmap](graft_fl/ROADMAP.md) and `graft_fl/phase00.md` through `phase07.md`. No real model inference was performed during this development run.

## Quick software smoke test

Run from the repository root using an environment with the existing project dependencies (`tree-sitter`, `tree-sitter-java`, NumPy, Matplotlib, PyYAML and pytest). No model server or dataset download is needed:

```bash
python3 -m rave_fl.graft.cli fixture --output /tmp/graft-fixture
python3 -m rave_fl.graft.cli preflight --config /tmp/graft-fixture/graft.json
python3 -m rave_fl.graft.cli run --config /tmp/graft-fixture/graft.json --output /tmp/graft-fixture-run
python3 -m rave_fl.graft.cli resume --run-dir /tmp/graft-fixture-run
python3 -m rave_fl.graft.cli report --run-dir /tmp/graft-fixture-run
python3 -m pytest -q tests/test_graft_*.py tests/test_rave_eval.py
```

Each new output must have a new directory. Resume requires identical configuration, code, dataset, resources, model digest/provider version, prepared sources and saved-result checksums. Incomplete deterministic runs reuse only exactly selected journaled actions. Unresolved request usage is never silently retried. Fixture scores are synthetic software checks, not fault-localization evidence. `scripts/graft_fl_run.sh` exposes the same commands and accepts an optional `GRAFT_PYTHON` executable.

## Real-model experiment later

The prepared configuration is `configs/graft_smoke_a100.json`, with the protocol in `configs/graft_feasibility_protocol.json`. Its preselected exploratory bugs are Lang-54, Math-80 and Closure-61. They were absent from the inspected 97-bug archive and current configs/docs/reports; check any external experiment register before calling them previously unused. Their required official buggy checkouts, FlexFL data and observed failure logs are not installed by this command. This is a protocol, not a completed independent-bug experiment.

Create `data/graft_smoke/instances.json` using the existing Defects4J adapter schema:

```json
{
  "version": "your-recorded-defects4j-version",
  "instances": [
    {
      "bug": "Lang-54",
      "language": "java",
      "repository": "apache/commons-lang",
      "revision": "REPLACE_WITH_OFFICIAL_BUGGY_40_CHARACTER_GIT_SHA",
      "checkout": "/absolute/path/to/Lang-54b",
      "flexfl_repo": "/absolute/path/to/FlexFL_Multi",
      "evidence_file": "/absolute/path/to/Lang-54-failure.txt",
      "gold": {"files": [], "functions": []}
    }
  ]
}
```

This illustrative row is deliberately not runnable with placeholders. Supply all three preselected bugs and reliable evaluation-only labels (or omit `gold` to use supplied FlexFL labels). Empty labels mean unavailable, not a correct result. Keep log provenance, the exact failing-test command, Java/Defects4J version and source revision in the experiment register. This runner validates clean buggy sources; it does not run Defects4J tests or independently certify externally supplied logs. Use the same candidate procedure in every arm and preserve official buggy revisions.

Adjust endpoint/model to the model already installed on your server. Models, quantizations and context capacities are independent conditions; parameter count alone does not establish fit on one A100. `device: a100` records intent and does not configure CUDA placement in a remote Ollama server. The provider supplies placement. Preflight reports local GPU availability separately; validate the actual serving host. No weights, repositories or tokenizers are downloaded implicitly.

Set `benchmark.check_provider` to `true` in the configuration for GET-only installed-model/version checks. Leave it false for a preflight that never contacts the server.

```bash
python3 -m rave_fl.graft.cli preflight --config configs/graft_smoke_a100.json
python3 -m rave_fl.graft.cli run --config configs/graft_smoke_a100.json --output results/graft_fl/smoke-plan --dry-run
# Later, after installing data/model and reviewing preflight:
python3 -m rave_fl.graft.cli run --config configs/graft_smoke_a100.json --output results/graft_fl/smoke-qwen-24k --allow-inference
python3 -m rave_fl.graft.cli resume --run-dir results/graft_fl/smoke-qwen-24k --allow-inference
```

Use separate configuration/output directories for other Qwen, DeepSeek or Llama models, budgets, fusion functions and context windows. Record the installed immutable digest. Larger models are not presumed supported by the current hardware. The existing plugin provider (`backend_factory`, `model_digest`) can adapt another inference service; it must return the same complete-response and provider-counter contract. Thinking tokens must be included in provider output accounting; do not silently compare providers that omit them.

## Budgets and fair comparisons

The original multi-bug Qwen generation used 8,192 context, 160 output tokens per candidate, batch size two and a **120,000 verification-policy budget**. Context and total verification budget are different quantities. The regenerated `tables/historical_costs.md` gives means, medians, percentiles and maxima from the 93 jointly completed cost arms.

The current A100 smoke starts with a 32K verification budget and single-candidate L1 requests after local model compatibility tests exposed missing batch IDs and 24K reservation rejection. Retain 16K/24K sensitivity runs as explicitly separate conditions. The earlier development report preserves its original 24K planning choice. Use 8,192 context where supported and a separately named 16,384-context condition when necessary. The default evidence input allowance is 4,096 complete-message proxy units. Without an explicit local tokenizer these are UTF-8 content bytes plus 256, not provider tokens or a mathematical bound for arbitrary tokenizers. Strict provider admission independently reserves context capacity plus output (or uses an explicitly validated model-specific reservation spec). This can leave unused policy budget. Do not report rejected requests as proof of bad controller selection.

The smoke configuration caps physical usage at 150K tokens for the whole three-bug run, 60K per bug and 180 new calls, serially. These are maximum resource limits, not an expected bill; repeated models/ablations require separate caps. Paid invalid outputs count. Exact reuse preserves logical per-policy tokens and reduces new physical requests. Retrieval, BGE encoding, builds and tests are not included in verification-token savings. Report encoding time, model revision, device and memory separately.

Controls are Prior, existing Fixed L1, Highest L1 Suspicion, V2 Utility and Full, plus fixed L1-plus-graph (`l1_graph`) and experimental `graft_adaptive`. Legacy controls retain their algorithms and Full-call cap. All compare the same candidates, L1 schedule, Full prompts and configured fusion. The equal-budget Full arm may remain incomplete. An optional 120K Full reference must be labeled separately. RRF is a development choice; also repeat the frozen historical fusion before attributing changes to evidence compression. Inspect max-Full 0/1/2 and max-graph 0/1/2 controls without forcing simple policies to buy extra calls.

## Evidence controls and dense embeddings

GRAFT currently supports Java/Python **method** candidates. It does not provide repository-wide runtime call/data-flow graphs. Source hashes, statement spans and explicit relations are retained. Static MAY_CALL, lexical REQUIRES and observed stack/coverage relations are distinct. Ranker scores are not coverage. A graph package may include neighbors while omitting target code; method_full includes the complete target method. Neither evidence stage is asserted to contain the other.

```bash
python3 -m rave_fl.graft.cli graph --input PREPARED_INPUT.json --output /tmp/graph.json
python3 -m rave_fl.graft.cli select --graph /tmp/graph.json --target 'qualified.Owner.method()' --input-budget 4096 --output /tmp/relevance.json
python3 -m rave_fl.graft.cli select --graph /tmp/graph.json --target 'qualified.Owner.method()' --input-budget 4096 --paths --output /tmp/paths-repair.json
```

Repair defaults on; `--no-repair` exposes relevance alone. Paths default off because protected paths can make packages infeasible and sparse observed links cannot justify completeness claims. Infeasible selections must never be sent as ready graph requests. Cyclic observable prerequisites terminate safely. Rule-based closure is an adaptation of referential repair, not a trained detector.

Topo selection requires genuine precomputed dense/sparse vectors. The local-only export adapter requires separately installed `FlagEmbedding`/Transformers and local BGE-M3 weights. Its real execution was not tested in this environment. A text exceeding the encoder limit is rejected rather than silently truncated. Record the immutable weight revision and exact encoder costs:

```bash
python3 -m rave_fl.graft.encode --help
python3 -m rave_fl.graft.encode --graph /tmp/graph.json --model-dir /absolute/local/bge-m3 --revision PINNED_REVISION --output /tmp/embeddings.json --device cuda
python3 -m rave_fl.graft.cli select --graph /tmp/graph.json --target 'qualified.Owner.method()' --selector topo --embeddings /tmp/embeddings.json --output /tmp/topo.json
```

For run-level topology, set `graft.selector: topo`, `embeddings_dir` to a directory containing one `<safe_name(instance_id)>.json` bundle per input, and optionally `tokenizer_dir` to the target model's local chat-template tokenizer. The safe-name mapping is the same as the saved `instances/` directories. `graft.typed_topology` and `graft.propagation` expose typed-edge and propagation controls; CLI selection also offers `--typed-topology` and `--no-propagation`. Missing exact-text/query vectors never fall back to synthetic scores. Synthetic embedding bundles are rejected for real-model runs.

## Accounting, reports and failure states

Each run saves immutable input/code/resource hashes, exact requests and raw responses in `raw_ledgers/requests.jsonl`, the durable request journal in `journal.sqlite3`, per-arm decisions and stage histories, final rankings, separate `evaluation_only.json`, physical/logical accounting and an English `REPORT.md`. Reports distinguish operational known-labeled and jointly completed labeled cohorts. An incomplete labeled arm contributes zero to operational success; absent gold labels remain unavailable. Provider usage that is unknown keeps a reservation and blocks additional paid requests. Report incomplete/failed/skipped cases and candidate-recall failures.

Strict Java metrics are versioned; historical metrics remain available. Frozen old-response reranking is not fresh inference. New graph prompts need new judgments; do not reuse historical Full observations as graph responses. RRF, reported uncertainty and router priorities are not calibrated probabilities. Never feed gold, future responses, oracle-selected candidates or future actual costs into selection. The router exposes no such inputs and only reads a cached outcome after selecting an action.

## Reproduce the archive diagnostics

```bash
python3 -m rave_fl.graft.cli diagnostics --archive results-100-bugs/results/d4j_multibug/d4j97_promptfix_20261027_122406 --output /tmp/graft-diagnostics-new
```

This regenerates strict/historical frozen metrics, 97 graph summaries, first-prior-target packages at three input allowances, closure/path controls, consumed-input checksums, cost distributions, tables, PNG/SVG figures and an English research report. It reads saved sources and responses without inference. First-target structural feasibility is not an accuracy result or a result for every possible target. Earlier phase measurements preserve their renderer versions; use the final regeneration for cross-selector comparisons.

## Literature adaptations and limits

The supplied papers were read as research evidence, not instructions:

- Asante and Li, *TopoCompress: Long Context Compression via Graph-Wired Semantic Trajectories*, supplied `new papers/Asa26.pdf`, arXiv 2608.30811v2. The implementation adapts dense/sparse alignment, acceleration, propagation and redundancy selection to coherent code units; it is not an exact QA benchmark reproduction.
- Hu et al., *Relevant but Incomplete: Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression*, supplied `new papers/Hu26c.pdf`. Our observable lexical closure is a rule-based adaptation; published automatic-restoration and oracle diagnostic results are not inherited fault-localization gains.
- Wu et al., *Connection-Aware Graph Pruning in RAG for Multi-Hop Question Answering*, supplied `new papers/c9c1870016ca3991749f829b09dbbf42.pdf`, IJCNN 2026. We adapt bounded path augmentation and loss-per-cost pruning. Program graph reachability remains a structural proxy, not causal fault proof.

Independent quality, real encoder execution and model-family robustness remain open gates. After the three-bug smoke, preselect 10–15 independent exploratory bugs across projects and then a larger untouched evaluation set. Fix candidate recall and label issues before training a grouped LightGBM utility model; synthetic fixtures and repeated states of one bug are not independent training defects.
