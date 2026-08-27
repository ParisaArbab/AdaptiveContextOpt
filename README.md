# AdaptiveContextOpt

Token optimization for LLM bug localization, evaluated on **FlexFL** across a
full ablation matrix and any SWE-bench-shaped benchmark.

Fault localization is a real two-stage **FlexFL** pipeline (Agent4SR space
reduction → Agent4LR refinement) combined with **GraphLocator**-style causal
graph reasoning over Graphify's real call-graph edges. See
`docs/flexfl_graphlocator_notes.md` for exactly where and how each paper's
method is applied.

## Pipeline

```
capture the REAL trigger test failing (FAIL_TO_PASS, test patch applied)
    -> graphify: structure map + real call graph          [cached per repo]
    -> lean-ctx: compress the captured output                  [ablatable]
    -> feedback loop: bidirectional restore/prune, <=2 rounds  [ablatable]
    -> graph structural briefing primes FlexFL Stage 1         [ablatable]
    -> FlexFL Stage 1 (Agent4SR)  -> Stage 2 (Agent4LR)
    -> GraphLocator causal expansion over real `calls` edges   [ablatable]
    -> evaluation: tokens (per stage) x accuracy (Top-k/MAP/MRR)
```

## Ablation matrix

Three independently-switchable elements → the full 2³ factorial, minus the
one combination that isn't meaningful (feedback with no compressor to give
feedback on). `python wp1/ablation.py` prints the table.

| arm | graphify | lean-ctx | feedback |
|---|---|---|---|
| `full` | on | on | on |
| `no_graphify` | off | on | on |
| `no_leanctx` | on | off | off¹ |
| `no_feedback` | on | on | off |
| `no_graphify_feedback` | off | on | off |
| `pure_flexfl` (control) | off | off | off |

¹ turning lean-ctx off forces feedback off — there is no compression to
audit.

`--arms default` runs the five named in the brief; `--arms all` runs
everything, which is what identifies element *interactions* rather than just
main effects.

The graphify ablation removes the **graph** (the pre-search structural
briefing and the GraphLocator expansion), not the structural index — FlexFL's
own `find_method` / `get_code_snippet_of_method` calls need an index in every
arm, and the real FlexFL builds one itself. Ablating that too would ablate
FlexFL rather than this project's contribution.

## Evaluation

Two axes, per arm, in one report (`wp1/compression_tax_analyzer.py`):

- **Tokens** — real tokenizer (tiktoken, or the served model's own tokenizer
  for `--backend local`), broken out per pipeline stage (feedback /
  Agent4SR / Agent4LR / GraphLocator) and per context bucket (raw capture,
  post-compression, agent input, structural briefing).
- **Accuracy** — FlexFL's own metric set: Top-1/3/5 hit rate, MAP, MRR, plus
  precision and recall, at **method** and **file** granularity.
- **Trade-off** — `tokens_per_correct_top1`, `top1_per_1k_tokens`, each arm's
  delta vs the `pure_flexfl` control, and `element_contributions`: each
  element's marginal effect obtained by differencing the arms that differ
  only in that element.
- **Compression tax cases** — instances the control localized but an
  optimized arm missed, tagged T1–T5 with the reason.

Results from lean-ctx reference mode, `local_fallback` captures, or the
key-free `heuristic` backend are marked `provisional` and reported
separately from headline numbers.

## Benchmarks

`--dataset` takes a registry alias or a raw HuggingFace id:
`swe-bench-lite`, `swe-bench-verified`, `swe-bench`, `swe-bench-java`,
`swe-bench-multimodal`. A fork whose columns differ needs no code change —
`--field-map fail_to_pass=<column>` remaps the schema. Language behaviour
(gold-patch symbol extraction, source suffixes, the tree-sitter grammar, and
how trigger tests are invoked) lives in `wp1/benchmarks.py`: Python runs
pytest node ids, Java detects Maven vs Gradle and builds the right selector.

## Layout

```
wp1/
  benchmarks.py             # dataset registry + per-language adapters
  fetch_instances.py        # instances + gold-patch ground truth (any dataset)
  docker_harness.py         # real trigger-test capture (Docker or local)
  graphify_structure.py     # structure map + real call graph, cached
  leanctx_compressor.py     # smart compressor (daemon mode + reference fallback)
  feedback_loop.py          # bidirectional restore/prune, 4 termination guards
  agent_localizer.py        # FlexFL (Agent4SR + Agent4LR) + GraphLocator
  token_meter.py            # real tokenizer + per-stage token accounting
  metrics.py                # Top-1/3/5, MAP, MRR, precision, recall
  ablation.py               # the arm matrix
  run_wp1_benchmark.py      # orchestrator
  compression_tax_analyzer.py  # evaluation framework + T1-T5 taxonomy
docs/
  leanctx_reference_notes.md
  flexfl_graphlocator_notes.md
  gpu_setup.md
```

## One command, whole study

```bash
./scripts/run_pipeline.sh --llm ollama --model qwen2.5-coder:32b
```

That fetches instances, runs **every** ablation arm (no optimization, full
pipeline, and each element removed), evaluates, and writes all figures.
Each run lands in its own `results/<llm>_<model>_<dataset>_<timestamp>/`
directory — instances, outcomes, report, plots, and a full log — so runs
never overwrite each other and two models stay comparable afterwards.

The model and the benchmark are both flags, so re-running the study
elsewhere is a flag change, not an edit:

```bash
./scripts/run_pipeline.sh --llm deepseek --dataset swe-bench-verified --n 30
./scripts/run_pipeline.sh --llm vllm --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --dataset swe-bench-java --language java --arms all
./scripts/run_pipeline.sh --llm openai --model gpt-4o --preflight
./scripts/run_pipeline.sh --dry-run --llm groq --model llama-3.3-70b-versatile
```

`--help` lists every flag; `--list-providers` prints the provider table;
`--preflight` makes one cheap call first so a long run doesn't die on call #1.

## Models

Any LLM, open-source or hosted, local or cloud — one registry in
`wp1/llm_backends.py`, no code change to add an endpoint.

| where | providers |
|---|---|
| local, your hardware | `ollama` `vllm` `lmstudio` `tgi` `llamacpp` |
| hosted, open weights | `deepseek` `qwen` `openrouter` `together` `groq` `mistral` `ollama-cloud` |
| hosted, proprietary | `openai` `anthropic` `gemini` |
| anything else | `custom --base-url <url> --model <id>` |
| no model | `heuristic` (key-free stand-in) |

Aliases (`claude`, `gpt`, `dashscope`, `local`, …) resolve to the right row.

Three things open-weight models need are handled in the provider layer rather
than scattered through the localizer:

- **Reasoning tags are stripped.** DeepSeek-R1, QwQ and Qwen3-thinking emit
  `<think>…</think>`; FlexFL parses one `FunctionName(Argument)` per turn and
  `Top_k : …` lines, both of which a reasoning block breaks. DeepSeek's
  out-of-band `reasoning_content` is handled too.
- **Retries with backoff**, because a local server 503s while a 32B model
  loads and a hosted one rate-limits — a single dropped call would otherwise
  abort a ReAct loop and lose that instance from every arm.
- **Context-limit errors are deliberately not retried**, so FlexFL's adaptive
  MAX (paper §3.2.1) still sees them and shrinks the loop instead of
  re-sending the same oversized prompt.

Token counting follows the model: an HF repo id (`Qwen/Qwen2.5-Coder-32B-Instruct`)
uses that model's own tokenizer when `transformers` is installed, otherwise
tiktoken. For an Ollama tag like `qwen2.5-coder:32b`, pass
`--tokenizer-model <hf-id>` for exact counts.

## Figures

`wp1/plot_results.py` runs at the end of every `run_pipeline.sh` invocation
(or standalone against any report):

| figure | question |
|---|---|
| `tradeoff_top1.png` | **the headline** — tokens vs accuracy, with the Pareto frontier drawn |
| `tokens_by_arm.png` | mean tokens per arm, labelled with % saved vs control |
| `accuracy_by_arm_{method,file}.png` | Top-1/3/5, MRR, MAP per arm |
| `tokens_by_stage.png` | where tokens actually go: feedback / Agent4SR / Agent4LR / GraphLocator |
| `element_contributions.png` | each element's token saving against its accuracy cost |
| `compression_funnel.png` | raw capture vs what the agent actually reads |
| `taxonomy.png` | T1–T5 tags and compression-tax cases per arm |
| `per_instance_heatmap.png` | which instances each arm got right — same instances, or different ones? |

Arms keep the same colour in every figure and the control arm is always
neutral grey, so figures read side by side. A provisional run also drops a
`PROVISIONAL.txt` next to the images naming exactly why.

## Running the steps individually

```bash
pip install -r requirements.txt

python wp1/fetch_instances.py --dataset swe-bench-lite --n 15 --seed 42 \
    --out data/instances.json

python wp1/run_wp1_benchmark.py \
    --instances data/instances.json \
    --arms default \
    --local-fallback \
    --llm ollama --model qwen2.5-coder:32b \
    --out results/wp1_results.json

python wp1/compression_tax_analyzer.py \
    --results results/wp1_results.json \
    --out results/compression_tax_report.json

python wp1/plot_results.py \
    --report results/compression_tax_report.json \
    --results results/wp1_results.json \
    --out-dir results/plots
```

`--llm heuristic` runs key-free for plumbing validation; any provider from
the table above switches FlexFL's Agent4SR/Agent4LR to real LLM-driven ReAct
loops and makes the token numbers measure actual model consumption rather
than just context size. Drop `--local-fallback` to run the trigger tests
inside the official SWE-bench Docker images. See `docs/gpu_setup.md` for
serving an open-weight model on your own GPU.

## Status and known stand-ins

- **Real trigger test capture**: the gold test patch is applied and only the
  FAIL_TO_PASS ids run, in both Docker and local mode. An instance whose
  capture contains no failure evidence is skipped and listed under
  `skipped`, rather than being localized from an empty capture.
- **Graphify**: verified against a real repo (psf/requests) — structure map
  and call graph both confirmed against real `graph.json` output.
- **FlexFL + GraphLocator**: integration-tested end to end on psf/requests.
- **lean-ctx daemon mode**: SDK verified against source; the real binary was
  blocked by a GitHub API rate limit on the original sandbox IP (see
  `docs/leanctx_reference_notes.md`). Until it's installed, the compressor
  runs in `reference` mode and every result is tagged `provisional`.
- **Stage 1's non-LLM half** is a stack-trace + lexical-overlap heuristic
  standing in for the paper's SBFL/IRFL ensemble (SBIR / Ochiai / BoostN)
  and its exact interleaving formula. This is the largest remaining fidelity
  gap and is documented in `docs/flexfl_graphlocator_notes.md`.
- **The `heuristic` backend** is a key-free stand-in for pipeline validation,
  not a result. Its feedback verifier is deterministic, so the feedback
  ablation is still a real variable without an API key — but the arm
  comparison only becomes meaningful on a real LLM backend.
