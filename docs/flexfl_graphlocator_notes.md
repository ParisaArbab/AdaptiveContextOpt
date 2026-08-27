# Where FlexFL and GraphLocator are actually applied

An earlier version of `agent_localizer.py` claimed to implement both papers
but only did a single regex pass over stack frames plus a same-community
lookup. That wasn't FlexFL (no two stages, no candidate list, no
refinement), and the "GraphLocator expansion" walked Graphify's `community`
clustering field, not a real graph. This note documents the actual
implementation after reading both papers directly.

## FlexFL (Xu et al., "FlexFL: Flexible and Effective Fault Localization
with Open-Source Large Language Models")

Real mechanism (from the paper, Section 3): two stages, each its own agent,
both following a three-step pipeline (task assignment -> interaction with
function calls -> summarization), bounded by a `MAX` iteration loop.

- **Stage 1 — Space Reduction (Agent4SR)**: combines non-LLM FL techniques
  (SBFL: dynamic coverage / test-spectrum signal; IRFL: lexical/bug-report
  matching) with an LLM agent that calls structure-query functions
  (`get_paths`, `get_classes_of_path`, `get_methods_of_class`, `find_class`,
  `find_method`, `get_code_snippet_of_method`, `exit` — Table 2 of the
  paper) to produce a ranked candidate list of suspicious methods.
- **Stage 2 — Localization Refinement (Agent4LR)**: takes that candidate
  list, retrieves each candidate's actual code, and re-reasons over the bug
  report + trigger test + code to produce a refined, re-ranked Top-k.

**Where this lives in the code:**
- `wp1/agent_localizer.py::StructureQueryTools` — the exact function-call
  set from Table 2, implemented against Graphify's structure map instead of
  a live code-execution sandbox (Graphify already indexed paths, classes,
  and methods offline, so these calls are direct lookups, not tool
  invocations that need a running interpreter).
- `wp1/agent_localizer.py::stage1_space_reduction` /
  `AGENT4SR_SYSTEM_PROMPT` — Stage 1. The heuristic backend's non-LLM
  signal (stack-trace evidence + lexical overlap) stands in for SBFL/IRFL;
  the LLM backend (`localize_with_llm`) runs an actual Agent4SR-style call
  using this same tool set.
- `wp1/agent_localizer.py::AGENT4LR_SYSTEM_PROMPT` and the Stage-2 block
  inside `localize_with_llm` — Agent4LR, given real code snippets pulled
  via `StructureQueryTools.get_code_snippet_of_method` (reads the actual
  checked-out repo file at the line Graphify recorded).
- `MAX_FLEXFL_ITERS` — the paper's bounded "Loop for MAX times" (Fig. 2).

**What's still a stand-in**: the paper's own SBFL/IRFL baselines (BoostNSift,
Ochiai, Dstar, etc.) are not reimplemented — Stage 1's non-LLM signal is a
simpler stack-trace + lexical-overlap heuristic used only to validate the
pipeline offline. A defensible final result should either integrate a real
SBFL tool or explicitly scope this as a simplification.

## GraphLocator (Liu et al., "GraphLocator: Graph-Guided Causal Reasoning
for Issue Localization")

Real mechanism (from the paper): a **causal issue graph (CIG)** — vertices
are code entities (plus discovered sub-issues), edges are causal
dependencies. Two-phase workflow: (1) **symptom vertices locating** — find
the entities directly implicated by the issue/failure, on the repository
graph; (2) **dynamic CIG discovering** — iteratively expand by reasoning
over *neighboring vertices on the repository graph*, discovering new
sub-issues and updating causal edges.

**Where this lives in the code:**
- `wp1/graphify_structure.py::build_call_graph` — the real "repository
  graph" GraphLocator expands over. Built from Graphify's actual `calls`
  edges (`graph.json`'s `links`, `relation == "calls"`), giving each node
  its real callers and callees — not the `community` clustering field an
  earlier version used, which groups by structural proximity, not causal
  (call) dependency.
- `wp1/agent_localizer.py::graphlocator_expand` — the expansion itself.
  Symptom vertices come from Stage 2's confirmed FlexFL output (or the
  trace-evidenced subset, for the heuristic backend); expansion walks real
  edges outward hop by hop (`MAX_GRAPH_HOPS`), and each candidate neighbor
  is passed through a confirmation step (`confirm_fn`) before being added
  — the paper's "iteratively reasoning over neighboring vertices," not an
  unconditional graph-neighbor grab.
- `_heuristic_confirm_fn` / `llm_confirm_fn` — the confirmation judgment:
  lexical-overlap heuristic for the key-free backend, an actual LLM causal
  plausibility call (`GRAPHLOCATOR_EXPAND_PROMPT`) for the real backends.

**What's still a stand-in**: the paper's CIG is richer than a simple
call-graph BFS — it tracks *sub-issues* as first-class vertices and
disentangles one-to-many mismatches (one issue -> multiple interdependent
entities) via dynamic issue disentangling, which isn't implemented here.
What's implemented is the graph-substrate and the confirmed-expansion loop;
issue disentangling would be a reasonable Stage 2 extension if `leanctx`
results show one-to-many mismatch cases in the taxonomy.

## Verification

Tested end to end against a real clone of `psf/requests` (not a synthetic
fixture): given a stack trace naming
`HTTPAdapter.build_connection_pool_key_attributes`, Stage 1 correctly
surfaced that exact method from real trace-frame parsing, and GraphLocator
expansion pulled in `get_connection_with_tls_context` — a real caller in
`adapters.py` connected via an actual `calls` edge in Graphify's graph, not
a community-membership coincidence.

## Second pass — checked against the real FlexFL replication package

The author shared the actual FlexFL authors' replication package
([ParisaArbab/FlexFL_OriginalReplication](https://github.com/ParisaArbab/FlexFL_OriginalReplication)),
which includes the real `pipeline.py` and `function_call.py`. Reading them
surfaced three real gaps in the first pass above, now fixed:

1. **Stage 1 and Stage 2 were single-shot, not the real multi-turn ReAct
   loop.** The real `pipeline.py` calls the model once per turn with a
   growing transcript, parses one `FunctionName(Argument)` per turn,
   dispatches it, appends the result, and repeats until `exit()` or
   `max_try` turns — only then asks for the final formatted answer. The
   first pass instead made one call and parsed candidates directly out of
   it, skipping the actual tool-use loop entirely. Fixed: `run_react_loop()`
   now mirrors this exactly, and `localize_with_llm` runs it once for Stage
   1 and once for Stage 2.
2. **Wrong constants.** Real `pipeline.py` uses `max_try = 10`, not 5. Real
   `rank.py` truncates the Stage 1 -> Stage 2 candidate handoff to
   `[:20]`, not 10. The final answer format is explicitly "top-5" in the
   real system prompt. `MAX_FLEXFL_ITERS`, `CANDIDATE_LIST_SIZE`, and the
   new `FINAL_TOPK` constant now match.
3. **No fuzzy-search fallback.** The real `function_call.py` falls back to
   Levenshtein-distance matching (distance <= 5, else the 5 closest) when
   an exact/token lookup misses — this is what lets the agent recover from
   a slightly-wrong name instead of just failing. `StructureQueryTools` now
   has the same fallback via `fuzzy_search()`/`_levenshtein()`.

Also confirmed from the real package but not yet adopted here:
`eval_FL.py` scores with Top-1/Top-3/Top-5 hit rates plus MAP/MRR, not the
binary file-level correctness `compression_tax_analyzer.py` currently uses.
Worth adopting once there's a real result set to score — binary
correctness is a coarser signal than the paper's own metric.

## Third pass — checked against the full paper text

The full FlexFL paper (readable this time, not the corrupted PDF extract
from the first pass) filled in details the replication package's code
didn't make obvious on its own. Two real gaps found and fixed:

1. **The final Top-k output was never postprocessed.** Section 3.2.1 Step 3
   is explicit: "the structured output of LLMs will be further refined
   using our postprocessing process, which matches the method names
   provided by LLMs to actual methods in the buggy program." The paper's
   own Time-25 case study shows this mattering — Agent4SR's raw 3rd-place
   guess was a slightly-wrong name, corrected by edit-distance matching to
   the real buggy method before Agent4LR ever saw it. The first pass here
   parsed `Top_i : ...` entries and used them as-is, including anything
   hallucinated. Fixed: `postprocess_topk()` runs every parsed entry through
   the same `fuzzy_search()` used for function-call arguments (Algorithm 1
   in the paper) before accepting it, for both Stage 1 and Stage 2 output.
2. **No adaptive MAX on context overflow.** Section 3.2.1: "If the whole
   conversation exceeds the maximum context length of the used LLM, we
   decrease the value of MAX by 1 and rerun this pipeline." Not implemented
   before. Fixed: `run_react_loop_with_adaptive_max()` catches
   context/token-limit errors from `chat_fn`, decrements `MAX_FLEXFL_ITERS`,
   and retries, down to a floor of 2 turns before giving up loudly.

Also fixed `fuzzy_search`'s tokenizer to split on `./():` (matching the
real `split4search`'s handling of `.` for names and `(` for signatures)
instead of just `.` and `:`.

**What's still the biggest documented gap, now precisely specified:**
Section 4.5's exact candidate-list interleaving formula for Stage 1's
non-LLM half:
- Bug report + trigger test both available: top-5 each of SBIR, Ochiai,
  BoostN, and Agent4SR, concatenated in that order (Agent4SR's results
  placed *last*, deliberately, since "methods localized by Agent4SR are
  more likely to be localized by Agent4LR... so we do not need to
  emphasize them via high ranking").
- Trigger test only: top-15 Ochiai + top-5 Agent4SR.
- Bug report only: top-15 BoostN + top-5 Agent4SR.

We don't have Python/SWE-bench equivalents of SBIR, Ochiai, or BoostN
wired in — `stage1_space_reduction`'s stack-trace + lexical-overlap
heuristic is a single stand-in for that whole ensemble, not a faithful
reproduction of the interleaving. A real SBFL pass (e.g., an Ochiai-style
suspiciousness score from pytest coverage) would be the natural first
addition if this needs to be more defensible than "pipeline validation."

## Fourth pass — graph understanding moved before FlexFL, not just after it

Per the author's direction: use the graph more for bug localization, so the
model understands the structure before FlexFL starts searching, rather than
only consulting the graph as a post-hoc refinement.

**Before this change:** `graphlocator_expand()` only ran after Agent4LR's
Stage 2 finished — the call graph was pure cleanup, never part of what
FlexFL actually reasoned over while searching.

**After this change:** `graph_structural_briefing()` runs first, using the
same real call-graph substrate (`graphify_structure.build_call_graph`):
finds symptom vertices from stack-trace evidence, walks their real
callers/callees, and produces a compact structural summary. That summary
now:
- gets prepended to FlexFL Stage 1's prompt (`localize_with_llm`), so
  Agent4SR's ReAct loop starts already knowing what's structurally
  connected to the failure, instead of discovering it cold through
  `find_class`/`find_method` calls alone;
- gets folded into the heuristic backend's Stage 1 candidate list directly
  (`HeuristicBackend.localize`), not just added afterward.

The post-Stage-2 `graphlocator_expand()` call still runs too — it's now a
second, confirmatory pass on top of an already graph-informed search,
rather than the only place the graph gets used. Both passes share the same
symptom-vertex detection (`_symptom_vertices_from_trace`).

Verified against `psf/requests`: given the same `build_connection_pool_key_attributes`
stack trace, `get_connection_with_tls_context` (a real 1-hop caller) now
appears in `stage1_candidates` itself — before the final expansion step
even runs — confirming the graph is informing the search, not just cleaning
up after it.


## Fifth pass — experiment harness rebuilt around the four gaps found on review

The FlexFL/GraphLocator implementations above were not the problem; the
harness around them was. Four fixes, in dependency order:

**1. The trigger test was never running.** `run_wp1_benchmark.py` called the
harness with `test_patch=""`, and the harness then ran a bare `pytest -v`
over the whole repo at `base_commit`. The FAIL_TO_PASS test ships *in* the
test patch, so it wasn't in the tree; the captured text was a whole-suite
run, not the trigger test's failure. Since Stage 1's non-LLM signal is
stack-trace evidence from exactly that failure, the input the whole
compression experiment operated on was largely missing the evidence it was
meant to compress. `docker_harness.py` now applies the gold test patch (three
fallback strategies) and runs only the FAIL_TO_PASS ids, in both modes;
`run_in_docker` executes an actual command inside the image instead of
`docker run --rm image` with no command, which ran no tests at all. A capture
with no failure evidence is skipped and recorded, not localized from.

**2. Tokens were never measured.** `CompressionResult` computed a chars/4
estimate that `run_one_condition` discarded, and `InstanceOutcome` had no
token fields at all — the project's headline claim was unrecorded.
`token_meter.py` adds a real tokenizer (tiktoken, or the served model's own
tokenizer for `--backend local`) and instruments `chat_fn` so every
prompt/completion is attributed to a pipeline stage. `metrics.py` replaces
the binary "any predicted file is in ground truth" with FlexFL's own
`eval_FL.py` metric set (Top-1/3/5, MAP, MRR) plus precision, at method and
file level. That last part mattered more than it looks: the old metric
ignored rank *and* rewarded breadth, so GraphLocator expansion — which adds
entities to the prediction set — could only ever improve the score. Rank
order is now load-bearing, so `agent_localizer` no longer sorts or set-ifies
its output anywhere; Agent4LR's refined ranking comes first, causal
expansions are appended behind it.

**3. There was no ablation.** Two conditions existed (`raw`, `leanctx`).
`ablation.py` defines the 2³ factorial over graphify / lean-ctx / feedback,
with `pure_flexfl` as the control and `element_contributions` in the analyzer
differencing each element's arm pairs. Every comparable pair is reported
rather than one summary number, because the marginal effect of lean-ctx is
not guaranteed to be the same with and without graphify — if the pairs
disagree, the elements interact, and a single number would be hiding that.

**4. The feedback loop only went one way.** It could restore pruned lines but
never remove useless ones, so it could only ever *add* tokens — a pure cost
against the project's own metric. It is bidirectional now (`MISSING:` in raw
`L` coordinates, `USELESS:` in current-text `C` coordinates), with four
independent termination guards: the round cap, a no-progress stop, an
idempotence ledger that blocks restore/prune oscillation on the same region,
and a per-round prune budget. Restores are spliced back at their original
position rather than appended, because appending would break the stack-frame
ordering Stage 1 reads causally.

Also fixed while in here: `compression_tax_analyzer.analyze` had its tax-case
loop indented outside the per-instance loop, so `conds` leaked from the last
iteration and the entire compression-tax output described one instance; and
`graph_structural_briefing` had three orphan lines of a clobbered
`_lexical_overlap_score` body sitting unreachable after its `return`.

**Benchmark generality** (`benchmarks.py`): dataset schema and language
behaviour are now data, not code. `DatasetSpec.field_map` remaps columns for
a fork; `LanguageAdapter` owns gold-patch symbol extraction, source suffixes,
the tree-sitter grammar name, and trigger-test invocation. Python runs pytest
node ids; Java detects Maven vs Gradle from the checkout and builds the right
`-Dtest=Class#method` or `--tests Class.method` selector.

## Sixth pass — provider layer, one-command runner, figures

**Every LLM, one contract.** `llm_backends.py` replaces the four hardcoded
`make_*_chat_fn` constructors with a registry: three provider *kinds*
(`openai_compatible`, `anthropic`, `heuristic`) covering local servers
(Ollama, vLLM, LM Studio, TGI, llama.cpp), hosted open-weight endpoints
(DeepSeek, Qwen/DashScope, OpenRouter, Together, Groq, Mistral, Ollama
Cloud) and proprietary APIs (OpenAI, Anthropic, Gemini), plus a `custom` row
for anything OpenAI-shaped. Providers differ only in base URL, key env var
and default model, so they are rows rather than code, and
`--llm custom --base-url ... --model ...` covers whatever isn't registered.

Three behaviours matter specifically for the open-source models FlexFL is
designed around, and they live in the provider layer so no parser downstream
has to know about them:

- **Reasoning tags.** R1/QwQ/Qwen3-thinking style models emit
  `<think>…</think>` (and DeepSeek's API returns `reasoning_content`
  out-of-band). FlexFL parses exactly one `FunctionName(Argument)` per turn
  and `Top_k : …` lines for the final answer; a reasoning block containing
  either string breaks both silently — the loop would dispatch a function the
  model was only *considering*. Stripped from every response, including the
  unterminated-opener case that happens when a model hits `max_tokens`
  mid-thought.
- **Retries.** A local server 503s while a 32B model loads; a hosted one
  rate-limits. Without retries a single dropped call aborts a ReAct loop and
  drops that instance from every arm, which biases the comparison toward
  whichever arm happened not to hit the blip.
- **Context errors are explicitly NOT retried.** `is_retryable` returns False
  for them so the exception propagates to
  `run_react_loop_with_adaptive_max`, which is the paper's §3.2.1 behaviour
  (decrement MAX, rerun). Retrying the identical oversized prompt would burn
  the budget and then fail anyway. The marker list is shared between the two
  modules so they cannot drift apart.

Temperature defaults to 0.0: arms are compared against each other, and
sampling noise between arms would be indistinguishable from an element's
effect.

**One command.** `scripts/run_pipeline.sh` runs fetch → every ablation arm →
evaluation → figures, with `--llm`/`--model` and `--dataset`/`--language` as
flags. Each invocation writes to `results/<llm>_<model>_<dataset>_<stamp>/`
so runs never clobber each other and two models remain comparable from the
artifacts alone. It fails fast on a run that produced zero outcomes, printing
the deduplicated skip reasons, rather than failing deep inside the plotting
code where the cause is no longer visible.

**Figures.** `plot_results.py` writes eight, but the load-bearing one is
`tradeoff_*.png`: tokens on x, accuracy on y, Pareto frontier drawn. The
whole project is a trade-off claim, and a frontier is the only presentation
where "saved tokens but lost accuracy" and "strictly better" are visually
distinguishable. `per_instance_heatmap.png` exists to answer the follow-up
question a reviewer will ask about any aggregate difference — whether arms
fail on the *same* instances (hard instances) or different ones (a real
compression effect). Arm colours are stable across figures and the control
arm is always neutral grey.
