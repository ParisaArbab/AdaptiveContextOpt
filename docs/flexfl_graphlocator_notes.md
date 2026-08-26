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
