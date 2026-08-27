"""
agent_localizer.py — WP1

This is a rewrite. The previous version claimed to implement FlexFL +
GraphLocator but only did a single regex pass over stack frames plus a
same-community lookup — that is not FlexFL, and the "GraphLocator expansion"
wasn't walking a real graph. Grounded against both papers
(FlexFL.pdf, GraphLocatorGraphGuided_Causal_Reasoning.pdf) this version
implements the actual two mechanisms:

FlexFL (Xu et al.) — two explicit stages, each its own agent:
  Stage 1, Space Reduction (Agent4SR): combines non-LLM FL signal (here:
    stack-trace/trigger-test evidence, standing in for SBFL's dynamic
    coverage signal, plus lexical overlap with the problem statement,
    standing in for IRFL) with an LLM agent that can call structure-query
    functions to browse the codebase, producing a ranked CANDIDATE LIST of
    suspicious methods. The paper's real function-call set: get_paths,
    get_classes_of_path, get_methods_of_class, get_code_snippet_of_method,
    find_class, find_method, exit — implemented below against Graphify's
    structure map, which already indexes exactly that information locally.
  Stage 2, Localization Refinement (Agent4LR): takes the Stage-1 candidate
    list, retrieves each candidate's real code snippet, and re-reasons over
    bug report + trigger test + code to produce a re-ranked Top-k. This is
    a ReAct loop bounded by MAX_FLEXFL_ITERS, matching the paper's
    "Loop for MAX times" (Fig. 2).

GraphLocator (Liu et al.) — real mechanism is a causal issue graph (CIG):
  vertices = code entities, edges = causal/call dependencies. Workflow:
  (1) locate symptom vertices (the entities directly implicated by observed
  failure evidence), (2) dynamically expand the CIG by iteratively reasoning
  over NEIGHBORING VERTICES ON THE REPOSITORY GRAPH. We use Graphify's real
  'calls' edges (graphify_structure.build_call_graph) as that repository
  graph — not a community-clustering proxy, which was the earlier version's
  mistake. Expansion walks actual callers/callees outward from symptom
  vertices, bounded by MAX_GRAPH_HOPS, and each expansion round is confirmed
  (heuristically or by the LLM) before being added — mirroring the paper's
  "iteratively reasoning over neighboring vertices" rather than grabbing
  every neighbor unconditionally.

Verified against ParisaArbab/FlexFL_OriginalReplication (the actual FlexFL
authors' replication package — real function_call.py, pipeline.py). Fixes
made after reading the real source, vs. the paper-only version before this:
  - Stage 1/2 are now real multi-turn ReAct loops (chat_fn called once per
    turn with the growing transcript, function call parsed out, dispatched,
    result appended) — not a single chat_fn call with candidates parsed
    from one response. run_react_loop() below mirrors pipeline.py's actual
    loop structure exactly (append transcript, "Now call a function in
    this format...", parse `FunctionName(Argument)`, dispatch, repeat).
  - MAX_FLEXFL_ITERS = 10 (real pipeline.py: `max_try = 10`), not 5.
  - CANDIDATE_LIST_SIZE = 20 (real rank.py: `suspicious_methods[:20]`
    is what SR hands to LR), not 10.
  - FINAL_TOPK = 5 (real pipeline.py: "top-5 most likely culprit methods"),
    added as an explicit constant rather than left implicit.
  - StructureQueryTools now falls back to Levenshtein-distance fuzzy search
    when an exact/substring lookup misses, matching function_call.py's real
    `fuzzy_search` (exact-token match first, Levenshtein distance <= 5 as
    fallback, else the 5 closest) instead of substring-only matching.

Model-agnostic by design (Claude/GPT/DeepSeek/Qwen), per the project's
stated comparison goal.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import llm_backends
import metrics
import token_meter
from contextlib import nullcontext as _nullcontext
from feedback_loop import (
    FIDELITY_SYSTEM_PROMPT,
    heuristic_verify_fn,
    run_feedback_loop,
)

MAX_FLEXFL_ITERS = 10          # real FlexFL pipeline.py: max_try = 10
MAX_GRAPH_HOPS = 2              # GraphLocator: bounded CIG expansion, not unbounded traversal
CANDIDATE_LIST_SIZE = 20        # real FlexFL rank.py: suspicious_methods = suspicious_methods[:20]
FINAL_TOPK = 5                  # real FlexFL pipeline.py: "top-5 most likely culprit methods"

FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([A-Za-z_][A-Za-z0-9_]*)')


@dataclass
class LocalizationResult:
    """predicted_functions and predicted_files are RANKED, best first — the
    Top-k/MAP/MRR metrics in metrics.py read rank, so nothing in this module
    may sort or set-ify them on the way out. GraphLocator's expansions are
    appended after the refined FlexFL ranking rather than merged into it,
    since they are causally-related context, not higher-confidence guesses."""

    instance_id: str
    predicted_files: List[str]
    predicted_functions: List[str]
    backend: str
    feedback_rounds_used: int = 0
    stage1_candidates: List[str] = field(default_factory=list)   # FlexFL Agent4SR output
    graph_expanded: List[str] = field(default_factory=list)      # GraphLocator CIG additions
    used_graph: bool = True
    used_feedback: bool = False
    feedback_restores: int = 0
    feedback_prunes: int = 0
    feedback_stop_reason: str = ""
    token_report: dict = field(default_factory=dict)


def _ordered_dedupe(*sequences: List[str]) -> List[str]:
    """Concatenate ranked lists, first occurrence wins, order preserved."""
    seen = set()
    out: List[str] = []
    for seq in sequences:
        for item in seq:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


class LLMChatBackend(Protocol):
    def complete(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
# Structure-query functions — Graphify-backed equivalents of FlexFL's real
# function-call set (Table 2 of the paper): get_paths, get_classes_of_path,
# get_methods_of_class, get_code_snippet_of_method, find_class, find_method.
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Standard edit-distance DP — matches the real function_call.py's use
    of Levenshtein.distance() as the fuzzy-search fallback metric."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def fuzzy_search(query: str, choices: List[str], max_results: int = 5) -> List[str]:
    """Real fuzzy_search from function_call.py: exact-token substring match
    first (splitting on '.'/'/'/'(' the way the real split4search does), and
    only when that returns nothing does it fall back to Levenshtein
    distance <= 5, else the closest max_results choices. This is Algorithm 1
    in the paper — used both as a function-call-argument fallback AND, per
    Section 3.2.1's Step 3, to refine the FINAL Top-k output before it's
    accepted (see postprocess_topk below)."""
    query_tokens = re.split(r"[./:()]+", query.lower())
    exact = [c for c in choices if all(t in c.lower() for t in query_tokens if t)]
    if exact:
        return exact

    distances = sorted(((c, _levenshtein(query.lower(), c.lower())) for c in choices),
                        key=lambda t: t[1])
    close = [c for c, d in distances if d <= 5]
    return close if close else [c for c, _ in distances[:max_results]]


def postprocess_topk(entries: List[str], structure_map: Dict[str, dict]) -> List[str]:
    """Real Section 3.2.1 Step 3: 'the structured output of LLMs will be
    further refined using our postprocessing process, which matches the
    method names provided by LLMs to actual methods in the buggy program.'
    This was missing from the first pass — Top_k entries were accepted
    as-is, including hallucinated names that don't exist in structure_map.
    The paper's own case study (Time-25) shows this mattering: Agent4SR's
    raw 3rd-place guess was wrong, and postprocessing corrected it via edit
    distance to the real buggy method before Agent4LR ever saw it."""
    choices = list(structure_map.keys())
    resolved: List[str] = []
    for entry in entries:
        if entry in structure_map:
            resolved.append(entry)
            continue
        matches = fuzzy_search(entry, choices, max_results=1)
        if matches and matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


class StructureQueryTools:
    """Wraps a Graphify structure_map + repo checkout so an agent (LLM or
    heuristic) can call the same function set FlexFL's paper defines,
    without needing a live code-execution sandbox — Graphify already
    indexed this offline."""

    def __init__(self, structure_map: Dict[str, dict], repo_root: Optional[Path] = None):
        self.structure_map = structure_map
        self.repo_root = Path(repo_root) if repo_root else None

    def get_paths(self) -> List[str]:
        return sorted({meta["file"] for meta in self.structure_map.values()})

    def get_classes_of_path(self, path: str) -> List[str]:
        exact = sorted(
            key.split("::", 1)[1]
            for key, meta in self.structure_map.items()
            if meta["file"] == path
        )
        if exact:
            return exact
        # real get_classes(): falls back to fuzzy-searching the path itself
        return fuzzy_search(path, self.get_paths())

    def get_methods_of_class(self, class_key: str) -> List[str]:
        # Graphify flattens classes/functions into one namespace; a real
        # "methods of class" query would need class-membership edges, which
        # Graphify's 'calls' links don't encode directly. We approximate
        # with all entities in the same file as the class, which is the
        # information FlexFL's function is actually used for in practice
        # (narrowing from a class to its neighborhood). Falls back to fuzzy
        # search on the class name itself when the file has nothing,
        # matching the real get_methods()'s fuzzy fallback.
        file = class_key.split("::", 1)[0] if "::" in class_key else None
        exact = self.get_classes_of_path(file) if file else []
        if exact:
            return exact
        return fuzzy_search(class_key, list(self.structure_map.keys()))

    def find_class(self, name: str) -> List[str]:
        name_lower = name.lower()
        exact = [key for key in self.structure_map if name_lower in key.lower()]
        return exact if exact else fuzzy_search(name, list(self.structure_map.keys()))

    def find_method(self, name: str) -> List[str]:
        return self.find_class(name)  # same fuzzy-search mechanism, flat namespace

    def get_code_snippet_of_method(self, key: str, context_lines: int = 8) -> Optional[str]:
        meta = self.structure_map.get(key)
        if not meta:
            # real get_code_snippet(): fuzzy-search and offer a "did you
            # mean" style correction instead of failing silently
            matches = fuzzy_search(key, list(self.structure_map.keys()), max_results=1)
            if len(matches) == 1:
                meta = self.structure_map.get(matches[0])
                key = matches[0]
        if not meta or meta.get("line") is None or self.repo_root is None:
            return None
        file_path = self.repo_root / meta["file"]
        if not file_path.exists():
            return None
        lines = file_path.read_text(errors="replace").splitlines()
        start = max(0, meta["line"] - 1)
        end = min(len(lines), start + context_lines)
        return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Stage 1: Space Reduction (Agent4SR)
# ---------------------------------------------------------------------------

def _symptom_vertices_from_trace(tool_output: str, structure_map: Dict[str, dict]) -> List[str]:
    """Real GraphLocator phase 1 ('symptom vertices locating'): the entities
    directly implicated by observed failure evidence. Shared by the
    pre-search briefing below and the post-refinement expansion, so both
    stages start from the same grounding."""
    return [
        key for key in structure_map
        if any(m.group(3).lower() in key.lower() for m in FRAME_RE.finditer(tool_output))
    ]


def graph_structural_briefing(
    tool_output: str,
    structure_map: Dict[str, dict],
    call_graph: Dict[str, Dict[str, list]],
    id_to_key: Dict[str, str],
    max_hops: int = 1,
) -> tuple[str, List[str]]:
    """Runs GraphLocator's real graph-substrate analysis BEFORE FlexFL
    starts searching, instead of only after Agent4LR finishes — this is the
    'understand the structure first' step. Symptom vertices come straight
    from stack-trace evidence; their real callers/callees (via Graphify's
    actual 'calls' edges) become a structural neighborhood that primes
    Agent4SR's starting context, so the ReAct loop begins already knowing
    what's structurally connected to the failure instead of discovering it
    cold through function calls alone.

    Returns (briefing_text, neighbor_keys) — the text goes into Stage 1's
    prompt, the keys are reused as a scoring boost in the heuristic backend.
    """
    symptom_vertices = _symptom_vertices_from_trace(tool_output, structure_map)
    if not symptom_vertices:
        return "", []

    neighbor_keys: List[str] = []
    frontier = list(symptom_vertices)
    for _hop in range(max_hops):
        next_frontier = []
        for vertex_key in frontier:
            meta = structure_map.get(vertex_key)
            node_id = meta.get("id") if meta else None
            if not node_id or node_id not in call_graph:
                continue
            for nid in call_graph[node_id]["callers"] + call_graph[node_id]["callees"]:
                key = id_to_key.get(nid)
                if key and key not in symptom_vertices and key not in neighbor_keys:
                    neighbor_keys.append(key)
                    next_frontier.append(key)
        frontier = next_frontier
        if not frontier:
            break

    lines = ["Structural context (Graphify call graph + GraphLocator symptom analysis, "
             "gathered before search):"]
    for sv in symptom_vertices:
        lines.append(f"- symptom vertex: {sv}")
    if neighbor_keys:
        lines.append(f"- structurally connected ({max_hops}-hop callers/callees): "
                      + ", ".join(neighbor_keys[:15]))
    return "\n".join(lines), neighbor_keys


def _lexical_overlap_score(text: str, key: str) -> int:
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    key_tokens = set(re.findall(r"[a-z0-9]+", key.lower()))
    return len(text_tokens & key_tokens)


def stage1_space_reduction(
    tool_output: str,
    problem_statement: str,
    tools: StructureQueryTools,
    top_k: int = CANDIDATE_LIST_SIZE,
) -> List[str]:
    """Non-LLM signal (stack-trace evidence, standing in for SBFL's dynamic
    coverage; lexical overlap with the problem statement, standing in for
    IRFL) combined into a ranked candidate list — this is Agent4SR's job in
    the paper, minus the LLM reasoning loop for the heuristic backend. The
    LLM backend (localize_with_llm below) replaces this ranking with an
    actual Agent4SR ReAct loop over the same StructureQueryTools."""
    scored: Dict[str, float] = {}

    for m in FRAME_RE.finditer(tool_output):
        _file_path, _line, func_name = m.groups()
        for key in tools.find_method(func_name):
            scored[key] = scored.get(key, 0.0) + 5.0  # direct trace evidence weighted highest

    for key in tools.structure_map:
        overlap = _lexical_overlap_score(problem_statement, key)
        if overlap:
            scored[key] = scored.get(key, 0.0) + overlap

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [key for key, _ in ranked[:top_k]]


AGENT4SR_SYSTEM_PROMPT = """You are a debugging assistant of our Python software. \
You will be presented with a bug report and/or trigger test and tools (functions) \
to access the source code of the system under test. Your task is to locate the \
top-{top_k} most likely culprit methods based on the available information and \
the information you retrieve using given functions.
Function calls you can use are as follows.
* find_class(`class_name`) -> Find a class by fuzzy search.
* find_method(`method_name`) -> Find a method by fuzzy search.
* get_paths() -> Get the paths of the Python software system.
* get_classes_of_path(`path_name`) -> Get the classes/functions in a path.
* get_methods_of_class(`class_key`) -> Get the methods belonging to a class.
* get_code_snippet_of_method(`key`) -> Get the code snippet of a method.
* exit() -> Exit function calling to give your final answer when confident.
You have {max_iters} chances to call a function."""

AGENT4LR_SYSTEM_PROMPT = """You are a debugging assistant of our Python software. \
You are given a bug report and/or trigger test and a candidate list of suspicious \
methods suggested by a prior stage. Function calls you can use are as follows.
* get_code_snippet_of_method(`method_number`) -> Get the code snippet of the \
Python method by its number in the suggested list.
* exit() -> Exit function calling to give your final answer when confident.
You have {max_iters} chances to call a function."""

REACT_TURN_INSTRUCTION = (
    "Now call a function in this format `FunctionName(Argument)` in a single "
    "line without any other word."
)

FINAL_ANSWER_INSTRUCTION_SR = """Based on the available information, provide the \
complete name of the top-{top_k} most likely culprit methods for the bug please. \
Since your answer will be processed automatically, please give your answer in \
the format as follows.
Top_1 : file::name
Top_2 : file::name
..."""

FINAL_ANSWER_INSTRUCTION_LR = FINAL_ANSWER_INSTRUCTION_SR

GRAPHLOCATOR_EXPAND_PROMPT = """You are performing GraphLocator-style causal \
issue graph expansion. You are given a confirmed symptom vertex (a method \
directly implicated by failure evidence) and its real callers/callees from \
the repository's call graph. Decide whether any of these neighbors are \
plausibly part of the same causal chain (e.g. a caller that passes bad data \
in, or a callee whose behavior the symptom method depends on). Respond with:
EXPAND: comma-separated file::name entries worth adding (empty if none)"""


def _parse_function_call(response: str) -> tuple[Optional[str], str]:
    """Parses 'FunctionName(Argument)' from a single line, matching the real
    pipeline.py's parsing: strip quotes, split on first '(' / last ')'."""
    line = response.strip().splitlines()[-1] if response.strip() else ""
    cleaned = line.replace("'", "").replace('"', "")
    if "(" not in cleaned or ")" not in cleaned:
        return None, ""
    name = cleaned[: cleaned.find("(")].strip()
    args = cleaned[cleaned.find("(") + 1 : cleaned.rfind(")")].strip().strip("`")
    return name or None, args


def _parse_topk(response: str) -> List[str]:
    """Parses the real 'Top_1 : ...' / 'Top_2 : ...' final-answer format."""
    entries = []
    for line in response.splitlines():
        m = re.match(r"\s*Top_\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            entries.append(m.group(1).strip())
    return entries


def run_react_loop(
    system_prompt: str,
    input_description: str,
    dispatch: Dict[str, "callable"],
    chat_fn,
    max_iters: int,
    final_instruction: str,
) -> str:
    """Mirrors the real FlexFL pipeline.py loop exactly: build a growing
    transcript, ask for one function call per turn, dispatch it, append the
    result, repeat up to max_iters times or until the model calls exit(),
    then ask for the final formatted answer. chat_fn is called once per
    turn with the full transcript as the 'user' content, since our chat_fn
    interface is (system, user) -> response rather than a stateful message
    list — functionally equivalent to the real multi-turn message array."""
    transcript = (
        f"{input_description}\n\n"
        "Let's locate the faulty method step by step using reasoning and function calls. "
        "Now reason and plan how to locate the buggy methods."
    )
    response = chat_fn(system_prompt, transcript)
    transcript += f"\nAssistant: {response}"

    for _ in range(max_iters):
        response = chat_fn(system_prompt, transcript + f"\n{REACT_TURN_INSTRUCTION}")
        transcript += f"\nAssistant: {response}"

        function_name, arguments = _parse_function_call(response)
        if function_name is None:
            transcript += "\nPlease call functions in the right format `FunctionName(Argument)`."
            continue
        if function_name == "exit":
            break
        if function_name not in dispatch:
            transcript += "\nPlease call functions in the right format `FunctionName(Argument)`."
            continue

        try:
            result = dispatch[function_name](arguments)
        except Exception as e:  # a bad argument shouldn't kill the whole run
            result = f"Error calling {function_name}: {e}"
        result_str = "\n".join(result) if isinstance(result, list) else str(result)
        transcript += f"\n{result_str}"

    final_response = chat_fn(system_prompt, transcript + f"\n{final_instruction}")
    return final_response


# Shared with llm_backends so the retry wrapper and this loop agree on what
# counts as "context overflow": the retry wrapper deliberately re-raises
# these instead of retrying, because shrinking MAX is the correct response,
# not trying the same oversized prompt again.
_CONTEXT_ERROR_MARKERS = llm_backends.CONTEXT_ERROR_MARKERS


def run_react_loop_with_adaptive_max(
    system_prompt_template: str,
    input_description: str,
    dispatch: Dict[str, "callable"],
    chat_fn,
    max_iters: int,
    final_instruction_template: str,
    format_kwargs: dict,
    min_iters: int = 2,
) -> str:
    """Real paper behavior (Section 3.2.1): 'If the whole conversation
    exceeds the maximum context length of the used LLM, we decrease the
    value of MAX by 1 and rerun this pipeline.' system_prompt_template and
    final_instruction_template take {max_iters} via format_kwargs so each
    retry re-renders them with the smaller MAX. Bottoms out at min_iters
    rather than retrying forever — a model that can't fit even a minimal
    loop should fail loudly, not silently degrade to something useless."""
    iters = max_iters
    last_error: Optional[Exception] = None
    while iters >= min_iters:
        try:
            system_prompt = system_prompt_template.format(max_iters=iters, **format_kwargs)
            final_instruction = final_instruction_template.format(**format_kwargs)
            return run_react_loop(system_prompt, input_description, dispatch, chat_fn,
                                   iters, final_instruction)
        except Exception as e:
            if not any(marker in str(e).lower() for marker in _CONTEXT_ERROR_MARKERS):
                raise
            last_error = e
            iters -= 1
    raise RuntimeError(
        f"Ran out of MAX reductions (down to {min_iters}) without fitting context: {last_error}"
    )


def _parse_expand(response: str) -> List[str]:
    for line in response.splitlines():
        if line.upper().startswith("EXPAND:"):
            return [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
    return []


# ---------------------------------------------------------------------------
# GraphLocator: symptom vertices + real causal-graph (call-graph) expansion
# ---------------------------------------------------------------------------

def graphlocator_expand(
    symptom_vertices: List[str],
    structure_map: Dict[str, dict],
    call_graph: Dict[str, Dict[str, list]],
    id_to_key: Dict[str, str],
    confirm_fn,  # Callable[[str, List[str]], List[str]] -> vertex_key, neighbor_keys -> confirmed subset
    max_hops: int = MAX_GRAPH_HOPS,
) -> List[str]:
    """Walks REAL 'calls' edges outward from symptom vertices (GraphLocator's
    'symptom vertices locating' phase already done by the caller), expanding
    the causal issue graph hop by hop ('dynamic CIG discovering'). Each
    candidate neighbor is passed through confirm_fn before being added —
    this is where an LLM (or the heuristic stand-in) does the paper's
    'iteratively reasoning over neighboring vertices', rather than adding
    every graph neighbor unconditionally."""
    confirmed = set(symptom_vertices)
    frontier = list(symptom_vertices)

    for _hop in range(max_hops):
        next_frontier = []
        for vertex_key in frontier:
            meta = structure_map.get(vertex_key)
            node_id = meta.get("id") if meta else None
            if not node_id or node_id not in call_graph:
                continue
            neighbor_ids = call_graph[node_id]["callers"] + call_graph[node_id]["callees"]
            neighbor_keys = [id_to_key[nid] for nid in neighbor_ids if nid in id_to_key]
            neighbor_keys = [k for k in neighbor_keys if k not in confirmed]
            if not neighbor_keys:
                continue
            accepted = confirm_fn(vertex_key, neighbor_keys)
            for key in accepted:
                if key not in confirmed:
                    confirmed.add(key)
                    next_frontier.append(key)
        if not next_frontier:
            break
        frontier = next_frontier

    return sorted(confirmed - set(symptom_vertices))


def _heuristic_confirm_fn(problem_statement: str):
    """Key-free stand-in for the LLM's causal-plausibility judgment: accept
    a neighbor only if it has non-trivial lexical overlap with the problem
    statement, so expansion doesn't just accept every call-graph neighbor
    (which would make the 'expansion' meaningless)."""

    def confirm(vertex_key: str, neighbor_keys: List[str]) -> List[str]:
        return [k for k in neighbor_keys if _lexical_overlap_score(problem_statement, k) >= 1]

    return confirm


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

class HeuristicBackend:
    """Deterministic, key-free implementation of the full FlexFL Stage1 ->
    Stage2 -> GraphLocator-expansion pipeline, using the heuristic
    stand-ins documented on each function above. Exists to validate the
    pipeline plumbing and scoring offline; results are tagged
    backend='heuristic' and excluded from formal WP1 conclusions."""

    name = "heuristic"

    def localize(
        self,
        tool_output: str,
        structure_map: Dict[str, dict],
        problem_statement: str,
        call_graph: Optional[Dict[str, Dict[str, list]]] = None,
        repo_root: Optional[Path] = None,
        use_graph: bool = True,
        use_feedback_loop: bool = False,
        raw_tool_output: Optional[str] = None,
        meter: Optional["token_meter.TokenMeter"] = None,
    ) -> LocalizationResult:
        """`use_graph=False` is the graphify ablation: the structural
        briefing and the GraphLocator expansion are both skipped, leaving
        FlexFL's own index-driven search. `use_feedback_loop=True` runs the
        deterministic stand-in verifier from feedback_loop, so the feedback
        ablation is a real variable on this key-free backend too."""
        tools = StructureQueryTools(structure_map, repo_root)
        graph_active = bool(call_graph) and use_graph
        id_to_key = (
            {meta["id"]: key for key, meta in structure_map.items() if meta.get("id")}
            if graph_active else {}
        )

        rounds_used = 0
        fb_restores = fb_prunes = 0
        fb_stop = ""
        if use_feedback_loop and raw_tool_output is not None:
            with (meter.stage(token_meter.STAGE_FEEDBACK) if meter else _nullcontext()):
                fb = run_feedback_loop(raw_tool_output, tool_output,
                                       heuristic_verify_fn(raw_tool_output))
            tool_output = fb.final_text
            rounds_used, fb_restores, fb_prunes = fb.rounds_used, fb.n_restores, fb.n_prunes
            fb_stop = fb.stop_reason
        if meter:
            meter.record_context("agent_input_tokens", tool_output)

        # Graph structural understanding FIRST, before FlexFL's own search —
        # this is the "understand the structure better before FlexFL" step.
        briefing_neighbors: List[str] = []
        if graph_active:
            _briefing_text, briefing_neighbors = graph_structural_briefing(
                tool_output, structure_map, call_graph, id_to_key
            )

        # FlexFL Stage 1, now with the structural neighborhood as a scoring
        # boost rather than something only discovered after the fact
        candidates = stage1_space_reduction(tool_output, problem_statement, tools)
        for key in briefing_neighbors:
            if key not in candidates and key in structure_map:
                candidates.append(key)
        candidates = candidates[:CANDIDATE_LIST_SIZE]

        # FlexFL Stage 2 (heuristic Agent4LR: trust Stage 1's ranking as-is —
        # a real LLM pass is where refinement actually happens; see
        # localize_with_llm for that path)
        refined = candidates

        # GraphLocator's second pass: further expansion from whatever Stage
        # 2 actually confirmed, on top of the pre-search briefing above
        graph_expanded: List[str] = []
        if graph_active:
            symptom_vertices = _symptom_vertices_from_trace(tool_output, structure_map) or refined[:1]
            graph_expanded = graphlocator_expand(
                symptom_vertices, structure_map, call_graph, id_to_key,
                confirm_fn=_heuristic_confirm_fn(problem_statement),
            )

        # Rank order is the metric's input: refined ranking first, causal
        # expansions after it. Sorting here would destroy Top-k and MRR.
        all_functions = _ordered_dedupe(refined, graph_expanded)
        files = metrics.files_from_symbols(
            k for k in all_functions if k in structure_map
        )

        return LocalizationResult(
            instance_id="",
            predicted_files=files,
            predicted_functions=all_functions,
            backend=self.name,
            feedback_rounds_used=rounds_used,
            stage1_candidates=candidates,
            graph_expanded=graph_expanded,
            used_graph=graph_active,
            used_feedback=bool(use_feedback_loop and raw_tool_output is not None),
            feedback_restores=fb_restores,
            feedback_prunes=fb_prunes,
            feedback_stop_reason=fb_stop,
            token_report=meter.report() if meter else {},
        )


def _select_llm_backend() -> Optional[str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"):
        return "qwen"
    return None


def localize_with_llm(
    instance_id: str,
    tool_output: str,
    structure_map: Dict[str, dict],
    problem_statement: str,
    chat_fn,  # Callable[[str, str], str] -> (system, user) -> response text
    backend_name: str,
    call_graph: Optional[Dict[str, Dict[str, list]]] = None,
    repo_root: Optional[Path] = None,
    use_feedback_loop: bool = True,
    raw_tool_output: Optional[str] = None,
    use_graph: bool = True,
    meter: Optional["token_meter.TokenMeter"] = None,
) -> LocalizationResult:
    """Real two-stage FlexFL + GraphLocator expansion, LLM-driven, now
    running the actual multi-turn ReAct loop from the FlexFL replication
    package's pipeline.py rather than a single-shot call per stage.
    chat_fn is provider-agnostic — Claude/GPT/DeepSeek/Qwen all plug in here."""
    tools = StructureQueryTools(structure_map, repo_root)
    meter = meter or token_meter.TokenMeter(token_meter.TokenCounter(backend_name))
    chat_fn = meter.wrap(chat_fn)
    rounds_used = 0
    fb_restores = fb_prunes = 0
    fb_stop = ""
    cache_text = raw_tool_output if raw_tool_output is not None else tool_output
    graph_active = bool(call_graph) and use_graph

    # The feedback loop runs BEFORE the structural briefing, so the briefing
    # is computed over the text the agent will actually reason about. Doing
    # it the other way round would let the briefing cite stack frames that
    # a prune had already removed from the evidence.
    if use_feedback_loop and raw_tool_output is not None:
        def verify_fn(payload: str, round_num: int) -> str:
            return chat_fn(FIDELITY_SYSTEM_PROMPT, payload)

        with meter.stage(token_meter.STAGE_FEEDBACK):
            fb_result = run_feedback_loop(cache_text, tool_output, verify_fn)
        working_text = fb_result.final_text
        rounds_used = fb_result.rounds_used
        fb_restores, fb_prunes = fb_result.n_restores, fb_result.n_prunes
        fb_stop = fb_result.stop_reason
    else:
        working_text = tool_output

    meter.record_context("agent_input_tokens", working_text)

    # Graph structural understanding FIRST — before FlexFL's own search
    # begins, not just as a post-hoc expansion after Agent4LR. This is what
    # lets Agent4SR start its ReAct loop already knowing the structural
    # neighborhood of the observed failure. Skipped entirely in the
    # graphify ablation arm.
    structural_briefing = ""
    id_to_key: Dict[str, str] = {}
    if graph_active:
        id_to_key = {meta["id"]: key for key, meta in structure_map.items() if meta.get("id")}
        structural_briefing, _ = graph_structural_briefing(
            working_text, structure_map, call_graph, id_to_key
        )
        meter.record_context("structural_briefing_tokens", structural_briefing)

    # --- FlexFL Stage 1: Agent4SR, real multi-turn ReAct loop, primed with
    # the structural briefing gathered above ---
    stage1_dispatch = {
        "get_paths": lambda _args: tools.get_paths(),
        "get_classes_of_path": lambda args: tools.get_classes_of_path(args),
        "get_methods_of_class": lambda args: tools.get_methods_of_class(args),
        "find_class": lambda args: tools.find_class(args),
        "find_method": lambda args: tools.find_method(args),
        "get_code_snippet_of_method": lambda args: tools.get_code_snippet_of_method(args) or "not found",
    }
    stage1_input = (
        (f"{structural_briefing}\n\n" if structural_briefing else "")
        + f"The trigger test / tool output is as follows:\n```\n{working_text}\n```\n"
        + f"The bug report is as follows:\n```\n{problem_statement}\n```"
    )
    with meter.stage(token_meter.STAGE_STAGE1):
        stage1_final = run_react_loop_with_adaptive_max(
            AGENT4SR_SYSTEM_PROMPT, stage1_input, stage1_dispatch, chat_fn, MAX_FLEXFL_ITERS,
            FINAL_ANSWER_INSTRUCTION_SR, format_kwargs={"top_k": CANDIDATE_LIST_SIZE},
        )
    candidates = postprocess_topk(_parse_topk(stage1_final), structure_map)
    if not candidates:  # model didn't follow the format — fall back to the non-LLM ranking
        candidates = stage1_space_reduction(working_text, problem_statement, tools)

    # --- FlexFL Stage 2: Agent4LR, real multi-turn ReAct loop over the candidate list ---
    def stage2_get_snippet(args: str):
        try:
            idx = int(args.strip()) - 1
            key = candidates[idx]
        except (ValueError, IndexError):
            return "Invalid method number. Use a number from the suggested list."
        snippet = tools.get_code_snippet_of_method(key) or "(snippet unavailable)"
        return f"The code snippet of {key} is as follows.\n{snippet}"

    stage2_dispatch = {"get_code_snippet_of_method": stage2_get_snippet}
    numbered_candidates = "\n".join(f"{i+1}.{c}" for i, c in enumerate(candidates))
    stage2_input = f"The suggested methods are as follows:\n```\n{numbered_candidates}\n```"
    with meter.stage(token_meter.STAGE_STAGE2):
        stage2_final = run_react_loop_with_adaptive_max(
            AGENT4LR_SYSTEM_PROMPT, stage2_input, stage2_dispatch, chat_fn, MAX_FLEXFL_ITERS,
            FINAL_ANSWER_INSTRUCTION_LR, format_kwargs={"top_k": FINAL_TOPK},
        )
    functions = postprocess_topk(_parse_topk(stage2_final), structure_map)
    if not functions:
        functions = candidates[:FINAL_TOPK]

    # --- GraphLocator expansion over the real call graph ---
    graph_expanded: List[str] = []
    if graph_active:
        def llm_confirm_fn(vertex_key: str, neighbor_keys: List[str]) -> List[str]:
            resp = chat_fn(
                GRAPHLOCATOR_EXPAND_PROMPT,
                f"Symptom vertex: {vertex_key}\nNeighbors: {', '.join(neighbor_keys)}",
            )
            return [k for k in _parse_expand(resp) if k in neighbor_keys]

        with meter.stage(token_meter.STAGE_GRAPH):
            graph_expanded = graphlocator_expand(
                functions[:1] or candidates[:1], structure_map, call_graph, id_to_key,
                confirm_fn=llm_confirm_fn,
            )

    # Ranked, not sorted: Agent4LR's refined ordering is the ranking the
    # Top-k/MRR metrics score, with causal expansions appended behind it.
    functions = _ordered_dedupe(functions, graph_expanded)
    files = metrics.files_from_symbols(k for k in functions if k in structure_map)

    return LocalizationResult(
        instance_id=instance_id,
        predicted_files=files,
        predicted_functions=functions,
        backend=backend_name,
        feedback_rounds_used=rounds_used,
        stage1_candidates=candidates,
        graph_expanded=graph_expanded,
        used_graph=graph_active,
        used_feedback=bool(use_feedback_loop and raw_tool_output is not None),
        feedback_restores=fb_restores,
        feedback_prunes=fb_prunes,
        feedback_stop_reason=fb_stop,
        token_report=meter.report(),
    )


# ---------------------------------------------------------------------------
# Backend construction
#
# Provider wiring lives in llm_backends.py — one registry covering hosted
# APIs (OpenAI, Anthropic, Gemini), hosted open-weight models (DeepSeek,
# Qwen/DashScope, OpenRouter, Together, Groq, Mistral, Ollama Cloud) and
# locally-served open-weight models (Ollama, vLLM, LM Studio, TGI,
# llama.cpp). The helpers below are thin compatibility shims so existing
# call sites keep working.
# ---------------------------------------------------------------------------

def make_chat_fn(provider: str, **kwargs):
    """provider is any name or alias from llm_backends.PROVIDERS."""
    cfg = llm_backends.resolve(provider, **kwargs)
    return llm_backends.build_chat_fn(cfg), cfg


def make_anthropic_chat_fn(model: str = "claude-sonnet-4-6"):
    return llm_backends.build_chat_fn(llm_backends.resolve("anthropic", model=model))


def make_openai_compatible_chat_fn(
    api_key_env: str, base_url: Optional[str] = None, model: str = "gpt-4o"
):
    """Covers every OpenAI-shaped endpoint — hosted or local."""
    return llm_backends.build_chat_fn(
        llm_backends.resolve("custom", model=model, base_url=base_url,
                             api_key_env=api_key_env)
    )


def make_local_gpu_chat_fn(base_url: Optional[str] = None, model: Optional[str] = None):
    """An open-source model served on your own GPU (vLLM by default; Ollama,
    LM Studio, TGI and llama.cpp all work by pointing --base-url at them)."""
    return llm_backends.build_chat_fn(
        llm_backends.resolve(
            "vllm",
            model=model or os.environ.get("LOCAL_LLM_MODEL"),
            base_url=base_url or os.environ.get("LOCAL_LLM_BASE_URL"),
        )
    )
