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

from feedback_loop import run_feedback_loop

MAX_FLEXFL_ITERS = 10          # real FlexFL pipeline.py: max_try = 10
MAX_GRAPH_HOPS = 2              # GraphLocator: bounded CIG expansion, not unbounded traversal
CANDIDATE_LIST_SIZE = 20        # real FlexFL rank.py: suspicious_methods = suspicious_methods[:20]
FINAL_TOPK = 5                  # real FlexFL pipeline.py: "top-5 most likely culprit methods"

FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in ([A-Za-z_][A-Za-z0-9_]*)')


@dataclass
class LocalizationResult:
    instance_id: str
    predicted_files: List[str]
    predicted_functions: List[str]
    backend: str
    feedback_rounds_used: int = 0
    stage1_candidates: List[str] = field(default_factory=list)   # FlexFL Agent4SR output
    graph_expanded: List[str] = field(default_factory=list)      # GraphLocator CIG additions


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
    first (splitting on '.' the way the real one splits on '.'/'$'), and
    only when that returns nothing does it fall back to Levenshtein
    distance <= 5, else the closest max_results choices. This is what makes
    find_class/find_method/get_methods_of_class resilient to a slightly
    wrong name instead of just returning nothing."""
    query_tokens = re.split(r"[.:]+", query.lower())
    exact = [c for c in choices if all(t in c.lower() for t in query_tokens if t)]
    if exact:
        return exact

    distances = sorted(((c, _levenshtein(query.lower(), c.lower())) for c in choices),
                        key=lambda t: t[1])
    close = [c for c, d in distances if d <= 5]
    return close if close else [c for c, _ in distances[:max_results]]


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
    ) -> LocalizationResult:
        tools = StructureQueryTools(structure_map, repo_root)

        # FlexFL Stage 1
        candidates = stage1_space_reduction(tool_output, problem_statement, tools)

        # FlexFL Stage 2 (heuristic Agent4LR: trust Stage 1's ranking as-is —
        # a real LLM pass is where refinement actually happens; see
        # localize_with_llm for that path)
        refined = candidates

        # GraphLocator expansion over the REAL call graph, symptom vertices
        # = the trace-evidenced subset of the refined candidates
        graph_expanded: List[str] = []
        if call_graph:
            id_to_key = {
                meta["id"]: key for key, meta in structure_map.items() if meta.get("id")
            }
            symptom_vertices = [
                key for key in refined
                if any(
                    m.group(3).lower() in key.lower()
                    for m in FRAME_RE.finditer(tool_output)
                )
            ] or refined[:1]
            graph_expanded = graphlocator_expand(
                symptom_vertices, structure_map, call_graph, id_to_key,
                confirm_fn=_heuristic_confirm_fn(problem_statement),
            )

        all_functions = sorted(set(refined) | set(graph_expanded))
        files = sorted({structure_map[k]["file"] for k in all_functions if k in structure_map})

        return LocalizationResult(
            instance_id="",
            predicted_files=files,
            predicted_functions=all_functions,
            backend=self.name,
            stage1_candidates=candidates,
            graph_expanded=graph_expanded,
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
) -> LocalizationResult:
    """Real two-stage FlexFL + GraphLocator expansion, LLM-driven, now
    running the actual multi-turn ReAct loop from the FlexFL replication
    package's pipeline.py rather than a single-shot call per stage.
    chat_fn is provider-agnostic — Claude/GPT/DeepSeek/Qwen all plug in here."""
    tools = StructureQueryTools(structure_map, repo_root)
    rounds_used = 0
    cache_text = raw_tool_output if raw_tool_output is not None else tool_output

    if use_feedback_loop:
        def verify_fn(current_text: str, round_num: int) -> str:
            return chat_fn(
                "You just localized a bug from a possibly-compressed tool output. "
                "If it contains enough evidence, respond exactly: OK\n"
                "If something critical looks missing/truncated, respond exactly: "
                "MISSING: L<start>-L<end> <reason>",
                f"CURRENT TEXT:\n{current_text}",
            )

        fb_result = run_feedback_loop(cache_text, tool_output, verify_fn)
        working_text = fb_result.final_text
        rounds_used = fb_result.rounds_used
    else:
        working_text = tool_output

    # --- FlexFL Stage 1: Agent4SR, real multi-turn ReAct loop ---
    stage1_dispatch = {
        "get_paths": lambda _args: tools.get_paths(),
        "get_classes_of_path": lambda args: tools.get_classes_of_path(args),
        "get_methods_of_class": lambda args: tools.get_methods_of_class(args),
        "find_class": lambda args: tools.find_class(args),
        "find_method": lambda args: tools.find_method(args),
        "get_code_snippet_of_method": lambda args: tools.get_code_snippet_of_method(args) or "not found",
    }
    stage1_input = f"The trigger test / tool output is as follows:\n```\n{working_text}\n```\n" \
                    f"The bug report is as follows:\n```\n{problem_statement}\n```"
    stage1_system = AGENT4SR_SYSTEM_PROMPT.format(
        top_k=CANDIDATE_LIST_SIZE, max_iters=MAX_FLEXFL_ITERS
    )
    stage1_final = run_react_loop(
        stage1_system, stage1_input, stage1_dispatch, chat_fn, MAX_FLEXFL_ITERS,
        FINAL_ANSWER_INSTRUCTION_SR.format(top_k=CANDIDATE_LIST_SIZE),
    )
    candidates = _parse_topk(stage1_final)
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
    stage2_system = AGENT4LR_SYSTEM_PROMPT.format(max_iters=MAX_FLEXFL_ITERS)
    stage2_final = run_react_loop(
        stage2_system, stage2_input, stage2_dispatch, chat_fn, MAX_FLEXFL_ITERS,
        FINAL_ANSWER_INSTRUCTION_LR.format(top_k=FINAL_TOPK),
    )
    functions = _parse_topk(stage2_final)
    if not functions:
        functions = candidates[:FINAL_TOPK]
    files = sorted({structure_map[k]["file"] for k in functions if k in structure_map})

    # --- GraphLocator expansion over the real call graph ---
    graph_expanded: List[str] = []
    if call_graph:
        id_to_key = {meta["id"]: key for key, meta in structure_map.items() if meta.get("id")}

        def llm_confirm_fn(vertex_key: str, neighbor_keys: List[str]) -> List[str]:
            resp = chat_fn(
                GRAPHLOCATOR_EXPAND_PROMPT,
                f"Symptom vertex: {vertex_key}\nNeighbors: {', '.join(neighbor_keys)}",
            )
            return [k for k in _parse_expand(resp) if k in neighbor_keys]

        graph_expanded = graphlocator_expand(
            functions[:1] or candidates[:1], structure_map, call_graph, id_to_key,
            confirm_fn=llm_confirm_fn,
        )
        functions = sorted(set(functions) | set(graph_expanded))
        files = sorted({structure_map[k]["file"] for k in functions if k in structure_map})

    return LocalizationResult(
        instance_id=instance_id,
        predicted_files=files,
        predicted_functions=functions,
        backend=backend_name,
        feedback_rounds_used=rounds_used,
        stage1_candidates=candidates,
        graph_expanded=graph_expanded,
    )


def make_anthropic_chat_fn(model: str = "claude-sonnet-4-6"):
    import anthropic

    client = anthropic.Anthropic()

    def chat_fn(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))

    return chat_fn


def make_openai_compatible_chat_fn(
    api_key_env: str, base_url: Optional[str] = None, model: str = "gpt-4o"
):
    """Covers OpenAI, DeepSeek, and Qwen (DashScope's OpenAI-compatible
    endpoint) — all three speak the same chat-completions shape."""
    import openai

    client = openai.OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def chat_fn(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    return chat_fn


def make_local_gpu_chat_fn(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
):
    """For running an open-source model (DeepSeek, Qwen, etc.) locally on
    your own NVIDIA GPU instead of a cloud API — the project's stated
    comparison includes open-source models, and a local vLLM/Ollama/TGI
    server is the natural way to run those on your own hardware.

    Any OpenAI-compatible local server works (vLLM's `vllm serve`, Ollama's
    `/v1` endpoint, TGI's OpenAI-compatible mode) since they all implement
    the same chat-completions shape as make_openai_compatible_chat_fn above
    — this is a thin convenience wrapper with GPU-friendly defaults:

        LOCAL_LLM_BASE_URL   default http://localhost:8000/v1 (vLLM's default)
        LOCAL_LLM_MODEL      default reads from env, no built-in default —
                              must match whatever you served (e.g. the repo
                              id you passed to `vllm serve`)

    No API key needed — local servers typically don't require one, so this
    passes a dummy key ("local") since the openai SDK requires the field
    to be non-empty even when the server ignores it.
    """
    import openai

    resolved_base_url = base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    resolved_model = model or os.environ.get("LOCAL_LLM_MODEL")
    if not resolved_model:
        raise ValueError(
            "No model specified. Pass model=..., or set LOCAL_LLM_MODEL to "
            "whatever you served, e.g. 'Qwen/Qwen2.5-Coder-32B-Instruct' or "
            "'deepseek-ai/DeepSeek-Coder-V2-Instruct'."
        )

    client = openai.OpenAI(api_key="local", base_url=resolved_base_url)

    def chat_fn(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    return chat_fn
