"""Gold-free evidence-guided feedback for adaptive context optimization.

The feedback controller never receives fault-localization ground truth. It
first asks which concrete evidence is still missing, retrieves that evidence
without changing LeanCTX density, and only falls back to a higher density when
targeted retrieval cannot satisfy the request.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re

from wp1.llm_backends import ChatBackend


ALLOWED_EVIDENCE_TYPES = {
    "runtime_detail",
    "source_snippet",
    "inheritance_chain",
    "file_structure",
}


FEEDBACK_SYSTEM = """You are an evidence-sufficiency evaluator for software fault localization.

You are NOT the localization agent and you are NOT given the gold faulty file,
method, class, patch, or answer. Never guess, request, or use gold information.

Your job is to identify the most important unresolved evidence that would make
the current localization better supported.

Prefer TARGETED_EXPAND over restoring more of the whole runtime context.
Request only evidence that is not already present in the runtime context,
Graphify transcript, or targeted-evidence ledger.

Valid evidence types:
- runtime_detail: a specific missing test/error/assertion/traceback detail.
- source_snippet: source for a concrete file::entity.
- inheritance_chain: the definition of a concrete class and its direct parents.
- file_structure: production entities in a concrete source file.

For source requests, anchor_entity must be a concrete path.py::Entity whenever
that path/entity is already known. For runtime_detail, anchor_entity should be a
specific test name, exception, assertion token, or other diagnostic anchor.

Return ONLY one JSON object with exactly these top-level keys:
{
  "decision": "STOP" | "TARGETED_EXPAND" | "EXPAND_DENSITY",
  "reason": "one short sentence",
  "missing_evidence": [
    {
      "evidence_type": "runtime_detail | source_snippet | inheritance_chain | file_structure",
      "anchor_entity": "specific evidence anchor",
      "question": "specific unresolved question",
      "why_needed": "why this evidence can distinguish the current candidates"
    }
  ]
}

Rules:
- STOP only when the current ranking is sufficiently supported.
- TARGETED_EXPAND requires at least one concrete missing_evidence item.
- EXPAND_DENSITY is a last resort only when the needed information cannot be
  requested concretely.
- Never use unknown gold correctness as a reason to STOP or expand.
"""


@dataclass
class FeedbackDecision:
    decision: str
    reason: str
    raw_response: str
    source: str = "llm"
    missing_evidence: list[dict] = field(default_factory=list)

    @property
    def expand(self) -> bool:
        return self.decision != "STOP"

    @property
    def targeted(self) -> bool:
        return self.decision == "TARGETED_EXPAND"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_request(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    evidence_type = str(item.get("evidence_type", "")).strip().lower()
    anchor = str(item.get("anchor_entity", "")).strip()
    question = str(item.get("question", "")).strip()
    why_needed = str(item.get("why_needed", "")).strip()
    if evidence_type not in ALLOWED_EVIDENCE_TYPES:
        return None
    if not anchor or not question or not why_needed:
        return None
    return {
        "evidence_type": evidence_type,
        "anchor_entity": anchor,
        "question": question,
        "why_needed": why_needed,
    }


def _json_object(text: str) -> dict | None:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_feedback_decision(text: str) -> FeedbackDecision:
    payload = _json_object(text)
    if payload is not None:
        decision = str(payload.get("decision", "")).strip().upper()
        reason = str(payload.get("reason", "")).strip() or "No reason supplied."
        requests = [
            request
            for request in (
                _clean_request(item)
                for item in (payload.get("missing_evidence") or [])
            )
            if request is not None
        ]

        if decision == "STOP":
            return FeedbackDecision(
                decision="STOP",
                reason=reason,
                raw_response=text or "",
                missing_evidence=[],
            )

        if decision == "TARGETED_EXPAND":
            if requests:
                return FeedbackDecision(
                    decision="TARGETED_EXPAND",
                    reason=reason,
                    raw_response=text or "",
                    missing_evidence=requests,
                )
            return FeedbackDecision(
                decision="EXPAND_DENSITY",
                reason=(
                    "Targeted feedback did not name concrete retrievable evidence; "
                    "density expansion is the safe fallback."
                ),
                raw_response=text or "",
                source="validation_fallback",
            )

        if decision == "EXPAND_DENSITY":
            return FeedbackDecision(
                decision="EXPAND_DENSITY",
                reason=reason,
                raw_response=text or "",
                missing_evidence=requests,
            )

    # Backward-compatible parser for historical two-line feedback.
    match = re.search(
        r"(?im)^\s*Decision\s*:\s*(STOP|EXPAND|TARGETED_EXPAND|EXPAND_DENSITY)\s*$",
        text or "",
    )
    reason_match = re.search(r"(?im)^\s*Reason\s*:\s*(.+?)\s*$", text or "")
    if match:
        decision = match.group(1).upper()
        if decision == "EXPAND":
            decision = "EXPAND_DENSITY"
        return FeedbackDecision(
            decision=decision,
            reason=(
                reason_match.group(1).strip()
                if reason_match
                else "No reason supplied."
            ),
            raw_response=text or "",
            source="legacy_parser",
        )

    return FeedbackDecision(
        decision="EXPAND_DENSITY",
        reason=(
            "Feedback response was not parseable, so density expansion is used "
            "as the safe fallback."
        ),
        raw_response=text or "",
        source="parse_fallback",
    )


def normalize_density_schedule(values: list[float] | tuple[float, ...]) -> list[float]:
    densities = sorted({round(float(value), 6) for value in values})
    if not densities:
        raise ValueError("density schedule cannot be empty")
    if densities[0] <= 0.0 or densities[-1] > 1.0:
        raise ValueError("densities must satisfy 0 < density <= 1")
    return densities


def next_density(current: float, schedule: list[float] | tuple[float, ...]) -> float | None:
    for density in normalize_density_schedule(schedule):
        if density > current + 1e-9:
            return density
    return None


def _request_key(request: dict) -> str:
    return "|".join(
        [
            str(request.get("evidence_type", "")).strip().lower(),
            str(request.get("anchor_entity", "")).strip().lower(),
            str(request.get("question", "")).strip().lower(),
        ]
    )


def _runtime_excerpt(raw_output: str, anchor: str, radius: int = 3) -> str:
    lines = (raw_output or "").splitlines()
    if not lines:
        return "No runtime output available."

    anchor = (anchor or "").strip()
    needles = [anchor]
    if "::" in anchor:
        needles.append(anchor.rsplit("::", 1)[-1])
    if "/" in anchor:
        needles.append(Path(anchor).name)

    matches: list[int] = []
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(needle and needle.lower() in lower for needle in needles):
            matches.append(i)

    if not matches:
        return f"No runtime lines matched anchor: {anchor}"

    selected: set[int] = set()
    for hit in matches[:6]:
        for i in range(max(0, hit - radius), min(len(lines), hit + radius + 1)):
            selected.add(i)

    return "\n".join(f"{i + 1}: {lines[i]}" for i in sorted(selected))


def _find_class_node(tree: ast.AST, class_name: str) -> ast.ClassDef | None:
    simple = class_name.split(".")[-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == simple:
            return node
    return None


def _resolve_imported_class_ref(
    graph,
    source_file: str,
    tree: ast.AST,
    class_name: str,
) -> str | None:
    repo = Path(graph.repo)
    simple = class_name.split(".")[-1]

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            visible = alias.asname or alias.name
            if visible != simple:
                continue

            module = node.module or ""
            if node.level:
                base = (repo / source_file).parent
                for _ in range(max(0, node.level - 1)):
                    base = base.parent
                module_path = base / module.replace(".", "/")
            else:
                module_path = repo / module.replace(".", "/")

            candidates = [
                module_path.with_suffix(".py"),
                module_path / "__init__.py",
            ]
            for candidate in candidates:
                if candidate.exists():
                    try:
                        rel = candidate.relative_to(repo).as_posix()
                    except ValueError:
                        continue
                    return f"{rel}::{alias.name.split('.')[-1]}"

    # Graphify fallback when import resolution is unavailable.
    for node in graph.nodes:
        label = str(node.label or "").replace("()", "").lstrip(".")
        tail = label.split(".")[-1]
        classish = str(node.callable_class or "").replace("()", "").lstrip(".")
        class_tail = classish.split(".")[-1] if classish else ""
        if simple not in {tail, class_tail}:
            continue
        path = str(node.source_file or "").replace("\\", "/").lstrip("./")
        if path.endswith(".py"):
            return f"{path}::{simple}"
    return None


def _inheritance_evidence(graph, anchor: str) -> str:
    if "::" not in anchor:
        return graph.snippet(anchor)

    source_file, entity = anchor.split("::", 1)
    source_file = source_file.strip().replace("\\", "/").lstrip("./")
    class_name = entity.strip().split(".")[0]
    path = Path(graph.repo) / source_file

    primary = graph.snippet(f"{source_file}::{class_name}")
    if not path.exists() or path.suffix != ".py":
        return primary

    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except Exception:
        return primary

    node = _find_class_node(tree, class_name)
    if node is None:
        return primary

    parent_names = []
    for base in node.bases:
        try:
            name = ast.unparse(base)
        except Exception:
            name = ""
        if name:
            parent_names.append(name)

    sections = [f"ANCHOR CLASS:\n{primary}"]
    for parent_name in parent_names[:5]:
        ref = _resolve_imported_class_ref(
            graph,
            source_file,
            tree,
            parent_name,
        )
        if ref:
            sections.append(
                f"DIRECT PARENT {parent_name} ({ref}):\n{graph.snippet(ref)}"
            )
        else:
            sections.append(
                f"DIRECT PARENT {parent_name}: source reference could not be "
                "resolved automatically."
            )
    return "\n\n".join(sections)


def _file_structure(graph, anchor: str) -> str:
    path = anchor.split("::", 1)[0].strip().replace("\\", "/").lstrip("./")
    values = []
    seen = set()
    for node in graph.nodes:
        node_path = str(node.source_file or "").replace("\\", "/").lstrip("./")
        if node_path != path:
            continue
        label = str(node.label or "")
        if label in seen:
            continue
        seen.add(label)
        values.append(label)
        if len(values) >= 60:
            break
    if not values:
        return f"No Graphify entities found for {path}"
    return "\n".join(f"{i}. {value}" for i, value in enumerate(values, 1))


def retrieve_targeted_evidence(
    graph,
    requests: list[dict],
    *,
    raw_runtime_output: str,
    existing_keys: set[str] | None = None,
) -> list[dict]:
    """Retrieve concrete evidence without changing LeanCTX density."""
    existing = existing_keys if existing_keys is not None else set()
    records: list[dict] = []

    for request in requests:
        cleaned = _clean_request(request)
        if cleaned is None:
            continue

        key = _request_key(cleaned)
        if key in existing:
            continue

        evidence_type = cleaned["evidence_type"]
        anchor = cleaned["anchor_entity"]

        try:
            if evidence_type == "runtime_detail":
                content = _runtime_excerpt(raw_runtime_output, anchor)
                retrieval = "raw_runtime_excerpt"
            elif evidence_type == "inheritance_chain":
                content = _inheritance_evidence(graph, anchor)
                retrieval = "graphify_inheritance"
            elif evidence_type == "file_structure":
                content = _file_structure(graph, anchor)
                retrieval = "graphify_file_structure"
            else:
                content = graph.snippet(anchor)
                retrieval = "graphify_source_snippet"
        except Exception as exc:
            content = f"Targeted retrieval failed: {type(exc).__name__}: {exc}"
            retrieval = "retrieval_error"

        if (
            not content
            or content.startswith("No source")
            or content.startswith("No Graphify")
            or content.startswith("Targeted retrieval failed")
        ):
            continue

        record = {
            **cleaned,
            "key": key,
            "retrieval": retrieval,
            "content": content,
        }
        records.append(record)
        existing.add(key)

    return records


def _ledger_text(evidence_ledger: list[dict] | None) -> str:
    if not evidence_ledger:
        return "(none)"
    rows = []
    for i, item in enumerate(evidence_ledger[-8:], 1):
        rows.append(
            f"{i}. type={item.get('evidence_type')} "
            f"anchor={item.get('anchor_entity')}\n"
            f"Question: {item.get('question')}\n"
            f"Retrieved evidence:\n{str(item.get('content', ''))[:1800]}"
        )
    return "\n\n".join(rows)


def evaluate_context_sufficiency(
    backend: ChatBackend,
    *,
    problem: str,
    failing_test: str,
    runtime_output: str,
    agent_result: dict,
    current_density: float,
    next_density_value: float | None,
    feedback_round: int,
    max_feedback_rounds: int,
    evidence_ledger: list[dict] | None = None,
) -> FeedbackDecision:
    """Return a gold-free evidence-guided feedback decision."""
    predictions = list(agent_result.get("predictions") or [])
    transcript = list(agent_result.get("transcript") or [])
    tools_used = set(agent_result.get("tools_used") or [])
    tool_calls = int(agent_result.get("tool_calls") or 0)

    if len(predictions) < 5:
        requests = []
        for candidate in predictions[:3]:
            if "::" in candidate:
                requests.append(
                    {
                        "evidence_type": "source_snippet",
                        "anchor_entity": candidate,
                        "question": (
                            "What source evidence supports or contradicts this "
                            "current localization candidate?"
                        ),
                        "why_needed": (
                            "The localization run ended without five valid "
                            "candidates, so the current ranking needs stronger "
                            "production-code evidence."
                        ),
                    }
                )
        if requests:
            return FeedbackDecision(
                decision="TARGETED_EXPAND",
                reason=(
                    "Agent4SR did not return five valid candidates; inspect the "
                    "current production candidates before changing density."
                ),
                raw_response="",
                source="hard_guard",
                missing_evidence=requests,
            )
        return FeedbackDecision(
            decision="EXPAND_DENSITY",
            reason=(
                "Agent4SR did not return five valid candidates and no concrete "
                "source anchor was available for targeted retrieval."
            ),
            raw_response="",
            source="hard_guard",
        )

    if tool_calls < 2 or "get_code_snippet" not in tools_used:
        top = next((p for p in predictions if "::" in p), "")
        if top:
            return FeedbackDecision(
                decision="TARGETED_EXPAND",
                reason=(
                    "The ranking lacks enough real source evidence; inspect the "
                    "leading production candidate before changing density."
                ),
                raw_response="",
                source="hard_guard",
                missing_evidence=[
                    {
                        "evidence_type": "source_snippet",
                        "anchor_entity": top,
                        "question": (
                            "Does the leading candidate contain source evidence "
                            "that explains the observed failure?"
                        ),
                        "why_needed": (
                            "A supported localization requires real source "
                            "evidence rather than ranking from context alone."
                        ),
                    }
                ],
            )

    history_lines: list[str] = []
    for index, item in enumerate(transcript[-8:], 1):
        assistant = str(item.get("assistant", ""))[:600]
        tool = str(item.get("tool", ""))[:1600]
        history_lines.append(
            f"Step {index} tool request:\n{assistant}\n"
            f"Actual Graphify result:\n{tool}"
        )

    next_text = (
        f"{next_density_value:.2f}"
        if next_density_value is not None
        else "(none; RAW/current maximum)"
    )

    prompt = f"""SWE-BENCH PROBLEM:
{problem}

FAILING TEST:
{failing_test}

CURRENT LEANCTX TARGET DENSITY: {current_density:.2f}
NEXT AVAILABLE DENSITY IF TARGETED RETRIEVAL FAILS: {next_text}
FEEDBACK ROUND: {feedback_round}/{max_feedback_rounds}

CURRENT RUNTIME CONTEXT:
{runtime_output}

CURRENT AGENT4SR TOP-5:
""" + "\n".join(
        f"{i}. {p}" for i, p in enumerate(predictions, 1)
    ) + """

REAL GRAPHIFY TOOL EVIDENCE:
""" + "\n\n".join(history_lines) + """

TARGETED EVIDENCE ALREADY RETRIEVED:
""" + _ledger_text(evidence_ledger) + """

Identify the single most useful unresolved evidence need. Prefer a concrete
TARGETED_EXPAND request. Do not request evidence already shown above. Use
EXPAND_DENSITY only when no concrete evidence target can be named.
"""

    response = backend.complete(FEEDBACK_SYSTEM, prompt)
    return parse_feedback_decision(response)
