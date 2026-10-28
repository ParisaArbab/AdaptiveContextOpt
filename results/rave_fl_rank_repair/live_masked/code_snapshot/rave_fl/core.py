"""RAVE-FL MVP: risk-aware evidence allocation for function fault localization.

Dependency-free core. The base localizer provides candidate functions and evidence;
RAVE-FL chooses whether to screen a batch or verify one function in full.
No oracle labels or ground truth are used in policy decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import exp
from typing import Callable, Iterable


class Action(str, Enum):
    BATCH_SCREEN = "BATCH_SCREEN"
    EXPAND_CONTEXT = "EXPAND_CONTEXT"
    FULL_VERIFY = "FULL_VERIFY"
    STOP = "STOP"


@dataclass(frozen=True)
class Candidate:
    function_id: str
    signature: str
    body: str = ""
    semantic_score: float = 0.0
    coverage_score: float = 0.0
    calls: tuple[str, ...] = ()
    complexity: float = 0.0


@dataclass(frozen=True)
class Sketch:
    function_id: str
    level: int
    text: str
    approx_tokens: int


@dataclass
class Evidence:
    function_id: str
    verdict: float
    reason: str
    tokens_used: int
    uncertainty: float
    action: Action


@dataclass
class State:
    remaining_tokens: int
    evidence: list[Evidence] = field(default_factory=list)
    inspected: set[str] = field(default_factory=set)
    spent_tokens: int = 0

    def spend(self, tokens: int) -> None:
        if tokens < 0 or tokens > self.remaining_tokens:
            raise ValueError("Token budget exceeded or negative usage")
        self.remaining_tokens -= tokens
        self.spent_tokens += tokens


def approx_tokens(text: str) -> int:
    """Simple stable approximation for planning, not a model tokenizer."""
    return max(1, (len(text) + 3) // 4) if text else 0


def sketch(candidate: Candidate, level: int) -> Sketch:
    """L0 signature; L1 adds calls; L2 includes bounded source body."""
    if level not in (0, 1, 2):
        raise ValueError("Sketch level must be 0, 1, or 2")
    text = candidate.signature
    if level >= 1:
        text += "\nCalls: " + ", ".join(candidate.calls)
    if level == 2:
        text += "\n" + candidate.body
    return Sketch(candidate.function_id, level, text, approx_tokens(text))


def features(c: Candidate, current_rank: int, next_score: float = 0.0) -> dict[str, float]:
    return {
        "semantic_score": c.semantic_score,
        "coverage_score": c.coverage_score,
        "complexity": c.complexity,
        "call_count": float(len(c.calls)),
        "source_token_cost": float(approx_tokens(c.body)),
        "rank": float(current_rank),
        "ranking_gap": max(0.0, c.semantic_score - next_score),
    }


def heuristic_rank_risk(c: Candidate) -> float:
    """Placeholder risk estimator until an empirically trained model is available."""
    z = -1.2 + 1.8 * c.semantic_score + 1.2 * c.coverage_score + 0.12 * min(c.complexity, 10)
    return 1.0 / (1.0 + exp(-z))


Verifier = Callable[[list[Sketch], Action], list[Evidence]]


def default_verifier(sketches: list[Sketch], action: Action) -> list[Evidence]:
    """Deterministic smoke-test stub, NOT an LLM verification."""
    return [
        Evidence(s.function_id, 0.0, "No verification backend configured", 0, 1.0, action)
        for s in sketches
    ]


class RaveFL:
    def __init__(
        self,
        candidates: Iterable[Candidate],
        token_budget: int,
        verifier: Verifier = default_verifier,
        risk_threshold: float = 0.7,
        batch_size: int = 3,
    ):
        if token_budget < 0 or batch_size < 1 or not 0 <= risk_threshold <= 1:
            raise ValueError("Invalid policy configuration")
        self.candidates = sorted(
            list(candidates), key=lambda c: (c.semantic_score + c.coverage_score), reverse=True
        )
        self.state = State(token_budget)
        self.verifier = verifier
        self.risk_threshold = risk_threshold
        self.batch_size = batch_size

    def _run(self, selected: list[Candidate], level: int, action: Action) -> bool:
        sketches = [sketch(c, level) for c in selected]
        input_cost = sum(s.approx_tokens for s in sketches)
        if input_cost > self.state.remaining_tokens:
            return False
        # The verifier reports actual output/model usage where available.
        # Input is reserved first and output must fit remaining budget.
        self.state.spend(input_cost)
        results = self.verifier(sketches, action)
        if {e.function_id for e in results} != {c.function_id for c in selected}:
            raise ValueError("Verifier must report one result for each selected candidate")
        output_cost = sum(e.tokens_used for e in results)
        if output_cost > self.state.remaining_tokens or output_cost < 0:
            raise ValueError("Verifier exceeded output token budget")
        self.state.spend(output_cost)
        self.state.evidence.extend(results)
        self.state.inspected.update(c.function_id for c in selected)
        return True

    def step(self) -> Action:
        remaining = [c for c in self.candidates if c.function_id not in self.state.inspected]
        if not remaining or self.state.remaining_tokens <= 0:
            return Action.STOP
        highest = remaining[0]
        if heuristic_rank_risk(highest) >= self.risk_threshold:
            if self._run([highest], 2, Action.FULL_VERIFY):
                return Action.FULL_VERIFY
            if self._run([highest], 1, Action.EXPAND_CONTEXT):
                return Action.EXPAND_CONTEXT
        if self._run(remaining[: self.batch_size], 0, Action.BATCH_SCREEN):
            return Action.BATCH_SCREEN
        return Action.STOP

    def execute(self, max_steps: int = 100) -> dict:
        actions: list[str] = []
        for _ in range(max_steps):
            action = self.step()
            actions.append(action.value)
            if action == Action.STOP:
                break
        verdicts = {e.function_id: e.verdict for e in self.state.evidence}
        ranking = sorted(
            self.candidates,
            key=lambda c: (
                verdicts.get(c.function_id, 0.0),
                c.semantic_score + c.coverage_score,
            ),
            reverse=True,
        )
        return {
            "ranked_functions": [c.function_id for c in ranking],
            "actions": actions,
            "tokens_used": self.state.spent_tokens,
            "tokens_remaining": self.state.remaining_tokens,
            "evidence": [
                {"function_id": e.function_id, "verdict": e.verdict,
                 "reason": e.reason, "uncertainty": e.uncertainty,
                 "action": e.action.value}
                for e in self.state.evidence
            ],
        }
