"""Explicit contracts for the adaptive pilot (gold never enters these objects)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from rave_fl.core import Action


@dataclass(frozen=True)
class PilotCandidate:
    function_id: str
    source: str
    prior: float
    rank: int
    levels: tuple[str, str, str]
    features: dict[str, float]
    sketch_metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    function_id: str
    suspicion: float
    uncertainty: float
    reason: str
    valid: bool = True
    missing_evidence: str = ""

    def __post_init__(self):
        if not all(math.isfinite(v) and 0 <= v <= 1
                   for v in (self.suspicion, self.uncertainty)):
            raise ValueError("Scores must be finite numbers in [0, 1]")


@dataclass
class CallResult:
    observations: list[Observation]
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    raw_response: str = ""
    error: str | None = None

    def __post_init__(self):
        for value in (self.input_tokens, self.output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Provider usage must be nonnegative integer counts")
        if not math.isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise ValueError("Invalid call duration")

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


@dataclass
class CandidateState:
    level: int = -1
    observation: Observation | None = None
    history: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    action: Action
    ids: tuple[str, ...] = ()
    level: int = -1
    reason: str = ""
    estimated_tokens: int = 0
    risks: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyConfig:
    token_budget: int = 16000
    batch_size: int = 4
    risk_threshold: float = 0.35
    uncertainty_threshold: float = 0.45
    suspicion_threshold: float = 0.50
    stop_gap: float = 0.25
    early_stop: bool = True
    max_steps: int = 200
    mode: str = "adaptive"

    def __post_init__(self):
        if self.token_budget < 0 or self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("Invalid budget, batch size, or step limit")
        if self.mode not in {"adaptive", "confidence", "full", "fixed_batch", "fixed_sketch", "rank_threshold", "probe_l0",
                             "repair_top", "repair_disagreement", "repair_suspicion", "repair_utility"}:
            raise ValueError("Unknown policy mode")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (
                self.risk_threshold, self.uncertainty_threshold,
                self.suspicion_threshold, self.stop_gap)):
            raise ValueError("Policy thresholds must be in [0, 1]")


def record(value):
    return asdict(value)
