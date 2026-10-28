"""Typed online contracts. Offline labels deliberately have a different type."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re


class CompatibilityError(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(value):
    return (
        re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:90]
        + "_"
        + hashlib.sha256(value.encode()).hexdigest()[:10]
    )


def relative_path(value):
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise CompatibilityError("Source paths must be nonempty repository-relative paths")
    return path.as_posix()


def beneath(root, value):
    root = Path(root).resolve()
    path = (root / relative_path(value)).resolve()
    if not path.is_relative_to(root):
        raise CompatibilityError("Source symlink escapes the declared checkout")
    return path


@dataclass(frozen=True)
class Evidence:
    kind: str
    text: str
    source: str = ""
    sha256: str = ""

    def __post_init__(self):
        if self.kind not in {"observed_failure", "trigger_description", "issue_only"}:
            raise CompatibilityError(
                "Evidence kind must identify observed_failure, trigger_description, or issue_only"
            )
        if not isinstance(self.text, str) or not self.text.strip():
            raise CompatibilityError(
                "No usable issue/failure evidence; supply evidence_file or an issue description"
            )
        actual = hashlib.sha256(self.text.encode()).hexdigest()
        if self.sha256 and self.sha256 != actual:
            raise CompatibilityError("Evidence checksum mismatch")
        object.__setattr__(self, "sha256", actual)


@dataclass(frozen=True)
class OnlineInstance:
    dataset: str
    version: str
    instance_id: str
    repository: str
    language: str
    revision: str
    checkout: str
    issue: str
    evidence: Evidence
    candidate_file: str = ""
    flexfl_repo: str = ""
    source_root: str = "."
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if any(
            not isinstance(v, str) or not v.strip()
            for v in [self.dataset, self.version, self.instance_id, self.repository, self.language]
        ):
            raise CompatibilityError(
                "Dataset/version/ID/repository/language must be nonempty strings"
            )
        if self.revision and not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise CompatibilityError(
                "Use the full immutable buggy/base commit SHA, not HEAD or a mutable branch"
            )
        if self.source_root != ".":
            relative_path(self.source_root)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, "evidence": Evidence(**value["evidence"])})


@dataclass(frozen=True)
class CodeEntity:
    entity_id: str
    path: str
    start_line: int
    end_line: int
    source: str
    prior: float
    ranks: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    kind: str = "function"

    def __post_init__(self):
        relative_path(self.path)
        if (
            not self.entity_id
            or not self.source.strip()
            or self.kind not in {"file", "function", "method", "class"}
        ):
            raise CompatibilityError("Candidate needs an ID, source and supported entity kind")
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or not 1 <= self.start_line <= self.end_line
        ):
            raise CompatibilityError(
                "Candidate source span must use positive ordered integer lines"
            )
        if (
            not isinstance(self.prior, (int, float))
            or isinstance(self.prior, bool)
            or not math.isfinite(self.prior)
            or not 0 <= self.prior <= 1
        ):
            raise CompatibilityError("Candidate prior must be a finite ranking score in [0,1]")
        if any(type(r) is not int or r < 1 for r in self.ranks.values()):
            raise CompatibilityError("Ranker positions must be positive integers")
        if any(
            not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s)
            for s in self.scores.values()
        ):
            raise CompatibilityError("Recorded ranker scores must be finite; missing is not zero")


@dataclass(frozen=True)
class GoldLabels:
    files: tuple = ()
    functions: tuple = ()
    semantics: str = "curated_localization"
    exclusions: tuple = ()
    unavailable_reason: str = ""


@dataclass(frozen=True)
class Capabilities:
    languages: tuple
    granularities: tuple = ("file", "method")
    runtime_evidence: str = "External observed logs; never fabricated"
    candidates: str = "Imported, lexical/BM25, or external provider; FlexFL optional for Java"
    checkout: str = "Existing immutable checkout or explicit local worktree preparation"
    file_gold: str = "Explicit labels or filtered reference-change proxy"
    method_gold: str = "Explicit reliable labels; Defects4J labels when supplied"
    execution: str = "Implemented and fixture-tested; requires external data/model/toolchains"
