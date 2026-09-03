"""
flexfl_merge.py — FlexFL's Stage 1 -> Stage 2 candidate handoff.

This is the step the pipeline was missing. Agent4SR's output was being used
either alone (when the model produced a parseable ranking) or replaced
wholesale by a heuristic (when it did not) — never merged with traditional
fault localization, which is what the paper actually specifies.

Section 4.5 of FlexFL, reproduced exactly:

    bug report AND trigger test available
        top-5 SBIR, then top-5 Ochiai, then top-5 BoostN, then top-5 Agent4SR
    trigger test only
        top-15 Ochiai, then top-5 Agent4SR
    bug report only
        top-15 BoostN, then top-5 Agent4SR

then truncated to 20 (`rank.py`: `suspicious_methods[:20]`) and handed to
Agent4LR, whose re-ranked top-5 is the final answer.

Two details that are easy to get backwards, and that the paper is explicit
about:

  * Agent4SR goes LAST, not first. The paper's reasoning: methods Agent4SR
    finds are likely to be found by Agent4LR anyway, "so we do not need to
    emphasize them via high ranking". Putting Agent4SR first would spend
    the scarce top slots on the one source that needs them least.

  * Agent4LR does not add to the list. It re-ranks the merged candidates and
    its top-5 replaces them. The final answer is a permutation of the merged
    20, never a superset.

The original `combine.py` appends without deduplicating, so a method found
by three rankers occupies three slots. That is preserved: deduplicating
would change the candidate distribution Agent4LR sees, and with it the
result, which makes it a different experiment rather than a tidier one.

Two sources of the traditional rankings, chosen automatically:
  * Defects4J — the replication package's precomputed CSVs, read through
    flexfl_data.py (merged from the Defects4J branch).
  * SWE-bench — computed live by traditional_fl.py, because no such CSVs
    exist for these instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import traditional_fl
from traditional_fl import RankedList

CANDIDATE_CAP = 20          # rank.py: suspicious_methods[:20]
PER_SOURCE_TOPK = 5         # §4.5: top-5 from each source when all are present
SINGLE_SOURCE_TOPK = 15     # §4.5: top-15 when only one traditional source applies
ORDER = ("SBIR", "Ochiai", "BoostN")   # combine.py's order, Agent4SR appended last


@dataclass
class MergeResult:
    candidates: List[str] = field(default_factory=list)
    mode: str = ""                       # which §4.5 branch was taken
    sources: Dict[str, dict] = field(default_factory=dict)
    agent4sr_used: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)
    provisional: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_candidates": len(self.candidates),
            "candidates": self.candidates,
            "agent4sr_used": self.agent4sr_used,
            "sources": self.sources,
            "unavailable": self.unavailable,
            "provisional": self.provisional,
            "note": self.note,
        }


def merge(
    agent4sr_top5: Sequence[str],
    rankers: Dict[str, RankedList],
    has_bug_report: bool,
    has_trigger_test: bool,
) -> MergeResult:
    """Implements the §4.5 branch table. `rankers` is keyed by SBIR / Ochiai
    / BoostN; a missing or unavailable ranker is recorded, not silently
    skipped — a merge that ran without Ochiai is a materially different
    experiment from one where Ochiai returned nothing."""
    agent4sr = list(agent4sr_top5)[:PER_SOURCE_TOPK]

    def available(name: str) -> Optional[RankedList]:
        r = rankers.get(name)
        return r if (r and r.available and r.entries) else None

    ochiai, boostn, sbir = available("Ochiai"), available("BoostN"), available("SBIR")
    merged: List[str] = []
    used: List[str] = []

    if has_bug_report and has_trigger_test and (sbir or ochiai or boostn):
        mode = "report+test: 5 SBIR + 5 Ochiai + 5 BoostN + 5 Agent4SR"
        for name in ORDER:
            r = available(name)
            if r:
                merged.extend(r.top(PER_SOURCE_TOPK))
                used.append(name)
    elif has_trigger_test and ochiai:
        mode = "test only: 15 Ochiai + 5 Agent4SR"
        merged.extend(ochiai.top(SINGLE_SOURCE_TOPK))
        used.append("Ochiai")
    elif has_bug_report and boostn:
        mode = "report only: 15 BoostN + 5 Agent4SR"
        merged.extend(boostn.top(SINGLE_SOURCE_TOPK))
        used.append("BoostN")
    else:
        mode = "agent4sr only: no traditional ranker was available"

    merged.extend(agent4sr)          # last, per §4.5 — deliberately not first

    unavailable = [
        f"{name}: {rankers[name].reason or 'no entries'}"
        for name in ORDER
        if name in rankers and not available(name)
    ]
    missing = [name for name in ORDER if name not in rankers]
    unavailable.extend(f"{name}: not computed for this run" for name in missing)

    return MergeResult(
        candidates=merged[:CANDIDATE_CAP],
        mode=mode,
        sources={name: r.as_dict() for name, r in rankers.items()},
        agent4sr_used=agent4sr,
        unavailable=unavailable,
        provisional=any(r.provisional for r in rankers.values() if r.available),
        note=("no traditional fault localizer contributed; this is Agent4SR alone, "
              "which is NOT the paper's pipeline")
        if not used else "",
    )


def merge_for_swebench(
    agent4sr_top5: Sequence[str],
    problem_statement: str,
    structure_map: Dict[str, dict],
    repo_root: Path,
    coverage_json: Optional[dict] = None,
    failing_test_ids: Sequence[str] = (),
    coverage_error: str = "",
) -> MergeResult:
    rankers = traditional_fl.compute_all(
        problem_statement=problem_statement,
        structure_map=structure_map,
        repo_root=repo_root,
        coverage_json=coverage_json,
        failing_test_ids=failing_test_ids,
        coverage_error=coverage_error,
    )
    return merge(
        agent4sr_top5, rankers,
        has_bug_report=bool(problem_statement and problem_statement.strip()),
        has_trigger_test=bool(failing_test_ids),
    )


def merge_for_defects4j(
    agent4sr_top5: Sequence[str],
    flexfl_repo: Path,
    bug: str,
    has_bug_report: bool = True,
    has_trigger_test: bool = True,
) -> MergeResult:
    """Uses the replication package's precomputed CSVs, via the reader
    merged from the Defects4J branch."""
    import flexfl_data

    rankers: Dict[str, RankedList] = {}
    for name in ORDER:
        csv_name = {"SBIR": "SBIR", "Ochiai": "Ochiai", "BoostN": "BoostN"}[name]
        try:
            entries = flexfl_data.read_traditional_top5(flexfl_repo, csv_name, bug)
            rankers[name] = RankedList(name=name, entries=entries)
        except (FileNotFoundError, ValueError) as e:
            rankers[name] = RankedList.unavailable(name, str(e))
    return merge(agent4sr_top5, rankers, has_bug_report, has_trigger_test)
