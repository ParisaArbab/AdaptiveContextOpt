"""Load real FlexFL candidate lists, no synthetic candidate generation."""
from __future__ import annotations
from pathlib import Path
from rave_fl.core import Candidate
from wp1.flexfl_data import merge_top20, read_traditional_top5


def load_candidates(flexfl_repo: Path, bug: str, method: str = "Ochiai",
                    agent4sr_top5: list[str] | None = None) -> list[Candidate]:
    """Ranked method identifiers from an existing FlexFL result file.

    Scores are ordinal rank priors, NOT source-model confidence or coverage.
    Missing method results fail loudly. Ground truth is never consulted here.
    """
    if method == "merged":
        if agent4sr_top5 is None:
            raise ValueError("Merged mode requires real Agent4SR top-5 predictions")
        names, _ = merge_top20(flexfl_repo, bug, agent4sr_top5)
    else:
        names = read_traditional_top5(flexfl_repo, method, bug)
    # De-duplicate repeated methods while preserving first occurrence.
    names = list(dict.fromkeys(names))
    return [
        Candidate(function_id=name, signature=name,
                  semantic_score=(len(names) - i) / max(1, len(names)))
        for i, name in enumerate(names)
    ]
