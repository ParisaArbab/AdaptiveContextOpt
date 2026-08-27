"""
ablation.py — WP1 step 3: the ablation arm matrix.

Three independently-switchable pipeline elements:

  graphify  — graph-based structural understanding: the pre-search
              structural briefing that primes FlexFL Stage 1, and the
              post-Stage-2 GraphLocator causal expansion. Note this ablates
              the GRAPH, not the structural index: FlexFL's own function
              calls (find_method, get_code_snippet_of_method, ...) need an
              index of the repository in every arm, and the real FlexFL
              builds one itself. Removing that too would ablate FlexFL
              rather than our contribution, and the comparison against
              "pure FlexFL" would stop being a comparison.
  leanctx   — the smart compressor on the captured tool output.
  feedback  — the bidirectional restore/prune loop over the compressed text.
              Forced off when leanctx is off: there is no compression to
              give feedback on, so an arm with feedback-but-no-compressor
              would be measuring nothing.

The full 2^3 factorial is generated so each element's main effect AND its
interactions are identifiable — "without graphify" and "without lean-ctx"
separately don't tell you whether the two overlap, and overlap is exactly
what you'd expect between a structural prior and a content compressor.
DEFAULT_ARMS is the five-arm subset from the project brief; `--arms all`
runs the whole factorial.

`pure_flexfl` (all three off) is the control: FlexFL + raw uncompressed
trigger-test output, i.e. the paper's own method with none of this
project's additions.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence

ELEMENTS = ("graphify", "leanctx", "feedback")


@dataclass(frozen=True)
class Arm:
    name: str
    use_graphify: bool
    use_leanctx: bool
    use_feedback: bool
    description: str = ""

    @property
    def enabled(self) -> tuple:
        return (self.use_graphify, self.use_leanctx, self.use_feedback)

    def as_dict(self) -> dict:
        return {
            "arm": self.name,
            "use_graphify": self.use_graphify,
            "use_leanctx": self.use_leanctx,
            "use_feedback": self.use_feedback,
        }


def _name_for(g: bool, l: bool, f: bool) -> str:
    """Named by what's MISSING relative to the full pipeline, since that's how
    the results get read. Turning lean-ctx off forces feedback off too (see
    the module docstring), so `graphify on, leanctx off` is the arm a reader
    means by "without lean-ctx" — it is named that, not `graphify_only`."""
    on = [n for n, v in zip(ELEMENTS, (g, l, f)) if v]
    off = [n for n, v in zip(ELEMENTS, (g, l, f)) if not v]
    if not on:
        return "pure_flexfl"
    if not off:
        return "full"
    if g and not l:
        return "no_leanctx"          # feedback is off by consequence, not by choice
    if len(off) == 1:
        return f"no_{off[0]}"
    return "no_" + "_".join(off)


def _build() -> Dict[str, Arm]:
    arms: Dict[str, Arm] = {}
    for g, l, f in product((True, False), repeat=3):
        if f and not l:
            continue  # feedback without a compressor has nothing to act on
        name = _name_for(g, l, f)
        arms[name] = Arm(name, g, l, f, description=_describe(g, l, f))
    return arms


def _describe(g: bool, l: bool, f: bool) -> str:
    if not any((g, l, f)):
        return "control: FlexFL alone on raw trigger-test output, no token optimization"
    parts = []
    parts.append("graph structural understanding + GraphLocator expansion" if g
                 else "no graph (FlexFL's own index only)")
    parts.append("lean-ctx compression" if l else "uncompressed tool output")
    parts.append("bidirectional feedback loop" if f else "no feedback loop")
    return "; ".join(parts)


ARMS: Dict[str, Arm] = _build()

# The brief's five: whole pipeline, minus each element in turn, and the
# no-optimization control.
DEFAULT_ARMS: List[str] = ["full", "no_graphify", "no_leanctx", "no_feedback", "pure_flexfl"]
ALL_ARMS: List[str] = list(ARMS)
CONTROL_ARM = "pure_flexfl"

# Convenience spellings so a name a reader might reach for still resolves.
ALIASES: Dict[str, str] = {
    "graphify_only": "no_leanctx",
    "leanctx_only": "no_graphify_feedback",
    "control": CONTROL_ARM,
    "raw": CONTROL_ARM,
}


def resolve_arms(spec: Sequence[str]) -> List[Arm]:
    """Accepts ['all'], ['default'], or explicit arm names."""
    if not spec or list(spec) == ["default"]:
        names = DEFAULT_ARMS
    elif list(spec) == ["all"]:
        names = ALL_ARMS
    else:
        names = []
        for item in spec:
            for part in item.split(","):
                part = part.strip()
                if not part:
                    continue
                part = ALIASES.get(part, part)
                if part not in ARMS:
                    raise ValueError(
                        f"unknown arm {part!r}; known: {', '.join(sorted(ARMS))}")
                if part not in names:
                    names.append(part)
    return [ARMS[n] for n in names]


if __name__ == "__main__":
    width = max(len(n) for n in ARMS)
    for name, arm in ARMS.items():
        flags = " ".join(f"{e}={'on' if v else 'off'}"
                         for e, v in zip(ELEMENTS, arm.enabled))
        mark = "*" if name in DEFAULT_ARMS else " "
        print(f"{mark} {name:<{width}}  {flags}")
    print("\n* = in DEFAULT_ARMS")
