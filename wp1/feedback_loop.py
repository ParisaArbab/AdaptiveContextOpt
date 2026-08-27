"""
feedback_loop.py — WP1 pipeline stage 3: BIDIRECTIONAL compression feedback.

The requirement is two-directional: "if something that was pruned was
needed, return it back; on the contrary if something was useless, filter it
— and make sure we don't get stuck in an infinite loop."

The previous version only did the first half. It could reveal lines the
compressor had cut, which means it could only ever ADD tokens; nothing in
the loop could ever remove content the compressor had kept but that turned
out to be noise. That made the loop a pure cost with respect to the
project's own headline metric. Both directions exist now:

    RESTORE  — `MISSING: L<a>-L<b> <reason>` re-inserts a specific range of
               the ORIGINAL text, in its original position.
    PRUNE    — `USELESS: C<a>-C<b> <reason>` deletes a specific range of the
               CURRENT (working) text.

Two different coordinate systems, deliberately: `L` numbers address the raw
capture (so a restore request can name something the agent cannot see), `C`
numbers address the numbered working text the agent is looking at right
now. Mixing them was the easiest way to get silently-wrong edits, so the
prefixes are required and a mismatch is rejected.

To make RESTORE requests answerable at all, the agent is shown a menu of
what was actually omitted (`_omitted_ranges`) rather than being asked to
guess line numbers of text it never saw.

## Termination — four independent guards, all necessary

1. Hard round cap (MAX_ROUNDS), as before.
2. No-progress stop: if a round applies zero edits, the loop exits
   immediately instead of burning the remaining rounds re-asking.
3. Idempotence ledger: a raw range already restored can't be restored
   again, and a range restored during this loop can't then be pruned. Those
   two rules are what make restore/prune oscillation impossible rather than
   merely unlikely — without them a model that disagrees with itself can
   ping-pong one region forever within any round budget.
4. Prune budget: at most PRUNE_BUDGET_PCT of the working text can be
   removed per round, so a single over-confident verdict can't collapse the
   evidence to nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MAX_ROUNDS = 2
PRUNE_BUDGET_PCT = 0.35

DIRECTIVE_RE = re.compile(
    r"^\s*(MISSING|USELESS)\s*:\s*([LC])(\d+)\s*(?:-\s*[LC]?(\d+))?\s*(.*)$",
    re.IGNORECASE,
)

FIDELITY_SYSTEM_PROMPT = """You are auditing a COMPRESSED tool output before a \
fault-localization agent reasons over it. Your job is to make the text as \
small as possible while keeping every piece of evidence that could identify \
the buggy code.

Reply with one or more directives, one per line, or exactly OK if the text \
is already right.
* MISSING: L<start>-L<end> <reason>  -- restore an omitted region. <start>/<end> \
MUST be one of the omitted ranges listed below; requests outside that list are \
rejected.
* USELESS: C<start>-C<end> <reason>  -- delete lines of the text shown below, \
addressed by their C-numbers. Use this for boilerplate, environment banners, \
repeated separators, and passing-test chatter.
Restore evidence (stack frames, assertion values, error types, file paths) and \
prune everything that could not distinguish one candidate method from another."""


@dataclass
class FeedbackTurn:
    round: int
    verdict: str
    restored: List[str] = field(default_factory=list)   # e.g. ["L40-L55"]
    pruned: List[str] = field(default_factory=list)     # e.g. ["C3-C9"]
    rejected: List[str] = field(default_factory=list)   # directives refused, with why
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass
class FeedbackResult:
    final_text: str
    rounds_used: int
    turns: List[FeedbackTurn] = field(default_factory=list)
    exhausted: bool = False          # agent still wanted changes when rounds ran out
    stop_reason: str = ""
    chars_restored: int = 0
    chars_pruned: int = 0

    @property
    def n_restores(self) -> int:
        return sum(len(t.restored) for t in self.turns)

    @property
    def n_prunes(self) -> int:
        return sum(len(t.pruned) for t in self.turns)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _omitted_ranges(raw_text: str, compressed_text: str) -> List[Tuple[int, int]]:
    """Contiguous 1-based raw line ranges absent from the compressed text.

    Membership is by line content rather than a real diff: the compressor
    preserves original order and inserts its own marker lines, so a
    content-set check identifies dropped regions correctly and is far
    cheaper than an alignment over multi-thousand-line pytest output."""
    kept = set(compressed_text.splitlines())
    ranges: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, line in enumerate(raw_text.splitlines(), 1):
        if line not in kept:
            start = i if start is None else start
        elif start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(raw_text.splitlines())))
    return ranges


def _clip_to_omitted(req: Tuple[int, int],
                     omitted: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """A restore is granted only for the part of the request that overlaps a
    genuinely-omitted region. Asking for a range that was never removed is
    a no-op rather than a free re-read of text already present."""
    best: Optional[Tuple[int, int]] = None
    for lo, hi in omitted:
        overlap = (max(req[0], lo), min(req[1], hi))
        if overlap[0] <= overlap[1]:
            if best is None or (overlap[1] - overlap[0]) > (best[1] - best[0]):
                best = overlap
    return best


def _overlaps_any(rng: Tuple[int, int], others: Sequence[Tuple[int, int]]) -> bool:
    return any(rng[0] <= hi and lo <= rng[1] for lo, hi in others)


def number_lines(text: str) -> str:
    """C-numbered rendering handed to the agent, so USELESS ranges address
    something it can actually see."""
    return "\n".join(f"C{i}| {line}" for i, line in enumerate(text.splitlines(), 1))


def build_audit_payload(current_text: str, omitted: Sequence[Tuple[int, int]],
                        round_num: int) -> str:
    menu = ", ".join(f"L{lo}-L{hi}" for lo, hi in omitted[:40]) or "(nothing omitted)"
    if len(omitted) > 40:
        menu += f", ... ({len(omitted) - 40} more omitted ranges)"
    return (
        f"Round {round_num} of {MAX_ROUNDS}.\n"
        f"Omitted ranges available to restore: {menu}\n\n"
        f"CURRENT TEXT:\n{number_lines(current_text)}"
    )


def parse_directives(verdict: str) -> List[Tuple[str, str, int, int, str]]:
    """-> [(kind, coord_prefix, start, end, reason)]"""
    out = []
    for line in verdict.splitlines():
        m = DIRECTIVE_RE.match(line)
        if not m:
            continue
        kind, prefix, start_s, end_s, reason = m.groups()
        start = int(start_s)
        end = int(end_s) if end_s else start
        if end < start:
            start, end = end, start
        out.append((kind.upper(), prefix.upper(), start, end, reason.strip()))
    return out


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------

def _restore(current_text: str, raw_text: str,
             rng: Tuple[int, int]) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Re-inserts raw lines rng in their original position where possible.

    Position matters: dumping restored lines at the end would break the
    stack-frame ordering that FlexFL's Stage 1 reads causally. We locate the
    nearest raw line before the range that survives in the current text and
    splice after it, falling back to appending only when no anchor exists."""
    raw_lines = raw_text.splitlines()
    lo, hi = rng
    block = raw_lines[lo - 1 : hi]
    if not block:
        return current_text, None

    cur_lines = current_text.splitlines()
    anchor_idx = None
    for raw_i in range(lo - 2, -1, -1):
        anchor = raw_lines[raw_i]
        if anchor in cur_lines:
            anchor_idx = len(cur_lines) - 1 - cur_lines[::-1].index(anchor)
            break

    marker = f"[feedback: restored L{lo}-L{hi}]"
    if anchor_idx is None:
        insert_at = len(cur_lines)          # 0-based index of the marker line
        merged = cur_lines + [marker] + block
    else:
        insert_at = anchor_idx + 1
        merged = (cur_lines[: anchor_idx + 1] + [marker] + block
                  + cur_lines[anchor_idx + 1 :])
    # 1-based C-coordinates the restored block now occupies, so the
    # oscillation guard protects exactly those lines and nothing else.
    span = (insert_at + 1, insert_at + 1 + len(block))
    return "\n".join(merged), span


def _prune(current_text: str, rng: Tuple[int, int]) -> str:
    lo, hi = rng
    lines = current_text.splitlines()
    keep = [l for i, l in enumerate(lines, 1) if not (lo <= i <= hi)]
    return "\n".join(keep)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_feedback_loop(
    raw_text: str,
    compressed_text: str,
    agent_verify_fn: Callable[[str, int], str],
) -> FeedbackResult:
    """agent_verify_fn(audit_payload, round_num) -> verdict string.

    Provider-agnostic by design: the caller supplies whatever produces the
    verdict (a real LLM call in agent_localizer.localize_with_llm, or the
    deterministic stand-in below for the key-free backend)."""
    current = compressed_text
    turns: List[FeedbackTurn] = []
    restored_ranges: List[Tuple[int, int]] = []   # raw coords, granted this session
    protected: List[Tuple[int, int]] = []          # current coords of restored blocks
    chars_restored = 0
    chars_pruned = 0
    stop_reason = "round_cap"
    exhausted = False

    for round_num in range(1, MAX_ROUNDS + 1):
        omitted = _omitted_ranges(raw_text, current)
        payload = build_audit_payload(current, omitted, round_num)
        verdict = agent_verify_fn(payload, round_num)

        turn = FeedbackTurn(round=round_num, verdict=verdict.strip()[:500],
                            tokens_before=len(current))

        directives = parse_directives(verdict)
        if not directives:
            turns.append(turn)
            turn.tokens_after = len(current)
            stop_reason = "agent_satisfied"
            break

        n_lines_before = len(current.splitlines())
        prune_budget_lines = max(1, int(n_lines_before * PRUNE_BUDGET_PCT))
        pruned_this_round = 0
        changed = False

        # Prunes are applied first, high line number to low, so that earlier
        # deletions never shift the C-coordinates of a later one. Restores
        # then run against the already-pruned text.
        prunes = sorted([d for d in directives if d[0] == "USELESS"],
                        key=lambda d: d[2], reverse=True)
        restores = [d for d in directives if d[0] == "MISSING"]

        for _kind, prefix, start, end, reason in prunes:
            if prefix != "C":
                turn.rejected.append(f"USELESS L{start}-L{end}: prunes must use C-coordinates")
                continue
            if _overlaps_any((start, end), protected):
                turn.rejected.append(
                    f"USELESS C{start}-C{end}: overlaps a range restored this session "
                    f"(oscillation guard)")
                continue
            span = end - start + 1
            if pruned_this_round + span > prune_budget_lines:
                turn.rejected.append(
                    f"USELESS C{start}-C{end}: exceeds the {int(PRUNE_BUDGET_PCT*100)}% "
                    f"per-round prune budget")
                continue
            before = current
            current = _prune(current, (start, end))
            if current == before:
                turn.rejected.append(f"USELESS C{start}-C{end}: out of bounds")
                continue
            pruned_this_round += span
            chars_pruned += len(before) - len(current)
            turn.pruned.append(f"C{start}-C{end}")
            changed = True

        for _kind, prefix, start, end, reason in restores:
            if prefix != "L":
                turn.rejected.append(f"MISSING C{start}-C{end}: restores must use L-coordinates")
                continue
            granted = _clip_to_omitted((start, end), _omitted_ranges(raw_text, current))
            if granted is None:
                turn.rejected.append(
                    f"MISSING L{start}-L{end}: that range was never omitted, nothing to restore")
                continue
            if _overlaps_any(granted, restored_ranges):
                turn.rejected.append(f"MISSING L{start}-L{end}: already restored (replay guard)")
                continue
            before = current
            current, span = _restore(current, raw_text, granted)
            if current == before or span is None:
                turn.rejected.append(f"MISSING L{start}-L{end}: nothing to insert")
                continue
            restored_ranges.append(granted)
            chars_restored += len(current) - len(before)
            turn.restored.append(f"L{granted[0]}-L{granted[1]}")
            protected.append(span)
            changed = True

        turn.tokens_after = len(current)
        turns.append(turn)

        if not changed:
            stop_reason = "no_progress"
            exhausted = bool(turn.rejected)
            break
    else:
        # Round cap hit with edits still landing — record whether the agent
        # would have kept going, without applying anything further.
        final_payload = build_audit_payload(current, _omitted_ranges(raw_text, current),
                                            MAX_ROUNDS + 1)
        exhausted = bool(parse_directives(agent_verify_fn(final_payload, MAX_ROUNDS + 1)))

    return FeedbackResult(
        final_text=current,
        rounds_used=len(turns),
        turns=turns,
        exhausted=exhausted,
        stop_reason=stop_reason,
        chars_restored=chars_restored,
        chars_pruned=chars_pruned,
    )


# ---------------------------------------------------------------------------
# Key-free stand-in
# ---------------------------------------------------------------------------

_EVIDENCE_RE = re.compile(
    r'File "|Traceback|AssertionError|Error:|assert |E\s+\w+Error|expected:|actual:',
)
_NOISE_RE = re.compile(
    r"^(C\d+\|\s*)?(platform \w+|rootdir|plugins:|cachedir|collecting|"
    r"Downloading|Requirement already satisfied|INFO |DEBUG |=+$|-+$|\s*$)",
)


def heuristic_verify_fn(raw_text: str):
    """Deterministic verdict generator so the feedback-loop ablation arm is a
    real variable on the key-free `heuristic` backend instead of a no-op.

    It does exactly what the LLM is asked to do, mechanically: restore the
    largest omitted region that contains failure evidence, and prune the
    longest run of visible boilerplate. Tagged heuristic everywhere
    downstream — it validates the loop's mechanics and its token effect, it
    is not a substitute for the model's judgment."""
    raw_lines = raw_text.splitlines()

    def verify(payload: str, round_num: int) -> str:
        directives: List[str] = []

        menu_match = re.search(r"Omitted ranges available to restore: (.*)", payload)
        if menu_match:
            best: Optional[Tuple[int, int, int]] = None  # (score, lo, hi)
            for lo_s, hi_s in re.findall(r"L(\d+)-L(\d+)", menu_match.group(1)):
                lo, hi = int(lo_s), int(hi_s)
                block = raw_lines[lo - 1 : hi]
                score = sum(1 for l in block if _EVIDENCE_RE.search(l))
                if score and (best is None or score > best[0]):
                    best = (score, lo, hi)
            if best:
                _score, lo, hi = best
                hi = min(hi, lo + 40)  # bounded restore: a region, not the whole capture
                directives.append(f"MISSING: L{lo}-L{hi} omitted region contains failure evidence")

        run_start: Optional[int] = None
        best_run: Optional[Tuple[int, int]] = None
        for m in re.finditer(r"^C(\d+)\|(.*)$", payload, re.MULTILINE):
            idx, content = int(m.group(1)), m.group(2)
            if _NOISE_RE.match(content.strip()) or not content.strip():
                run_start = idx if run_start is None else run_start
                if best_run is None or (idx - run_start) > (best_run[1] - best_run[0]):
                    best_run = (run_start, idx)
            else:
                run_start = None
        if best_run and best_run[1] > best_run[0]:
            # Stay inside the loop's own per-round prune budget rather than
            # proposing an over-large deletion that will just be rejected.
            n_lines = len(re.findall(r"^C\d+\|", payload, re.MULTILINE))
            budget = max(1, int(n_lines * PRUNE_BUDGET_PCT))
            lo, hi = best_run
            hi = min(hi, lo + budget - 1)
            if hi > lo:
                directives.append(
                    f"USELESS: C{lo}-C{hi} environment/boilerplate lines "
                    f"with no discriminating evidence")

        return "\n".join(directives) if directives else "OK"

    return verify
