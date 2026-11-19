"""Pre-registered exploratory Code Card -> Full selection rules.

All features are available before the Full response: Code Card suspicion/uncertainty,
existing prior, candidate file rank, and literal failures in the observed log.
No gold labels, full-response scores, or future observations enter selection.
"""
from __future__ import annotations

import re

POLICIES = ('card_rank_full', 'card_failure_full', 'card_stop_full', 'card_egav')
STOP_POLICIES = ('card_stop_full', 'card_egav')


def observed_failure_anchor(candidate: dict, evidence: str) -> float:
    """Strong literal match against the *observed* failure text, not patch/gold.

    An explicit filename or method call is required: generic language overlap
    is deliberately ignored to limit noise from words like 'time' or 'test'.
    """
    if not evidence:
        return 0.0
    file_basename = str(candidate.get('file', '')).replace('\\', '/').rsplit('/', 1)[-1]
    if file_basename and re.search(r'(?<![\w.])' + re.escape(file_basename) + r'(?![\w.])', evidence, re.I):
        return 1.0
    cid = str(candidate.get('id', ''))
    method_name = cid.split('::')[-1].split('(')[0].rsplit('.', 1)[-1]
    if (len(method_name) >= 4 and method_name.lower() not in
            {'test', 'main', 'execute', 'run', 'init', 'get', 'set', 'load', 'assert'} and
            re.search(r'(?<!\w)' + re.escape(method_name) + r'\s*\(', evidence)):
        return 0.75
    return 0.0


def file_rank_by_candidate(candidates: list[dict], state: dict) -> dict:
    """Ranks derived only from currently observed card/full scores."""
    file_scores = {}
    for c in candidates:
        f = c.get('file', c['id'])
        file_scores[f] = max(file_scores.get(f, float('-inf')), float(state[c['id']]))
    files = sorted(file_scores, key=lambda f: (-file_scores[f], f))
    ranks = {f: i + 1 for i, f in enumerate(files)}
    return {c['id']: ranks[c.get('file', c['id'])] for c in candidates}


def boundary_weight(rank: int) -> float:
    """At the Top-5 boundary weight is 1; distant ranks are less important."""
    return max(0.0, 1.0 - abs(rank - 5) / 5.0)


def ranked_full_targets(candidates, state, seen, failure_text, policy, risk_fn):
    """Order Full targets using observed card scores, not future Full outcomes."""
    if policy not in POLICIES:
        raise ValueError('Unknown exploratory EGAV policy: ' + policy)
    ranks = file_rank_by_candidate(candidates, state)

    def priority(c):
        risk = risk_fn(c, seen, graph=True)
        boundary = boundary_weight(ranks[c['id']])
        anchor = observed_failure_anchor(c, failure_text)
        if policy == 'card_rank_full':
            return 0.60 * risk + 0.40 * boundary
        if policy == 'card_failure_full':
            return 0.55 * risk + 0.45 * anchor
        if policy == 'card_stop_full':
            return risk
        return 0.45 * risk + 0.25 * boundary + 0.30 * anchor

    return sorted(candidates, key=lambda c: (-priority(c), c['id']))


def evidence_sufficient(candidates, state, seen) -> bool:
    """Conservative optional stopping heuristic, not calibrated confidence.

    Applies after the card screening stage and again after each verified method.
    A large lead, high score, low uncertainty, and no missing context are needed.
    """
    scores = {}
    representatives = {}
    for c in candidates:
        f = c.get('file', c['id'])
        s = float(state[c['id']])
        if f not in scores or s > scores[f]:
            scores[f] = s
            representatives[f] = c['id']
    ordered = sorted(scores, key=lambda f: (-scores[f], f))
    if len(ordered) < 2:
        return False
    top, runner = ordered[:2]
    top_observed = seen.get(representatives[top])
    if not top_observed:
        return False
    return (scores[top] >= 0.80 and scores[top] - scores[runner] >= 0.22
            and float(top_observed['uncertainty']) <= 0.25
            and not bool(top_observed.get('missing_dependency')))