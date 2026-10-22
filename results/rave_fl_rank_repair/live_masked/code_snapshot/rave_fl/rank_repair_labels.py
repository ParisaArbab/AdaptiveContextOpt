"""Offline marginal action-value labels conditional on the current cheap state.

Counterfactual complete-code outcomes are used ONLY here, after policy runs.
Rows are correlated within bugs/states; this module never trains a predictor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from rave_fl.adaptive import fused_score
from rave_fl.code_sketch import state_features
from rave_fl.core import Action
from rave_fl.pilot_types import CandidateState, Observation
from wp1.evaluation import first_relevant_rank


def action_value_rows(corpus, truths, runs, bug, budget):
    fixed = corpus.references['fixed_batch']
    originals = {c.function_id: c.rank for c in corpus.candidates}
    base_status = {i: CandidateState(1, Observation(**s['observation'])) for i, s in fixed['states'].items()}
    seen = set()
    rows = []
    def metrics(status):
        scores = {c.function_id: fused_score(c, status[c.function_id]) for c in corpus.candidates}
        order = sorted(scores, key=lambda i: (-scores[i], originals[i]))
        rank = first_relevant_rank(order, truths)
        return {'MRR': 1 / rank if rank else 0, 'top1': int(bool(rank and rank <= 1)),
                'top5': int(bool(rank and rank <= 5))}, order, scores
    for run in runs:
        status = deepcopy(base_status)
        spent = fixed['actual_tokens']
        selected = []
        for next_call in [c for c in run['calls'] if c['level'] == 2] + [None]:
            key = (tuple(sorted(selected)), spent, budget)
            if key not in seen:
                seen.add(key)
                state_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
                base_metric, order, scores = metrics(status)
                ranks = {ident: n for n, ident in enumerate(order, 1)}
                for c in corpus.candidates:
                    ident = c.function_id
                    if status[ident].level == 2:
                        continue
                    estimate = corpus.backend().estimate([c], Action.FULL_VERIFY, 2)
                    features = state_features(c, status[ident], max(0, budget - spent) / max(1, budget), 1)
                    index = ranks[ident] - 1
                    features.update(current_rank=float(ranks[ident]), current_score=scores[ident],
                                    full_cost_estimate=float(estimate),
                                    gap_above=scores[order[index - 1]] - scores[ident] if index else 0,
                                    gap_below=scores[ident] - scores[order[index + 1]] if index + 1 < len(order) else 0,
                                    verified_fraction=len(selected) / len(status))
                    # No future answer/cost above this line enters the feature map.
                    full = corpus.observed_full(ident)
                    hybrid = deepcopy(status)
                    hybrid[ident] = CandidateState(2, full.observations[0])
                    after, _, _ = metrics(hybrid)
                    delta = {k: after[k] - base_metric[k] for k in after}
                    rows.append({'schema': 'rank_repair_action_value_v1', 'bug': bug, 'state_id': state_id,
                                 'candidate_id': ident, 'already_full_ids': sorted(selected),
                                 'features': features, 'admissible_at_state': estimate <= budget - spent,
                                 'targets': {**{'delta_' + k: v for k, v in delta.items()},
                                             'observed_full_tokens': full.total_tokens,
                                             'delta_MRR_per_1000_tokens': delta['MRR'] * 1000 / max(1, full.total_tokens)},
                                 'independent_sampling_unit': 'bug', 'gold_is_offline_only': True})
            if next_call is None:
                break
            ident = next_call['candidate_ids'][0]
            result = corpus.observed_full(ident)
            status[ident] = CandidateState(2, result.observations[0])
            selected.append(ident)
            spent += result.total_tokens
    return rows
