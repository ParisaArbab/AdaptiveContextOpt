"""Offline BM25-384 -> 24-candidate filtering experiments, zero Qwen calls.

Ablation data comes from the already-used ten EGAV development bugs. The
selections NEVER use gold, patch, saved model responses, or source after fix.
Gold is read only after every pool has been selected. All output is file-level
retrieval recall, not final LLM localization accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research_method_agent'))
from rave_fl.eval.adapters import load_instances
from rave_fl.eval.candidates import provide
from rave_fl.eval.runner import validate_checkout
from egav_policies import observed_failure_anchor

CONFIGS = ROOT / 'data/graft_100_cohort_20261115/batches'
OUTPUT = ROOT / 'results/egav_validation_10bugs'
POOL = 384
CAP = 24
STRATEGIES = ('bm25_top24', 'unique_file_first', 'two_per_file',
              'failure_anchor_first', 'failure_anchor_file_balanced')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def norm(path):
    return str(path).replace('\\', '/').lstrip('./')


def candidate_file(candidate):
    return norm(candidate.path)


def order_with_quota(candidates, limit, max_per_file):
    """Gold-blind file diversity; candidates stay in BM25 priority order."""
    selected = []
    counts = {}
    used = set()
    for i, c in enumerate(candidates):
        f = candidate_file(c)
        if counts.get(f, 0) < max_per_file:
            selected.append(i)
            used.add(i)
            counts[f] = counts.get(f, 0) + 1
            if len(selected) >= limit:
                return selected
    for i in range(len(candidates)):
        if i not in used:
            selected.append(i)
            if len(selected) >= limit:
                break
    return selected


def select_indices(candidates, failure, strategy, limit=CAP):
    """Only current BM25 ranking, source path, method name and failure text.

    This is the online selection policy. No parameter accepts offline gold.
    """
    if strategy not in STRATEGIES:
        raise ValueError('Unknown retrieval strategy')
    if limit < 1:
        raise ValueError('Limit must be positive')
    if strategy == 'bm25_top24':
        return list(range(min(limit, len(candidates))))
    if strategy == 'unique_file_first':
        return order_with_quota(candidates, limit, 1)
    if strategy == 'two_per_file':
        return order_with_quota(candidates, limit, 2)
    def anchored_score(c):
        return observed_failure_anchor({'file': candidate_file(c), 'id': c.entity_id}, failure)
    ordered = sorted(range(len(candidates)), key=lambda i: (-anchored_score(candidates[i]), i))
    if strategy == 'failure_anchor_first':
        return ordered[:limit]
    # Anchor bonus first, but allocate one slot per distinct file before filling.
    local = order_with_quota([candidates[i] for i in ordered], limit, 1)
    return [ordered[j] for j in local]


def retrieve_batch(batch):
    path = CONFIGS / ('batch_' + batch) / 'experiment/config.json'
    cfg = load(path)['benchmark']
    if cfg['candidate_provider'] != 'bm25' or cfg['candidate_count'] != 24:
        raise RuntimeError('Wrong frozen BM25 settings for batch ' + batch)
    opts = dict(cfg)
    opts['candidate_count'] = POOL
    opts['allow_downloads'] = False
    opts['materialize_checkouts'] = False
    instances, _ = load_instances(opts)
    expected_ids = set(cfg['bugs'])
    if len(expected_ids) != 5:
        raise RuntimeError('Expected exactly five pre-registered bugs')
    by_id = {i.instance_id: i for i in instances if i.instance_id in expected_ids}
    if set(by_id) != expected_ids:
        raise RuntimeError('Missing original buggy Defects4J checkout')
    original_path = OUTPUT / ('batch_' + batch)
    original = {b['id']: b for b in load(original_path / 'manifest.json')['bugs']}
    if set(original) != expected_ids:
        raise RuntimeError('Frozen 24-method experiment does not match the source cohort')

    # IMPORTANT: fully select all methods for all strategies before reading gold.
    staged = {}
    for bugid in sorted(by_id):
        inst = validate_checkout(by_id[bugid])
        if inst.evidence.kind != 'observed_failure':
            raise RuntimeError('Must use observed failure evidence, not gold')
        candidates, provenance = provide(inst, opts)
        prefix = [c.entity_id for c in candidates[:len(original[bugid]['candidates'])]]
        saved = [c['id'] for c in original[bugid]['candidates']]
        if prefix != saved:
            raise RuntimeError('BM25 retrieval prefix changed: ' + bugid)
        arms = {}
        for arm in STRATEGIES:
            selected_idx = select_indices(candidates, inst.evidence.text, arm)
            selected = [candidates[i] for i in selected_idx]
            if len({c.entity_id for c in selected}) != len(selected):
                raise RuntimeError('Repeated candidate method in ' + arm)
            arms[arm] = {
                'method_indices_bm25': [i + 1 for i in selected_idx],
                'method_ids': [c.entity_id for c in selected],
                'files': [candidate_file(c) for c in selected],
                'distinct_files': len(set(candidate_file(c) for c in selected)),
            }
        staged[bugid] = {
            'indexed_methods': provenance.get('indexed_entities'),
            'retrieved_methods': len(candidates),
            'arms': arms,
            'all_bm25_files': [candidate_file(c) for c in candidates],
        }

    labels = load(original_path / 'gold_offline.json')['bugs']
    for bid, info in staged.items():
        gold = {norm(x) for x in labels[bid]['files']}
        if not gold:
            raise RuntimeError('Missing file gold for bug ' + bid)
        bm25 = info.pop('all_bm25_files')
        rank = next((i for i, path in enumerate(bm25, 1) if path in gold), None)
        info['gold_file_first_method_rank'] = rank
        info['oracle_pool_384'] = rank is not None
        info['gold_files'] = sorted(gold)
        for values in info['arms'].values():
            values['file_hit'] = bool(set(values['files']) & gold)
            values['first_selected_bm25_rank_of_gold'] = next(
                (idx for idx, f in zip(values['method_indices_bm25'], values['files'])
                 if f in gold), None)
            del values['method_ids']
            del values['files']
    return staged


def main():
    data = {}
    for batch in ('01', '02'):
        data.update(retrieve_batch(batch))
    if len(data) != 10:
        raise RuntimeError('Expected exactly ten independent development bugs')
    summary = {
        arm: {
            'hits': sum(row['arms'][arm]['file_hit'] for row in data.values()),
            'total': len(data),
            'mean_distinct_files': sum(row['arms'][arm]['distinct_files'] for row in data.values()) / len(data),
            'missed_bugs': [bid for bid in sorted(data) if not data[bid]['arms'][arm]['file_hit']],
        }
        for arm in STRATEGIES
    }
    payload = {
        'schema': 'bm25-384-to-24-filter-audit-v1',
        'source': '10 already-explored EGAV development bugs',
        'no_model_calls': True,
        'candidate_filter_uses_gold': False,
        'deep_bm25_pool': POOL,
        'selected_methods_per_bug': CAP,
        'summary': summary,
        'rows': data,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / 'high_recall_filter_audit.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('NO-INFERENCE BM25 FILTER: 384 -> 24 METHODS')
    print(f"{'Filter':30} {'Gold hits':>10} {'Mean files':>11} Missing bugs")
    for arm, result in summary.items():
        print(f"{arm:30} {str(result['hits']) + '/10':>10} "
              f"{result['mean_distinct_files']:>11.1f} {', '.join(result['missed_bugs'])}")
    print('\nGold-file first BM25 positions (method rank):')
    for bugid, row in sorted(data.items()):
        print(f"{bugid:14} {row['gold_file_first_method_rank']}")
    print('\nThis only measures whether a correct file is in the selected pool, not LLM ranking.')
    print('All selection policies used BM25 and observed failure evidence, never gold.')
    print('Saved:', path)


if __name__ == '__main__':
    main()