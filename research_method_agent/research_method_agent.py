"""Evidence-selection research agent for *paired* fault-localization pilots.

No gold labels are passed to collection or any online policy. The agent does not
claim to reproduce LLMLingua-2, GistRank, or the OECP exact-verifier theorem.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import time
import urllib.request
from pathlib import Path

ARMS = (
    'l1', 'adaptive_k', 'conformal_k', 'dependency_repair', 'risk_full', 'esr_proxy',
    'cascade', 'code_cards', 'code_cards_risk_full', 'combined', 'llmlingua2',
    'card_rank_full', 'card_failure_full', 'card_stop_full', 'card_egav',
)
OBS_KINDS = ('l1', 'full', 'dependency', 'card', 'llmlingua2')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8', 'surrogatepass')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    # Escape unpaired surrogates returned by some model JSON strings. The exact
    # Python strings can be recovered on load, without writing invalid UTF-8.
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def validate(manifest):
    if any(term in manifest for term in ('gold', 'labels', 'patch', 'test_patch', 'fix_patch')):
        raise ValueError('Offline labels and patches must not enter the online manifest')
    if manifest.get('schema') != 'research-method-agent-v1':
        raise ValueError('Expected research-method-agent-v1 manifest')
    ids = set()
    for bug in manifest.get('bugs', []):
        if any(term in bug for term in ('gold', 'labels', 'patch', 'test_patch', 'fix_patch')):
            raise ValueError('Gold/patch leaked into online bug: ' + bug['id'])
        bid = bug['id']
        if bid in ids:
            raise ValueError('Duplicate bug id: ' + bid)
        ids.add(bid)
        if not bug.get('evidence') or not bug.get('candidates'):
            raise ValueError('Every bug requires observed evidence and candidates: ' + bid)
        cids = [c['id'] for c in bug['candidates']]
        if len(set(cids)) != len(cids):
            raise ValueError('Duplicate candidate id: ' + bid)
        if len(cids) < 2:
            raise ValueError('Need at least two candidate methods')
        for c in bug['candidates']:
            if any(term in c for term in ('gold', 'labels', 'is_faulty', 'fix_patch')):
                raise ValueError('Candidate contains forbidden offline labels')
            if not (0 <= float(c['prior']) <= 1) or not math.isfinite(float(c['prior'])):
                raise ValueError('Invalid prior')
            if not c.get('l1') or not c.get('full'):
                raise ValueError('Need real L1 and full text for ' + c['id'])
            if any(dep not in cids or dep == c['id'] for dep in c.get('depends_on', [])):
                raise ValueError('Invalid dependency edge for ' + c['id'])
    if not ids:
        raise ValueError('No bugs in manifest')


def sorted_candidates(bug):
    return sorted(bug['candidates'], key=lambda c: (-float(c['prior']), c['id']))


def adaptive_k(candidates, buffer=1, max_fraction=0.9):
    """Largest consecutive prior-score gap, plus optional buffer. Not an accuracy bound."""
    scores = [float(c['prior']) for c in candidates]
    if len(scores) < 2:
        return len(scores)
    end = max(1, min(len(scores) - 1, int(math.ceil(len(scores) * max_fraction))))
    gap_index = max(range(end), key=lambda i: (scores[i] - scores[i + 1], -i))
    return max(1, min(len(scores), gap_index + 1 + buffer))


def code_card(source, limit=650):
    """Deterministic code card, NOT learned gist tokens. Protects exact source lines."""
    lines = source.splitlines()
    important = [line for line in lines if re.search(
        r'\b(def|class|public|private|return|raise|throw|if|elif|else|assert|for|while|call|import)\b|[=!<>]=', line
    )]
    return '\n'.join((important or lines)[:24])[:limit]


def observation_key(bid, cid, kind):
    return json.dumps([bid, cid, kind], separators=(',', ':'))


def obs_index(data):
    index = {}
    for item in data.get('observations', []):
        key = observation_key(item['bug'], item['candidate'], item['kind'])
        if key in index:
            raise ValueError('Duplicate observation: ' + key)
        if item.get('status') == 'ok':
            for field in ('score', 'uncertainty'):
                v = item[field]
                if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
                    raise ValueError('Invalid ' + field)
            for field in ('input_tokens', 'output_tokens'):
                if type(item.get(field)) is not int or item[field] < 0:
                    raise ValueError('Missing or invalid provider usage')
        index[key] = item
    return index


def ollama_call(url, model, evidence, candidate, text, kind, max_output=384):
    """One paid model call. Return even malformed replies with exact provider usage.

    Only a validated response can become a fault-localization observation.
    Every provider response is independently logged by collect() as an attempt.
    """
    if len(evidence) > 12000 or len(text) > 24000:
        raise ValueError('Evidence exceeds declared prompt cap; refusing silent truncation')
    prompt = (
        'You are evaluating one candidate code location for a software failure. '
        'Use only the observed failure and code supplied. Do not invent source facts. '
        'Reply with ONLY the requested JSON fields. Use numbers between 0 and 1 for '
        'suspicion and uncertainty, boolean missing_dependency, and an explanation '
        'of 20 words or less.\n'
        'Failure evidence:\n' + evidence + '\n'
        'Candidate ID: ' + candidate + '\n'
        'Evidence form: ' + kind + '\nCode evidence:\n' + text
    )
    schema = {
        'type': 'object',
        'properties': {
            'suspicion': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'uncertainty': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'missing_dependency': {'type': 'boolean'},
            'explanation': {'type': 'string', 'maxLength': 160},
        },
        'required': ['suspicion', 'uncertainty', 'missing_dependency', 'explanation'],
        'additionalProperties': False,
    }
    data = {'model': model, 'stream': False, 'format': schema,
            'messages': [{'role': 'user', 'content': prompt}],
            'options': {'temperature': 0, 'seed': 42, 'num_ctx': 16384, 'num_predict': max_output}}
    payload = json.dumps(data).encode()
    request = urllib.request.Request(url.rstrip('/') + '/api/chat', data=payload,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=360) as response:
        raw = json.loads(response.read())
    elapsed = time.monotonic() - started
    if (type(raw.get('prompt_eval_count')) is not int or
            type(raw.get('eval_count')) is not int or
            raw['prompt_eval_count'] < 0 or raw['eval_count'] < 0):
        raise RuntimeError('Provider did not return exact nonnegative input/output token counts')
    response_text = raw.get('message', {}).get('content', '')
    base = {'input_tokens': raw['prompt_eval_count'], 'output_tokens': raw['eval_count'],
            'duration_seconds': elapsed, 'model': model, 'prompt_sha256': sha(data),
            'prompt': prompt, 'raw_response': response_text,
            'done_reason': raw.get('done_reason', ''), 'num_predict': max_output,
            'usage_known': True}
    try:
        answer = json.loads(response_text)
        if not isinstance(answer, dict):
            raise ValueError('JSON root is not an object')
        if set(answer) != set(schema['required']):
            raise ValueError('JSON fields are missing or unexpected')
        if type(answer['missing_dependency']) is not bool or not isinstance(answer['explanation'], str):
            raise ValueError('Invalid dependency flag or explanation type')
        if len(answer['explanation']) > 160:
            raise ValueError('Explanation exceeds structured response limit')
        if (any(type(answer[field]) not in (float, int) for field in ('suspicion', 'uncertainty'))
                or not all(math.isfinite(answer[field]) and 0 <= answer[field] <= 1
                           for field in ('suspicion', 'uncertainty'))):
            raise ValueError('Invalid numeric scores in JSON response')
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'invalid_model_output', 'error': str(exc)[:400], **base}
    return {'status': 'ok', 'score': float(answer['suspicion']),
            'uncertainty': float(answer['uncertainty']),
            'missing_dependency': answer['missing_dependency'],
            'reason': answer['explanation'], **base}


def compression_prompt(evidence, compressor):
    if compressor is None:
        return None
    answer = compressor.compress_prompt(evidence, rate=0.5)
    if not isinstance(answer, dict) or not answer.get('compressed_prompt'):
        raise RuntimeError('LLMLingua-2 returned empty compressed text')
    return answer['compressed_prompt']


def collect(manifest, output, url, l1_model, full_model, max_bugs, max_candidates,
            max_calls, use_llmlingua2=False, physical_token_cap=500000):
    """Collect independent real LLM observations, logging every paid attempt.

    A malformed response receives one bounded retry. It is never assigned a
    synthetic score. Failed attempts count toward the physical token cap.
    """
    validate(manifest)
    output = Path(output)
    if output.exists():
        saved = load(output)
        if saved.get('manifest_hash') != sha(manifest):
            raise ValueError('Refusing to resume with changed manifest')
        if saved.get('models') != {'l1': l1_model, 'full': full_model}:
            raise ValueError('Refusing to resume with different model identity')
        if any(o.get('status') != 'ok' for o in saved['observations']):
            raise ValueError('Legacy failed observation has unknown physical usage. '
                             'Preserve this file and use a NEW output directory for a clean study.')
        if 'attempts' not in saved:
            saved['legacy_observation_tokens'] = sum(
                o['input_tokens'] + o['output_tokens'] for o in saved['observations'])
            saved['attempts'] = []
        # An interrupted atomic write leaves the previous valid checkpoint and
        # a partial .tmp file. Preserve the latter for auditing, and disclose
        # that the last paid model call may be missing from token accounting.
        interrupted = output.with_name(output.name + '.tmp')
        if interrupted.exists():
            archive = output.with_name(output.name + '.interrupted-write.bak')
            if not archive.exists():
                shutil.copy2(interrupted, archive)
            warning = ('Abandoned atomic checkpoint detected and preserved as ' + str(archive)
                       + '. A paid model call may be missing from the saved ledger. '
                       'Physical token totals are a lower bound; logged observations '
                       'remain suitable for matched logical policy replay.')
            saved['physical_accounting_complete'] = False
            warnings = saved.setdefault('accounting_warnings', [])
            if warning not in warnings:
                warnings.append(warning)
    else:
        saved = {'schema': 'research-agent-observations-v1',
                 'provenance': 'live_ollama', 'models': {'l1': l1_model, 'full': full_model},
                 'manifest_hash': sha(manifest), 'observations': [], 'attempts': [],
                 'legacy_observation_tokens': 0}
    done = obs_index(saved)
    compressor = None
    if use_llmlingua2:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
                                      use_llmlingua2=True)
    calls = 0
    total_tokens = (saved.get('legacy_observation_tokens', 0) +
                    sum(a['input_tokens'] + a['output_tokens'] for a in saved['attempts']
                        if a.get('usage_known')))

    def outcome(paused):
        return {'new_calls': calls, 'physical_tokens': total_tokens,
                'physical_accounting_complete': saved.get('physical_accounting_complete', True),
                'accounting_warnings': saved.get('accounting_warnings', []),
                'logged_attempts': len(saved['attempts']),
                'invalid_output_calls': sum(a.get('status') == 'invalid_model_output'
                                            for a in saved['attempts']),
                'limit_reached': paused, 'file': str(output)}

    for bug in manifest['bugs'][:max_bugs]:
        candidates = sorted_candidates(bug)[:max_candidates]
        lookup = {c['id']: c for c in bug['candidates']}
        compressed = compression_prompt(bug['evidence'], compressor)
        for c in candidates:
            dep_text = '\n\n'.join('Dependency ' + dep + ':\n' + lookup[dep]['full']
                                     for dep in c.get('depends_on', []))
            tasks = [('l1', c['l1'], bug['evidence'], l1_model),
                     ('full', c['full'], bug['evidence'], full_model),
                     ('card', code_card(c['full']), bug['evidence'], l1_model)]
            snippet_text = '\n'.join(c.get('support_snippets', []))
            if dep_text or snippet_text:
                tasks.append(('dependency', c['l1'] + '\n\nSUPPORT:\n' + dep_text + '\n' + snippet_text,
                              bug['evidence'], l1_model))
            if compressed:
                tasks.append(('llmlingua2', c['l1'], compressed, l1_model))
            for kind, source, evidence, model in tasks:
                key = observation_key(bug['id'], c['id'], kind)
                if key in done:
                    continue
                previous = [a for a in saved['attempts']
                            if observation_key(a['bug'], a['candidate'], a['kind']) == key]
                if any(a.get('status') not in ('ok', 'invalid_model_output') for a in previous):
                    raise RuntimeError('Unknown provider usage for prior call: ' + key)
                if len(previous) >= 2:
                    raise RuntimeError('Two invalid responses recorded, manual review required: ' + key)
                for try_index in range(len(previous), 2):
                    if calls >= max_calls or total_tokens >= physical_token_cap:
                        save(output, saved)
                        return outcome(True)
                    try:
                        result = ollama_call(url, model, evidence, c['id'], source, kind,
                                             max_output=384 if try_index == 0 else 768)
                    except Exception as exc:
                        item = {'bug': bug['id'], 'candidate': c['id'], 'kind': kind,
                                'status': 'request_error_unknown_usage', 'usage_known': False,
                                'error': str(exc)[:1000]}
                        saved['attempts'].append(item)
                        save(output, saved)
                        raise RuntimeError('Provider call failed with unknown usage; '
                                           'do not silently retry: ' + key) from exc
                    item = {'bug': bug['id'], 'candidate': c['id'], 'kind': kind, **result}
                    saved['attempts'].append(item)
                    calls += 1
                    total_tokens += result['input_tokens'] + result['output_tokens']
                    if result['status'] == 'ok':
                        saved['observations'].append(item)
                        done[key] = item
                    save(output, saved)
                    if result['status'] == 'ok':
                        break
                    if try_index == 1:
                        raise RuntimeError('Two invalid JSON replies, both attempts preserved with '
                                           'physical token usage; no fabricated score: ' + key)
    return outcome(False)


def uncertainty_priority(c, seen, graph=False):
    o = seen.get(c['id'])
    if o is None:
        return -1
    uncertainty = float(o.get('uncertainty', 0))
    risk = 0.65 * uncertainty + 0.35 * (1 - abs(o['score'] - 0.5) * 2)
    if graph and (c.get('depends_on') or c.get('support_snippets')):
        risk += 0.20 * min(len(c.get('depends_on', [])) + len(c.get('support_snippets', [])), 3) / 3
    if o.get('missing_dependency'):
        risk += 0.15
    return risk + 0.12 * float(c['prior'])


def evaluate_arm(bug, obs, arm, budget, buffer=1, max_verify=2, conformal_threshold=None):
    """Replay decisions made from priors/current observations, never gold or future scores."""
    candidates = sorted_candidates(bug)
    lookup = {c['id']: c for c in candidates}
    k = adaptive_k(candidates, buffer)
    selected = set(c['id'] for c in candidates[:k])
    state = {c['id']: float(c['prior']) for c in candidates}
    initial = {c['id']: float(c['prior']) for c in candidates}
    seen = {}
    actions, spent, stopped, nverify = [], 0, None, 0
    policy_stop_reason = None

    def take(c, kind):
        nonlocal spent, stopped
        if stopped:
            return False
        record = obs.get(observation_key(bug['id'], c['id'], kind))
        if record is None or record.get('status') != 'ok':
            stopped = 'missing_real_observation:' + kind + ':' + c['id']
            return False
        # In replay, recorded cost is charged only after a candidate/action has been chosen.
        # Admission uses a conservative fixed allowance (not future actual usage).
        estimated = min(budget, max(100, len(c['full'] if kind == 'full' else c['l1']) // 3 + 150))
        if spent + estimated > budget:
            stopped = 'estimated_budget_exhausted'
            return False
        actual = record['input_tokens'] + record['output_tokens']
        spent += actual
        # A real backend overrun cannot be reversed; it stops further actions and is reported.
        if spent > budget:
            stopped = 'measured_budget_overrun'
        score = float(record['score'])
        state[c['id']] = score if kind == 'full' else 0.30 * initial[c['id']] + 0.70 * score
        seen[c['id']] = record
        actions.append({'candidate': c['id'], 'kind': kind, 'tokens': actual,
                        'score': round(score, 5), 'uncertainty': record['uncertainty']})
        return True

    if arm == 'adaptive_k':
        base = [c for c in candidates if c['id'] in selected]
    elif arm == 'conformal_k':
        if conformal_threshold is None:
            return {'bug': bug['id'], 'arm': arm, 'status': 'not_calibrated',
                    'ranked': [c['id'] for c in candidates], 'tokens': 0, 'llm_calls': 0,
                    'full_calls': 0, 'selected_k': 0, 'actions': [], 'budget': budget, 'overrun': False}
        base = [c for c in candidates if float(c['prior']) >= conformal_threshold]
    else:
        base = candidates
    base_kind = ('card' if arm in ('code_cards', 'code_cards_risk_full',
                                    'card_rank_full', 'card_failure_full',
                                    'card_stop_full', 'card_egav')
                 else 'llmlingua2' if arm == 'llmlingua2' else 'l1')
    for c in base:
        if not take(c, base_kind):
            break
    if not stopped and arm in ('dependency_repair', 'combined'):
        dep_candidates = sorted((c for c in candidates if c.get('depends_on') or c.get('support_snippets')),
                                key=lambda c: (-uncertainty_priority(c, seen, graph=True), c['id']))
        for c in dep_candidates[:max_verify]:
            if not take(c, 'dependency'):
                break
    # New experimental policies read only the *current* card/full evidence.
    # Recompute target order after each expensive response. The saved Full
    # response cannot affect whether its candidate is selected.
    if not stopped and arm in ('card_rank_full', 'card_failure_full',
                               'card_stop_full', 'card_egav'):
        from egav_policies import ranked_full_targets, evidence_sufficient, STOP_POLICIES
        checked = set()
        for _ in range(max_verify):
            if arm in STOP_POLICIES and evidence_sufficient(candidates, state, seen):
                policy_stop_reason = 'strong_file_lead_heuristic'
                break
            target = next((candidate for candidate in ranked_full_targets(
                candidates, state, seen, bug['evidence'], arm, uncertainty_priority)
                if candidate['id'] not in checked), None)
            if target is None:
                break
            if not take(target, 'full'):
                break
            checked.add(target['id'])
            nverify += 1
    if not stopped and arm in ('risk_full', 'code_cards_risk_full', 'cascade', 'esr_proxy', 'combined'):
        if arm == 'esr_proxy':
            # Influence heuristic inspired by ESR, not OECP's exact-verifier theorem.
            influences = {c['id']: 1 + sum(c['id'] in x.get('depends_on', [])
                                          for x in candidates) for c in candidates}
            ranked = sorted(candidates, key=lambda c: (-uncertainty_priority(c, seen) * influences[c['id']], c['id']))
        else:
            ranked = sorted(candidates, key=lambda c: (-uncertainty_priority(c, seen, graph=True), c['id']))
        for c in ranked[:max_verify]:
            if arm == 'cascade' and uncertainty_priority(c, seen) < 0.50:
                continue
            if not take(c, 'full'):
                break
            nverify += 1
    ranked_final = sorted(candidates, key=lambda c: (-state[c['id']], c['id']))
    return {'bug': bug['id'], 'arm': arm, 'status': 'complete' if not stopped else stopped,
            'ranked': [c['id'] for c in ranked_final], 'tokens': spent, 'llm_calls': len(actions),
            'full_calls': nverify, 'selected_k': k if arm == 'adaptive_k' else len(candidates),
            'actions': actions, 'budget': budget, 'overrun': spent > budget,
            'policy_stop_reason': policy_stop_reason}


def metric(ranking, gold):
    ranks = [i + 1 for i, x in enumerate(ranking) if x in gold]
    if not ranks:
        return {'mrr': 0., 'top1': 0, 'top5': 0, 'gold_rank': None}
    r = min(ranks)
    return {'mrr': 1 / r, 'top1': int(r == 1), 'top5': int(r <= 5), 'gold_rank': r}


def conformal_threshold(calibration_bugs, gold_map, alpha=0.1):
    """Split-conformal marginal candidate coverage, under exchangeability.

    If the calibration set is too small, include all candidates rather than
    incorrectly claiming a fixed-size / exact method guarantee.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between 0 and 1')
    losses = []
    for bug in calibration_bugs:
        gold = set(gold_map[bug['id']]['functions'])
        pool = {c['id'] for c in bug['candidates']}
        if not gold or not gold.issubset(pool):
            raise ValueError('Conformal calibration needs complete method gold inside pool')
        best = max(float(c['prior']) for c in bug['candidates'] if c['id'] in gold)
        losses.append(1 - best)
    if not losses:
        raise ValueError('Calibration bugs required')
    index = math.ceil((len(losses) + 1) * (1 - alpha))
    if index > len(losses):
        return 0.0  # Include all, safe small-calibration fallback.
    return max(0.0, 1.0 - sorted(losses)[index - 1])


def import_rave_inputs(run_dir, output, max_bugs=10, per_project=2):
    """Import authentic rave_fl/eval/runner.py instances/*/input.json inputs.

    Call-name links are hypotheses, NOT proved semantic dependencies.
    This command never reads evaluation_only.json or reference patches.
    """
    files = sorted(Path(run_dir).glob('instances/*/input.json'))
    if not files:
        raise ValueError('No RAVE-FL input.json found under ' + str(run_dir))
    by_project = {}
    for f in files:
        source = load(f)
        online = source['online']
        project = online['instance_id'].split('-', 1)[0]
        if len(by_project.get(project, [])) >= per_project:
            continue
        entities = {e['entity_id']: e for e in source['entities']}
        candidates = []
        for c in source['candidates']:
            entity = entities[c['function_id']]
            levels = c.get('levels') or []
            if len(levels) < 3:
                raise ValueError('Prepared instance lacks L1/full sketches: ' + str(f))
            candidates.append({'id': c['function_id'], 'prior': c['prior'],
                               'l1': levels[1], 'full': levels[2],
                               'file': entity['path'], 'depends_on': [],
                               'dependency_provenance': 'none_declared'})
        short = {}
        for c in candidates:
            name = c['id'].split('::')[-1].split('(')[0].split('.')[-1]
            short.setdefault(name, []).append(c['id'])
        for c in candidates:
            hits = []
            for name, ids in short.items():
                if len(ids) == 1 and ids[0] != c['id'] and re.search(r'\b' + re.escape(name) + r'\s*\(', c['full']):
                    hits.append(ids[0])
            c['depends_on'] = sorted(set(hits))
            if hits:
                c['dependency_provenance'] = 'heuristic_call_name_not_verified'
        by_project.setdefault(project, []).append({
            'id': online['instance_id'], 'dataset': online['dataset'],
            'evidence': online['evidence']['text'],
            'source_input_sha256': sha(source), 'candidates': candidates})
    picks = []
    # Interleave projects, avoid accidentally selecting only one project.
    while len(picks) < max_bugs and any(by_project.values()):
        for project in sorted(by_project):
            if by_project[project] and len(picks) < max_bugs:
                picks.append(by_project[project].pop(0))
    result = {'schema': 'research-method-agent-v1', 'bugs': picks}
    validate(result)
    save(output, result)
    return {'bugs': [b['id'] for b in picks], 'output': str(output),
            'note': 'Call-name edges are heuristic and must be checked'}


def import_rave_gold(run_dir, manifest, output):
    """Separate offline import, never called during observation collection."""
    mapping = {}
    approved = {b['id'] for b in manifest['bugs']}
    for path in Path(run_dir).glob('instances/*/evaluation_only.json'):
        data = load(path)
        source = load(path.with_name('input.json'))
        bid = source['online']['instance_id']
        if bid in approved:
            mapping[bid] = {'functions': data.get('functions', []),
                            'files': data.get('files', []),
                            'semantics': data.get('semantics', 'unspecified')}
    result = {'schema': 'research-agent-gold-v1', 'bugs': mapping}
    save(output, result)
    return {'bugs_with_labels': len(mapping), 'output': str(output)}


def import_graft_config(config_path, output, bug_ids=(), candidate_limit=24):
    """Gold-free import from an existing GRAFT 100-bug batch config.

    Requires immutable original checkouts and the existing local RAVE-FL package.
    Uses GRAFT's actual typed AST graph; MAY_CALL is only a candidate hypothesis.
    """
    from rave_fl.eval.adapters import load_instances
    from rave_fl.eval.runner import prepare_one
    from rave_fl.graft.graph import build_graph

    config = load(config_path)
    opts = config['benchmark']
    instances, _ = load_instances(opts)
    expected = set(bug_ids or opts.get('bugs') or [])
    cases = []
    for instance in instances:
        if expected and instance.instance_id not in expected:
            continue
        prepared, entities, candidates, metadata, _, value = prepare_one(instance, opts)
        if candidate_limit and len(candidates) > candidate_limit:
            candidates = candidates[:candidate_limit]
        selected = {c.function_id for c in candidates}
        ent = {e.entity_id: e for e in entities}
        graph = build_graph(prepared, entities)
        by_node = graph.by_id
        links = {ident: set() for ident in selected}
        snippets = {ident: [] for ident in selected}
        for edge in graph.edges:
            node = by_node[edge.source]
            if edge.relation == 'MAY_CALL' and edge.target in graph.targets.values():
                current = node.owner
                target = next((k for k, v in graph.targets.items() if v == edge.target), None)
                if current in selected and target in selected and target != current:
                    links[current].add(target)
            elif edge.relation == 'REQUIRES' and node.owner in selected:
                required = by_node[edge.target]
                if required.kind == 'code' and required.owner == node.owner:
                    snippets[node.owner].append(required.text)
        rows = []
        for c in candidates:
            entity = ent[c.function_id]
            extra = [t for t in dict.fromkeys(snippets[c.function_id])
                     if t not in c.levels[1]][:4]
            rows.append({'id': c.function_id, 'prior': c.prior,
                         'l1': c.levels[1], 'full': c.levels[2],
                         'file': entity.path,
                         'depends_on': sorted(links[c.function_id]),
                         'support_snippets': extra,
                         'dependency_provenance': 'GRAFT AST MAY_CALL/REQUIRES; not a causal proof'})
        cases.append({'id': instance.instance_id, 'dataset': 'defects4j',
                      'evidence': instance.evidence.text, 'candidates': rows,
                      'source_revision': instance.revision, 'graph_sha256': graph.hash,
                      'original_pool_n': len(entities),
                      'candidate_limit': candidate_limit})
    if expected and {c['id'] for c in cases} != expected:
        raise ValueError('Requested bug(s) unavailable in exact prepared GRAFT config')
    result = {'schema': 'research-method-agent-v1', 'bugs': sorted(cases, key=lambda c: c['id'])}
    validate(result)
    save(output, result)
    return {'bugs': [c['id'] for c in cases], 'output': str(output),
            'note': 'GRAFT MAY_CALL and REQUIRES graph hypotheses, not exact fault dependencies'}


def import_graft_gold(config_path, manifest, output):
    """Offline gold import from GRAFT batch config dataset_file, never an online call."""
    config = load(config_path)
    dataset = load(config['benchmark']['dataset_file'])
    mapping = {}
    expected = {b['id'] for b in manifest['bugs']}
    for case in dataset['cases']:
        if case['bug'] in expected:
            gold = case['gold']
            mapping[case['bug']] = {'files': gold.get('files', []),
                                     'functions': gold.get('functions', []),
                                     'semantics': gold.get('semantics', 'unspecified')}
    if set(mapping) != expected:
        raise ValueError('Missing required gold labels, no sample substitution')
    save(output, {'schema': 'research-agent-gold-v1', 'bugs': mapping})
    return {'bugs_with_labels': len(mapping), 'output': str(output)}


def summarize(results, baseline='l1'):
    complete = {(r['bug'], r['arm']): r for r in results if r['status'] == 'complete'}
    bugs = sorted({r['bug'] for r in results})
    arms = sorted({r['arm'] for r in results})
    summary = []
    rng = random.Random(42)
    for arm in arms:
        paired = [(complete[(bid, baseline)], complete[(bid, arm)]) for bid in bugs
                  if (bid, baseline) in complete and (bid, arm) in complete and
                  'mrr' in complete[(bid, baseline)] and 'mrr' in complete[(bid, arm)]]
        if not paired:
            summary.append({'arm': arm, 'paired_n': 0, 'status': 'insufficient_data'})
            continue
        delta = [b['mrr'] - a['mrr'] for a, b in paired]
        cost = [b['tokens'] - a['tokens'] for a, b in paired]
        ci = None
        if len(paired) >= 5:
            draws = [statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(2000)]
            draws.sort()
            ci = [round(draws[49], 5), round(draws[1949], 5)]
        summary.append({'arm': arm, 'paired_n': len(paired),
                        'mean_mrr': round(statistics.mean(b['mrr'] for _, b in paired), 5),
                        'delta_mrr_vs_l1': round(statistics.mean(delta), 5),
                        'delta_mrr_ci95': ci,
                        'mean_top1': round(statistics.mean(b['top1'] for _, b in paired), 5),
                        'mean_top5': round(statistics.mean(b['top5'] for _, b in paired), 5),
                        'mean_tokens': round(statistics.mean(b['tokens'] for _, b in paired), 2),
                        'mean_token_delta': round(statistics.mean(cost), 2),
                        'win_tie_loss': [sum(x > 0 for x in delta), sum(x == 0 for x in delta),
                                         sum(x < 0 for x in delta)]})
    return summary


def write_report(output, results, summary, manifest, observation):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'results.json', {'schema': 'research-agent-evaluation-v1',
                                    'manifest_sha256': sha(manifest),
                                    'observation_sha256': sha(observation),
                                    'provenance': observation.get('provenance', 'unspecified'),
                                    'physical_accounting_complete': observation.get('physical_accounting_complete', True),
                                    'accounting_warnings': observation.get('accounting_warnings', []),
                                    'results': results, 'summary': summary})
    fields = ['arm', 'paired_n', 'mean_mrr', 'delta_mrr_vs_l1', 'mean_top1',
              'mean_top5', 'mean_tokens', 'mean_token_delta', 'win_tie_loss', 'status']
    with (output / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for item in summary:
            w.writerow(item)
    lines = ['# Evidence selection pilot report', '',
             'Source: ' + observation.get('provenance', 'unspecified') + '.',
             'These are observation-replay evaluations, not live policy wall-clock timings.',
             'All token counts below are logical policy costs using recorded provider tokens.',
             'Collection of unused observations has additional physical cost.',
             'Gold is read only for this offline report.', '',
             '| Arm | Matched bugs | MRR | Change vs L1 | Top-1 | Top-5 | Mean tokens |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        if s['paired_n']:
            lines.append(f"| {s['arm']} | {s['paired_n']} | {s['mean_mrr']:.3f} | "
                         f"{s['delta_mrr_vs_l1']:+.3f} | {s['mean_top1']:.3f} | "
                         f"{s['mean_top5']:.3f} | {s['mean_tokens']:.0f} |")
        else:
            lines.append(f"| {s['arm']} | 0 | N/A | N/A | N/A | N/A | N/A |")
    lines += ['', '## Interpretation',
              'A promising arm must improve accuracy or preserve it with meaningful token savings '
              'on complete matched bugs. Confidence intervals require at least five independent bugs.',
              'No arm is declared superior on synthetic fixtures, incomplete arms, or one bug.',
              'Code cards are an inexpensive *proxy*, not GistRank latent gist tokens.',
              'ESR proxy is a scheduling heuristic, not an OECP exact-containment verifier.',
              'Adaptive-k is based on local prior gaps, not guaranteed top-k recall.',
              'The optional LLMLingua-2 arm uses actual compressed failure text only if collected.']
    if observation.get('physical_accounting_complete') is False:
        lines.insert(5, '**PHYSICAL USAGE INCOMPLETE:** an interrupted save may have lost one paid model call. '
                        'Collection token totals are only a lower bound, even though recorded '
                        'logical policy costs use real measured tokens.')
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def evaluate(manifest, observation, labels, output, budget, max_verify, buffer, alpha=0.1):
    validate(manifest)
    if observation.get('manifest_hash') != sha(manifest):
        raise ValueError('Observations were collected for a different manifest')
    if labels.get('schema') != 'research-agent-gold-v1':
        raise ValueError('Gold file must use research-agent-gold-v1')
    data = obs_index(observation)
    results = []
    calibrations = [b for b in manifest['bugs'] if b.get('split') == 'calibration']
    evaluated = [b for b in manifest['bugs'] if b.get('split') != 'calibration']
    threshold = (conformal_threshold(calibrations, labels['bugs'], alpha)
                 if calibrations else None)
    levels = set()
    for bug in evaluated:
        label = labels.get('bugs', {}).get(bug['id'], {})
        if label.get('functions'):
            gold = set(label['functions'])
            level = 'method'
            pool = {c['id'] for c in bug['candidates']}
        elif label.get('files'):
            gold = set(label['files'])
            level = 'file_proxy'
            pool = {c.get('file') for c in bug['candidates']}
        else:
            raise ValueError('Missing real method/file gold for bug ' + bug['id'])
        pool_hit = bool(gold.intersection(pool))
        levels.add(level)
        for arm in ARMS:
            result = evaluate_arm(bug, data, arm, budget, buffer=buffer,
                                  max_verify=max_verify, conformal_threshold=threshold)
            if level == 'file_proxy':
                by_id = {c['id']: c for c in bug['candidates']}
                # Rank distinct files, don't count duplicated candidate method hits.
                ranked = list(dict.fromkeys(by_id[x].get('file') for x in result['ranked']))
            else:
                ranked = result['ranked']
            result.update(metric(ranked, gold))
            result['gold_semantics'] = label.get('semantics', level)
            result['gold_level'] = level
            result['pool_hit'] = pool_hit
            results.append(result)
    if len(levels) > 1:
        raise ValueError('Cannot aggregate file and method localization results')
    summary = summarize(results)
    write_report(output, results, summary, manifest, observation)
    return {'bugs': len(evaluated), 'calibration_bugs': len(calibrations),
            'conformal_threshold': threshold, 'gold_level': next(iter(levels)),
            'complete_arms': sum(r['status'] == 'complete' for r in results),
            'total_arms': len(results), 'report': str(Path(output) / 'report.md')}


def svg_tradeoff(entries, path):
    """Portable SVG plot of exploratory MRR change vs measured logical token delta."""
    from xml.sax.saxutils import escape
    pts = [p for p in entries if p['arm'] != 'l1' and p['paired_n'] >= 2]
    if not pts:
        Path(path).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="100"><text x="20" y="50">No paired evidence for plot</text></svg>')
        return
    w, h, left, right, top, bottom = 800, 470, 95, 45, 50, 70
    xs = [x['mean_token_delta'] for x in pts] + [0]
    ys = [x['delta_mrr_vs_l1'] for x in pts] + [0]
    mnx, mxx = min(xs), max(xs)
    mny, mxy = min(ys), max(ys)
    if mnx == mxx: mnx, mxx = mnx - 1, mxx + 1
    if mny == mxy: mny, mxy = mny - 0.02, mxy + 0.02
    dx, dy = (mxx - mnx) * 0.08, (mxy - mny) * 0.08
    mnx, mxx, mny, mxy = mnx - dx, mxx + dx, mny - dy, mxy + dy
    def px(x): return left + (x - mnx) * (w - left - right) / (mxx - mnx)
    def py(y): return h - bottom - (y - mny) * (h - bottom - top) / (mxy - mny)
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="30" y="25" font-size="17">MRR change vs token cost, exploratory replay</text>',
             f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#333"/>',
             f'<line x1="{px(0):.1f}" y1="{top}" x2="{px(0):.1f}" y2="{h-bottom}" stroke="#888" stroke-dasharray="5,5"/>',
             f'<line x1="{left}" y1="{py(0):.1f}" x2="{w-right}" y2="{py(0):.1f}" stroke="#888" stroke-dasharray="5,5"/>',
             f'<text x="{(left+w-right)/2:.1f}" y="{h-18}" text-anchor="middle" font-size="13">Mean logical token difference vs L1, negative = fewer tokens</text>',
             f'<text x="22" y="{h/2}" transform="rotate(-90 22 {h/2})" text-anchor="middle" font-size="13">Mean MRR difference vs L1</text>']
    for val in (mnx, 0, mxx):
        items.append(f'<text x="{px(val):.1f}" y="{h-bottom+19}" font-size="11" text-anchor="middle">{val:.0f}</text>')
    for val in (mny, 0, mxy):
        items.append(f'<text x="{left-9}" y="{py(val)+4:.1f}" font-size="11" text-anchor="end">{val:+.3f}</text>')
    for item in pts:
        tip = escape(f"{item['arm']}, {item['config']}, paired={item['paired_n']}, deltaMRR={item['delta_mrr_vs_l1']:+.4f}, deltaTokens={item['mean_token_delta']:+.0f}")
        items.append(f'<circle cx="{px(item["mean_token_delta"]):.1f}" cy="{py(item["delta_mrr_vs_l1"]):.1f}" r="5" fill="#3c699f" fill-opacity="0.45"><title>{tip}</title></circle>')
    items.append('<text x="30" y="449" font-size="11" fill="#555">Each point is one arm/budget configuration on matched bugs, not an independent study.</text>')
    items.append('</svg>')
    Path(path).write_text('\n'.join(items), encoding='utf-8')


def run_sweep(manifest, observation, labels, output, budgets, max_verifies, buffers, alpha=0.1):
    """Offline grid search for *development only*, no new model calls.

    Reports all configurations, avoids selecting a single flattering result.
    """
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for budget in sorted(set(budgets)):
        for verify in sorted(set(max_verifies)):
            for buffer in sorted(set(buffers)):
                if budget < 100 or verify < 0 or buffer < 0:
                    raise ValueError('Invalid sweep budget, verification count or buffer')
                config = f'budget_{budget}_full_{verify}_buffer_{buffer}'
                directory = root / config
                evaluate(manifest, observation, labels, directory, budget, verify, buffer, alpha)
                record = load(directory / 'results.json')
                for s in record['summary']:
                    if s['paired_n']:
                        entries.append({'config': config, 'arm': s['arm'], 'budget': budget,
                                        'verify': verify, 'buffer': buffer, **s})
    # The comparisons below are a screening index only, not hypothesis tests.
    # Requiring paired observations prevents incomplete arms from becoming wins.
    promoted = [x for x in entries if x['arm'] != 'l1' and x['paired_n'] >= 5
                and (x['delta_mrr_vs_l1'] > 0.001 or
                     (x['delta_mrr_vs_l1'] >= -0.01 and x['mean_token_delta'] <= -200))]
    promoted.sort(key=lambda x: (-x['delta_mrr_vs_l1'], x['mean_token_delta'], -x['paired_n']))
    save(root / 'grid.json', {'schema':'method-discovery-sweep-v1',
                             'provenance': observation.get('provenance', 'unspecified'),
                             'configs': entries, 'exploratory_candidates': promoted})
    lines = ['# Method-discovery sweep', '',
             f"Provenance: {observation.get('provenance', 'unspecified')}. No new model calls.",
             'All candidates below are exploratory and must be tested on untouched bugs.',
             'Grid search over multiple budgets risks selection bias; it is not a statistical proof.', '',
             '| Candidate arm | Budget | Full cap | Buffer | Paired n | Delta MRR | Delta tokens |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in promoted[:30]:
        lines.append(f"| {s['arm']} | {s['budget']} | {s['verify']} | {s['buffer']} | "
                     f"{s['paired_n']} | {s['delta_mrr_vs_l1']:+.4f} | {s['mean_token_delta']:+.0f} |")
    if not promoted:
        lines.append('| No candidate met the exploratory screen | | | | | | |')
    lines += ['', 'Run an independent held-out evaluation before claiming a promising result.',
              'Inspect regressions, actual call counts and dependency coverage in each config results.json.']
    (root / 'sweep_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    svg_tradeoff(entries, root / 'tradeoff.svg')
    return {'configs': len(set(x['config'] for x in entries)),
            'candidate_configs': len(promoted), 'report': str(root / 'sweep_report.md')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('collect', help='Independent real Ollama observations, no gold loaded')
    b = sub.add_parser('evaluate', help='Offline paired replay, gold loaded here only')
    c = sub.add_parser('import-rave', help='Import RAVE-FL prepared instance inputs')
    d = sub.add_parser('import-gold', help='Import RAVE-FL offline labels separately')
    e = sub.add_parser('import-graft', help='Import existing GRAFT batch, no model calls')
    f = sub.add_parser('import-graft-gold', help='Import GRAFT gold strictly offline')
    for z in (e, f):
        z.add_argument('--config', required=True)
        z.add_argument('--output', required=True)
    e.add_argument('--bugs', default='')
    e.add_argument('--candidate-limit', type=int, default=24)
    f.add_argument('--manifest', required=True)
    sweep = sub.add_parser('sweep', help='Cost/accuracy grid from saved real observations, no new calls')
    sweep.add_argument('--manifest', required=True)
    sweep.add_argument('--observations', required=True)
    sweep.add_argument('--gold', required=True)
    sweep.add_argument('--output', required=True)
    sweep.add_argument('--budgets', default='16000,24000,40000,60000')
    sweep.add_argument('--max-verifies', default='1,2,3')
    sweep.add_argument('--buffers', default='0,1')
    sweep.add_argument('--alpha', type=float, default=0.10)
    study = sub.add_parser('study', help='One-command GRAFT sample preparation and optional real study')
    study.add_argument('--config', required=True)
    study.add_argument('--output-dir', required=True)
    study.add_argument('--bugs', default='Chart-17,Closure-165,Lang-28,Math-95,Time-26')
    study.add_argument('--candidate-limit', type=int, default=24)
    study.add_argument('--allow-inference', action='store_true')
    study.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    study.add_argument('--l1-model', default='qwen2.5:7b')
    study.add_argument('--full-model', default='qwen2.5:7b')
    study.add_argument('--max-calls', type=int, default=450)
    study.add_argument('--physical-token-cap', type=int, default=500000)
    study.add_argument('--budget', type=int, default=60000)
    study.add_argument('--max-verify', type=int, default=2)
    study.add_argument('--llmlingua2', action='store_true')
    for z in (c, d):
        z.add_argument('--run-dir', required=True)
        z.add_argument('--output', required=True)
    c.add_argument('--max-bugs', type=int, default=10)
    c.add_argument('--per-project', type=int, default=2)
    d.add_argument('--manifest', required=True)
    for z in (a, b):
        z.add_argument('--manifest', required=True)
        z.add_argument('--output', required=True)
    a.add_argument('--observations', required=True)
    a.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    a.add_argument('--l1-model', default='qwen2.5:7b')
    a.add_argument('--full-model', default='qwen2.5:7b')
    a.add_argument('--max-bugs', type=int, default=5)
    a.add_argument('--max-candidates', type=int, default=100000)
    a.add_argument('--max-calls', type=int, default=100)
    a.add_argument('--physical-token-cap', type=int, default=500000)
    a.add_argument('--llmlingua2', action='store_true')
    b.add_argument('--observations', required=True)
    b.add_argument('--gold', required=True)
    b.add_argument('--budget', type=int, default=20000)
    b.add_argument('--max-verify', type=int, default=2)
    b.add_argument('--adaptive-buffer', type=int, default=1)
    b.add_argument('--alpha', type=float, default=0.10)
    args = p.parse_args(argv)
    if args.command == 'sweep':
        manifest = load(args.manifest)
        def values(s):
            return [int(x.strip()) for x in s.split(',') if x.strip()]
        print(json.dumps(run_sweep(manifest, load(args.observations), load(args.gold),
              args.output, values(args.budgets), values(args.max_verifies),
              values(args.buffers), args.alpha), indent=2))
        return
    if args.command == 'study':
        out = Path(args.output_dir)
        path = out / 'manifest.json'
        bugs = tuple(b.strip() for b in args.bugs.split(',') if b.strip())
        prep = import_graft_config(args.config, path, bugs, args.candidate_limit)
        if not args.allow_inference:
            print(json.dumps({'status': 'prepared_only', 'inference_calls': 0, **prep}, indent=2))
            return
        manifest = load(path)
        collected = collect(manifest, out / 'observations.json', args.ollama_url,
                            args.l1_model, args.full_model, len(manifest['bugs']),
                            100000, args.max_calls, args.llmlingua2, args.physical_token_cap)
        if collected['limit_reached']:
            print(json.dumps({'status': 'collection_paused_at_cap', **collected}, indent=2))
            return
        goldfile = out / 'gold_offline.json'
        import_graft_gold(args.config, manifest, goldfile)
        result = evaluate(manifest, load(out / 'observations.json'), load(goldfile),
                          out / 'evaluation', args.budget, args.max_verify, 1)
        print(json.dumps({'status': 'evaluated', 'collection': collected, 'evaluation': result}, indent=2))
        return
    if args.command == 'import-graft':
        bugs = tuple(b.strip() for b in args.bugs.split(',') if b.strip())
        print(json.dumps(import_graft_config(args.config, args.output, bugs, args.candidate_limit), indent=2))
        return
    if args.command == 'import-rave':
        print(json.dumps(import_rave_inputs(args.run_dir, args.output, args.max_bugs, args.per_project), indent=2))
        return
    manifest = load(args.manifest)
    if args.command == 'import-graft-gold':
        print(json.dumps(import_graft_gold(args.config, manifest, args.output), indent=2))
        return
    if args.command == 'import-gold':
        print(json.dumps(import_rave_gold(args.run_dir, manifest, args.output), indent=2))
        return
    if args.command == 'collect':
        result = collect(manifest, args.observations, args.ollama_url, args.l1_model,
                         args.full_model, args.max_bugs, args.max_candidates,
                         args.max_calls, args.llmlingua2, args.physical_token_cap)
    else:
        if args.budget < 100 or args.max_verify < 0:
            p.error('Budget must be >=100 and max-verify nonnegative')
        result = evaluate(manifest, load(args.observations), load(args.gold),
                          args.output, args.budget, args.max_verify, args.adaptive_buffer, args.alpha)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()