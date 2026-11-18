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
import random
import re
import statistics
import time
import urllib.request
from pathlib import Path

ARMS = (
    'l1', 'adaptive_k', 'conformal_k', 'dependency_repair', 'risk_full', 'esr_proxy',
    'cascade', 'code_cards', 'combined', 'llmlingua2',
)
OBS_KINDS = ('l1', 'full', 'dependency', 'card', 'llmlingua2')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')


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


def ollama_call(url, model, evidence, candidate, text, kind, max_output=130):
    prompt = (
        'You are evaluating one candidate code location for a software failure. '
        'Use only the observed failure and code supplied. Do not invent source facts. '
        'Return JSON with numeric suspicion between 0 and 1, numeric uncertainty between 0 and 1, '
        'boolean missing_dependency, and a brief explanation.\n'
        'Failure evidence:\n' + evidence[:12000] + '\n'
        'Candidate ID: ' + candidate + '\n'
        'Evidence form: ' + kind + '\nCode evidence:\n' + text[:24000]
    )
    data = {'model': model, 'stream': False, 'format': 'json',
            'messages': [{'role': 'user', 'content': prompt}],
            'options': {'temperature': 0, 'seed': 42, 'num_predict': max_output}}
    payload = json.dumps(data).encode()
    request = urllib.request.Request(url.rstrip('/') + '/api/chat', data=payload,
                                     headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=360) as response:
        raw = json.loads(response.read())
    elapsed = time.monotonic() - started
    if type(raw.get('prompt_eval_count')) is not int or type(raw.get('eval_count')) is not int:
        raise RuntimeError('Provider did not return exact input/output token counts')
    answer = json.loads(raw['message']['content'])
    score = float(answer['suspicion'])
    uncertainty = float(answer['uncertainty'])
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (score, uncertainty)):
        raise RuntimeError('LLM output scores must be in [0,1]')
    return {'status': 'ok', 'score': score, 'uncertainty': uncertainty,
            'missing_dependency': answer.get('missing_dependency') is True,
            'reason': str(answer.get('explanation', ''))[:700],
            'input_tokens': raw['prompt_eval_count'], 'output_tokens': raw['eval_count'],
            'duration_seconds': elapsed, 'model': model, 'prompt_sha256': sha(data)}


def compression_prompt(evidence, compressor):
    if compressor is None:
        return None
    answer = compressor.compress_prompt(evidence, rate=0.5)
    if not isinstance(answer, dict) or not answer.get('compressed_prompt'):
        raise RuntimeError('LLMLingua-2 returned empty compressed text')
    return answer['compressed_prompt']


def collect(manifest, output, url, l1_model, full_model, max_bugs, max_candidates,
            max_calls, use_llmlingua2=False):
    """Collect independent real LLM observations; no labels are loaded here."""
    validate(manifest)
    output = Path(output)
    if output.exists():
        saved = load(output)
        if saved.get('manifest_hash') != sha(manifest):
            raise ValueError('Refusing to resume with changed manifest')
    else:
        saved = {'schema': 'research-agent-observations-v1',
                 'provenance': 'live_ollama', 'manifest_hash': sha(manifest), 'observations': []}
    done = obs_index(saved)
    compressor = None
    if use_llmlingua2:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
                                      use_llmlingua2=True)
    calls = 0
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
            if dep_text:
                tasks.append(('dependency', c['l1'] + '\n\nSUPPORT:\n' + dep_text,
                              bug['evidence'], l1_model))
            if compressed:
                tasks.append(('llmlingua2', c['l1'], compressed, l1_model))
            for kind, source, evidence, model in tasks:
                key = observation_key(bug['id'], c['id'], kind)
                if key in done:
                    continue
                if calls >= max_calls:
                    save(output, saved)
                    return {'new_calls': calls, 'limit_reached': True, 'file': str(output)}
                try:
                    result = ollama_call(url, model, evidence, c['id'], source, kind)
                except Exception as exc:
                    result = {'status': 'error', 'error': str(exc)[:1000]}
                    item = {'bug': bug['id'], 'candidate': c['id'], 'kind': kind, **result}
                    saved['observations'].append(item)
                    save(output, saved)
                    raise RuntimeError('Real collection failed; error saved, no fake result: ' + key) from exc
                item = {'bug': bug['id'], 'candidate': c['id'], 'kind': kind, **result}
                saved['observations'].append(item)
                done[key] = item
                calls += 1
                save(output, saved)
    return {'new_calls': calls, 'limit_reached': False, 'file': str(output)}


def uncertainty_priority(c, seen, graph=False):
    o = seen.get(c['id'])
    if o is None:
        return -1
    uncertainty = float(o.get('uncertainty', 0))
    risk = 0.65 * uncertainty + 0.35 * (1 - abs(o['score'] - 0.5) * 2)
    if graph and c.get('depends_on'):
        risk += 0.20 * min(len(c['depends_on']), 3) / 3
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
    base_kind = 'card' if arm == 'code_cards' else 'llmlingua2' if arm == 'llmlingua2' else 'l1'
    for c in base:
        if not take(c, base_kind):
            break
    if not stopped and arm in ('dependency_repair', 'combined'):
        dep_candidates = sorted((c for c in candidates if c.get('depends_on')),
                                key=lambda c: (-uncertainty_priority(c, seen, graph=True), c['id']))
        for c in dep_candidates[:max_verify]:
            if not take(c, 'dependency'):
                break
    if not stopped and arm in ('risk_full', 'cascade', 'esr_proxy', 'combined'):
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
            'actions': actions, 'budget': budget, 'overrun': spent > budget}


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('collect', help='Independent real Ollama observations, no gold loaded')
    b = sub.add_parser('evaluate', help='Offline paired replay, gold loaded here only')
    c = sub.add_parser('import-rave', help='Import RAVE-FL prepared instance inputs')
    d = sub.add_parser('import-gold', help='Import RAVE-FL offline labels separately')
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
    a.add_argument('--llmlingua2', action='store_true')
    b.add_argument('--observations', required=True)
    b.add_argument('--gold', required=True)
    b.add_argument('--budget', type=int, default=20000)
    b.add_argument('--max-verify', type=int, default=2)
    b.add_argument('--adaptive-buffer', type=int, default=1)
    b.add_argument('--alpha', type=float, default=0.10)
    args = p.parse_args(argv)
    if args.command == 'import-rave':
        print(json.dumps(import_rave_inputs(args.run_dir, args.output, args.max_bugs, args.per_project), indent=2))
        return
    manifest = load(args.manifest)
    if args.command == 'import-gold':
        print(json.dumps(import_rave_gold(args.run_dir, manifest, args.output), indent=2))
        return
    if args.command == 'collect':
        result = collect(manifest, args.observations, args.ollama_url, args.l1_model,
                         args.full_model, args.max_bugs, args.max_candidates,
                         args.max_calls, args.llmlingua2)
    else:
        if args.budget < 100 or args.max_verify < 0:
            p.error('Budget must be >=100 and max-verify nonnegative')
        result = evaluate(manifest, load(args.observations), load(args.gold),
                          args.output, args.budget, args.max_verify, args.adaptive_buffer, args.alpha)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()