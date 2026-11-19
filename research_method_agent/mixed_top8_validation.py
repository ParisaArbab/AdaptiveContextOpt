"""Frozen Top-8 L1 + Code Cards evaluation on 15 new Defects4J bugs.

The selection policy is fixed: sort by BM25 prior, use L1 for first eight
methods and deterministic Code Cards for remaining candidates. All 24
candidates are ranked with the same 0.3 prior + 0.7 Qwen suspicion fusion.
Gold is loaded only during report generation. No Full or dependency calls.

Usage:
  python research_method_agent/mixed_top8_validation.py preflight
  python research_method_agent/mixed_top8_validation.py run-03 --allow-inference
  python research_method_agent/mixed_top8_validation.py run-05 --allow-inference
  python research_method_agent/mixed_top8_validation.py run-07 --allow-inference
  python research_method_agent/mixed_top8_validation.py report
  python research_method_agent/mixed_top8_validation.py status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research_method_agent'))
import research_method_agent as agent

BATCHES = ('03', '05', '07')
OLD_BUGS = {
    'Chart-17', 'Closure-165', 'Lang-28', 'Math-95', 'Time-26',
    'Chart-1', 'Closure-128', 'Lang-35', 'Math-57', 'Time-14',
    'Chart-12', 'Closure-32', 'Lang-6', 'Math-56', 'Time-6',
}
OUTPUT = ROOT / 'results' / 'mixed_top8_validation_15bugs'
SPEC = {
    'schema': 'frozen-top8-l1-cards-v1',
    'batches': list(BATCHES),
    'top_k_l1': 8,
    'candidate_count': 24,
    'token_budget_per_bug': 60000,
    'model': 'qwen2.5:7b',
    'prompt': 'unchanged research_method_agent.ollama_call prompt, card and l1 kinds',
    'fusion': '0.30 prior + 0.70 independent suspicion; tie candidate id',
    'seed': 42,
    'temperature': 0,
    'gold_level': 'file_proxy',
    'stage': 'new development confirmation, not final untouched test',
}


def file_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def protocol():
    result = {**SPEC, 'implementation_sha256': file_sha()}
    path = OUTPUT / 'frozen_protocol.json'
    if path.is_file():
        if agent.load(path) != result:
            raise RuntimeError('Frozen Top-8 protocol/code was changed. Use a NEW output root.')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        agent.save(path, result)
    return result


def batch_config(batch):
    if batch not in BATCHES:
        raise ValueError('Unknown predeclared batch ' + batch)
    path = ROOT / f'data/graft_100_cohort_20261115/batches/batch_{batch}/experiment/config.json'
    conf = agent.load(path)
    benchmark = conf['benchmark']
    bugs = tuple(benchmark['bugs'])
    if len(bugs) != 5 or len(set(bugs)) != 5 or set(bugs) & OLD_BUGS:
        raise RuntimeError('Batch overlaps earlier experiment or lacks five distinct cases: ' + str(bugs))
    if benchmark['candidate_provider'] != 'bm25' or benchmark['candidate_count'] != 24:
        raise RuntimeError('Expected original BM25 method pool of exactly 24')
    if not Path(benchmark['dataset_file']).exists():
        raise RuntimeError('Original prepared GRAFT manifest is missing: ' + benchmark['dataset_file'])
    return path, bugs


def out_dir(batch):
    return OUTPUT / ('batch_' + batch)


def preflight():
    protocol()
    used = set(OLD_BUGS)
    for batch in BATCHES:
        config, bugs = batch_config(batch)
        if used & set(bugs):
            raise RuntimeError('Duplicate bug among cohorts')
        used |= set(bugs)
        dest = out_dir(batch)
        manifest_path = dest / 'manifest.json'
        if manifest_path.exists():
            manifest = agent.load(manifest_path)
            if {b['id'] for b in manifest['bugs']} != set(bugs):
                raise RuntimeError('Saved manifest does not match frozen selected bugs')
            agent.validate(manifest)
        else:
            print('Preparing new bugs:', ', '.join(bugs), flush=True)
            agent.import_graft_config(config, manifest_path, bugs, candidate_limit=24)
            manifest = agent.load(manifest_path)
        if len(manifest['bugs']) != 5:
            raise RuntimeError('Expected exactly five prepared bugs')
        if any(len(b['candidates']) > 24 for b in manifest['bugs']):
            raise RuntimeError('Candidate pool unexpectedly exceeds 24')
        print('READY', batch, sorted(b['id'] for b in manifest['bugs']))
    print('Prepared 15 new bugs. No model calls or gold were used.')


def kind_for_rank(rank):
    return 'l1' if rank < SPEC['top_k_l1'] else 'card'


def tasks(bug):
    """All independent L1/Card observations needed for three matched arms.

    Collect both forms for all candidates so L1, Cards and Mixed policies
    have genuine measured responses. Selection of Mixed is frozen by prior.
    """
    for c in agent.sorted_candidates(bug):
        yield c, 'l1', c['l1']
        yield c, 'card', agent.code_card(c['full'])


def collect(batch, args):
    if not args.allow_inference:
        raise RuntimeError('Paid Qwen calls require explicit --allow-inference')
    protocol()
    config, bugs = batch_config(batch)
    dest = out_dir(batch)
    path = dest / 'manifest.json'
    if not path.exists():
        raise RuntimeError('Run preflight before model inference')
    manifest = agent.load(path)
    agent.validate(manifest)
    if {b['id'] for b in manifest['bugs']} != set(bugs):
        raise RuntimeError('Wrong bug cohort in saved manifest')
    ledger = dest / 'observations.json'
    if ledger.exists():
        data = agent.load(ledger)
        if data.get('manifest_hash') != agent.sha(manifest):
            raise RuntimeError('Manifest changed after observations were collected')
        if data.get('models') != {'l1': SPEC['model'], 'full': SPEC['model']}:
            raise RuntimeError('Model changed since first observation')
        if data.get('collection_scope') != ['l1', 'card']:
            raise RuntimeError('Unexpected collection scope')
        if any(x.get('status') != 'ok' for x in data['observations']):
            raise RuntimeError('Legacy malformed observation; preserve it for audit')
        temporary = ledger.with_name(ledger.name + '.tmp')
        if temporary.exists():
            backup = ledger.with_name(ledger.name + '.interrupted-write.bak')
            if not backup.exists():
                shutil.copy2(temporary, backup)
            data['physical_accounting_complete'] = False
            data.setdefault('accounting_warnings', []).append('Interrupted checkpoint, physical usage is a lower bound')
    else:
        data = {
            'schema': 'research-agent-observations-v1',
            'provenance': 'live_ollama',
            'models': {'l1': SPEC['model'], 'full': SPEC['model']},
            'manifest_hash': agent.sha(manifest),
            'collection_scope': ['l1', 'card'],
            'observations': [], 'attempts': [],
            'physical_accounting_complete': True,
            'accounting_warnings': [],
        }
    indexed = agent.obs_index(data)
    calls = 0
    physical = sum(int(a['input_tokens']) + int(a['output_tokens'])
                   for a in data['attempts'] if a.get('usage_known'))
    if any(not a.get('usage_known') for a in data['attempts']):
        raise RuntimeError('Unknown provider usage, cannot safely resume')

    def finish(paused):
        agent.save(ledger, data)
        status = {
            'status': 'collection_paused_at_cap' if paused else 'collected',
            'batch': batch, 'new_calls': calls,
            'logged_attempts': len(data['attempts']),
            'recorded_physical_tokens': physical,
            'physical_accounting_complete': data.get('physical_accounting_complete', True),
            'valid_observations': len(data['observations']),
            'expected_observations': sum(2 * len(b['candidates']) for b in manifest['bugs']),
            'observations_file': str(ledger),
        }
        print(json.dumps(status, indent=2))
        return status

    for bug in manifest['bugs']:
        for c, kind, source in tasks(bug):
            key = agent.observation_key(bug['id'], c['id'], kind)
            if key in indexed:
                continue
            previous = [x for x in data['attempts']
                        if agent.observation_key(x['bug'], x['candidate'], x['kind']) == key]
            if len(previous) >= 2:
                raise RuntimeError('Two invalid model replies for ' + key)
            if any(x['status'] not in ('ok', 'invalid_model_output') for x in previous):
                raise RuntimeError('Unexpected attempt status in ledger')
            for attempt in range(len(previous), 2):
                if calls >= args.max_calls or physical >= args.physical_token_cap:
                    return finish(True)
                try:
                    response = agent.ollama_call(
                        args.ollama_url, SPEC['model'], bug['evidence'],
                        c['id'], source, kind, max_output=384 if attempt == 0 else 768)
                except Exception as exc:
                    data['attempts'].append({
                        'bug': bug['id'], 'candidate': c['id'], 'kind': kind,
                        'status': 'request_error_unknown_usage', 'usage_known': False,
                        'error': str(exc)[:1000]})
                    agent.save(ledger, data)
                    raise RuntimeError('Unknown token usage, stop and inspect saved ledger') from exc
                item = {'bug': bug['id'], 'candidate': c['id'], 'kind': kind, **response}
                data['attempts'].append(item)
                physical += response['input_tokens'] + response['output_tokens']
                calls += 1
                if response['status'] == 'ok':
                    data['observations'].append(item)
                    indexed[key] = item
                agent.save(ledger, data)
                if response['status'] == 'ok':
                    break
                if attempt == 1:
                    raise RuntimeError('Two invalid model outputs, both attempts preserved')
    return finish(False)


def mixed_rank(bug, observations, token_budget=60000):
    """Gold-blind fixed Top-8 L1 selection, independent of future Qwen scores."""
    candidates = agent.sorted_candidates(bug)
    state = {}
    tokens = 0
    actions = []
    for i, c in enumerate(candidates):
        kind = kind_for_rank(i)
        rec = observations.get(agent.observation_key(bug['id'], c['id'], kind))
        if rec is None or rec.get('status') != 'ok':
            return {'status': 'missing_real_observation', 'ranked': [], 'tokens': tokens, 'actions': actions}
        real = int(rec['input_tokens']) + int(rec['output_tokens'])
        tokens += real
        actions.append({'candidate': c['id'], 'kind': kind, 'tokens': real})
        state[c['id']] = 0.30 * float(c['prior']) + 0.70 * float(rec['score'])
    ordered = sorted(candidates, key=lambda c: (-state[c['id']], c['id']))
    return {'status': 'complete' if tokens <= token_budget else 'measured_budget_overrun',
            'ranked': [c['id'] for c in ordered], 'tokens': tokens, 'actions': actions,
            'llm_calls': len(actions), 'full_calls': 0}


def metric_for_bug(bug, ranking, gold):
    by_id = {c['id']: c.get('file') for c in bug['candidates']}
    files = list(dict.fromkeys(by_id[cid] for cid in ranking))
    return agent.metric(files, set(gold))


def bootstrap_paired_ci(diffs, repeats=10000):
    if len(diffs) < 5:
        return None
    rng = random.Random(42)
    n = len(diffs)
    boot = sorted(statistics.mean(diffs[rng.randrange(n)] for _ in range(n))
                  for _ in range(repeats))
    return [boot[int(.025 * repeats)], boot[min(repeats - 1, int(.975 * repeats))]]


def report():
    protocol()
    rows, all_bugs = [], set()
    for batch in BATCHES:
        config, bugs = batch_config(batch)
        directory = out_dir(batch)
        manifest = agent.load(directory / 'manifest.json')
        ledger = agent.load(directory / 'observations.json')
        if ledger.get('manifest_hash') != agent.sha(manifest):
            raise RuntimeError('Manifest mismatch in ' + batch)
        if ledger.get('provenance') != 'live_ollama':
            raise RuntimeError('Not real Qwen observations')
        index = agent.obs_index(ledger)
        gold_path = directory / 'gold_offline.json'
        if not gold_path.exists():
            agent.import_graft_gold(config, manifest, gold_path)
        gold_data = agent.load(gold_path)['bugs']
        for bug in manifest['bugs']:
            bid = bug['id']
            if bid in all_bugs:
                raise RuntimeError('Overlap in independent bug cohorts')
            all_bugs.add(bid)
            gold = gold_data[bid].get('files', [])
            if not gold:
                raise RuntimeError('Expected file-level modified-class gold')
            pool_hit = bool(set(gold) & {c.get('file') for c in bug['candidates']})
            for arm in ('l1', 'code_cards', 'mixed_top8'):
                replay = (mixed_rank(bug, index)
                          if arm == 'mixed_top8'
                          else agent.evaluate_arm(bug, index, arm, 60000, max_verify=0))
                metrics = metric_for_bug(bug, replay['ranked'], gold) if replay['status'] == 'complete' else {
                    'mrr': None, 'top1': None, 'top5': None, 'gold_rank': None}
                rows.append({
                    'bug': bid, 'arm': arm, 'status': replay['status'],
                    'gold_level': 'file_proxy', 'pool_hit': pool_hit,
                    'tokens': replay['tokens'], **metrics,
                })
    if len(all_bugs) != 15:
        raise RuntimeError('Expected 15 distinct bugs, got ' + str(len(all_bugs)))
    bykey = {(r['bug'], r['arm']): r for r in rows}
    summary = []
    for arm in ('l1', 'code_cards', 'mixed_top8'):
        pairs = [(bykey[b, 'l1'], bykey[b, arm]) for b in sorted(all_bugs)
                 if bykey[b, 'l1']['status'] == 'complete' and bykey[b, arm]['status'] == 'complete']
        item = {'arm': arm, 'paired_n': len(pairs), 'total_bugs': len(all_bugs)}
        if pairs:
            diffs = [b['mrr'] - a['mrr'] for a, b in pairs]
            item.update({
                'mrr': statistics.mean(b['mrr'] for a, b in pairs),
                'top1': statistics.mean(b['top1'] for a, b in pairs),
                'top5': statistics.mean(b['top5'] for a, b in pairs),
                'tokens': statistics.mean(b['tokens'] for a, b in pairs),
                'delta_mrr_vs_l1': statistics.mean(diffs),
                'delta_mrr_ci95': bootstrap_paired_ci(diffs),
                'delta_tokens_vs_l1': statistics.mean(b['tokens'] - a['tokens'] for a, b in pairs),
                'win_tie_loss': [sum(b['mrr'] > a['mrr'] for a, b in pairs),
                                 sum(b['mrr'] == a['mrr'] for a, b in pairs),
                                 sum(b['mrr'] < a['mrr'] for a, b in pairs)],
            })
        summary.append(item)
    payload = {'schema': 'mixed-top8-15bug-study-v1', 'spec': SPEC,
               'note': 'New development check, not an untouched final test',
               'summary': summary, 'results': rows}
    agent.save(OUTPUT / 'results.json', payload)
    lines = ['# Frozen Top-8 L1 + Code Cards, 15 new development bugs', '',
             'Independent real Qwen2.5:7b responses. File-level modified-class gold proxy.',
             'The Top-8 selection was fixed before these 15 new bugs were run.',
             'All tokens are measured *logical policy* tokens; physical collection also runs unused replies.', '',
             '| Arm | Paired n/15 | MRR | Top-1 | Top-5 | Mean tokens | Delta MRR vs L1 | Delta tokens vs L1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in summary:
        if x['paired_n']:
            lines.append(f"| {x['arm']} | {x['paired_n']}/15 | {x['mrr']:.4f} | {x['top1']:.2f} | {x['top5']:.2f} | {x['tokens']:.0f} | {x['delta_mrr_vs_l1']:+.4f} | {x['delta_tokens_vs_l1']:+.0f} |")
        else:
            lines.append(f"| {x['arm']} | 0/15 | N/A | N/A | N/A | N/A | N/A | N/A |")
    lines += ['', 'Paired MRR confidence intervals and regressions:', '']
    for x in summary:
        if x['paired_n']:
            lines.append(f"- {x['arm']}: CI={x['delta_mrr_ci95']}, wins/ties/losses={x['win_tie_loss']}")
    lines += ['', '## Per-bug file rank and MRR', '',
              '| Bug | Gold in top 24 | L1 rank | Code Card rank | Mixed rank | Mixed token delta vs L1 |',
              '|---|---|---:|---:|---:|---:|']
    for bid in sorted(all_bugs):
        a, b, c = [bykey[(bid, arm)] for arm in ('l1', 'code_cards', 'mixed_top8')]
        def val(r):
            return str(r['gold_rank']) if r['status'] == 'complete' and r['gold_rank'] else 'MISS'
        lines.append(f"| {bid} | {a['pool_hit']} | {val(a)} | {val(b)} | {val(c)} | {c['tokens'] - a['tokens']:+d} |")
    lines += ['', 'Do not claim an improvement if paired n < 15 or the CI contains zero.',
              'This does not demonstrate a latency speedup or method-level localization.',
              'Mixed Card and L1 prompts differ in both evidence and the Evidence form label.']
    report_path = OUTPUT / 'report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(report_path.read_text(encoding='utf-8'))
    print('Saved:', report_path)


def status():
    protocol()
    for batch in BATCHES:
        destination = out_dir(batch)
        manifest_file = destination / 'manifest.json'
        observed_file = destination / 'observations.json'
        print('Batch', batch, 'prepared:', manifest_file.exists())
        if manifest_file.exists():
            manifest = agent.load(manifest_file)
            print(' Bugs:', [b['id'] for b in manifest['bugs']])
            expected = 2 * sum(len(b['candidates']) for b in manifest['bugs'])
            if observed_file.exists():
                obs = agent.load(observed_file)
                print(' Calls/valid/expected:', len(obs['attempts']),
                      len(obs['observations']), expected)
                print(' Physical accounting complete:', obs.get('physical_accounting_complete', True))
        print(' Report exists:', (OUTPUT / 'report.md').exists())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'run-03', 'run-05', 'run-07', 'report', 'status'])
    p.add_argument('--allow-inference', action='store_true')
    p.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    p.add_argument('--max-calls', type=int, default=300)
    p.add_argument('--physical-token-cap', type=int, default=800000)
    args = p.parse_args(argv)
    if args.command == 'preflight':
        preflight()
    elif args.command.startswith('run-'):
        collect(args.command[-2:], args)
    elif args.command == 'report':
        report()
    else:
        status()


if __name__ == '__main__':
    main()