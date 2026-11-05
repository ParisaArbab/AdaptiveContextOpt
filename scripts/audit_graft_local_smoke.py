"""Offline audit of a completed GRAFT smoke run; never invokes a provider."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rave_fl.eval.schema import CodeEntity, GoldLabels, file_hash, safe_name
from rave_fl.eval.fusion import Fusion
from rave_fl.pilot_types import PilotCandidate, CandidateState, Observation
from rave_fl.graft.evaluation import evaluate_profile


def audit(run_dir, output):
    root, output = Path(run_dir).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((root / 'results.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    settings = manifest['config']['benchmark']
    spending = json.loads((root / 'accounting.json').read_text())
    connection = sqlite3.connect(f'file:{root / "journal.sqlite3"}?mode=ro', uri=True)
    paid = connection.execute('select key,instance,status,input,output from calls').fetchall()
    connection.close()
    physical = sum((r[3] or 0) + (r[4] or 0) for r in paid)
    assert physical == spending['known_provider_tokens']
    summaries, evolution, checks = [], [], []
    inputs = {}
    for row in rows:
        ident = row['instance_id']
        folder = root / 'instances' / safe_name(ident)
        if (folder / 'input.json').exists():
            inputs[ident] = json.loads((folder / 'input.json').read_text())
        arm = row.get('arm', {})
        calls = arm.get('calls', [])
        total = sum(c['input_tokens'] + c['output_tokens'] for c in calls)
        actual = row.get('accounted_tokens')
        if arm:
            checks.append({'bug': ident, 'policy': row['policy'], 'call_cost_matches': total == actual})
            assert total == actual, 'Paid-call mismatch; investigate before reporting'
        metric = row.get('metrics', {}).get('java_strict_v2', {}).get('method') or {}
        stages = Counter(c.get('stage', 'method_full' if c.get('level') == 2 else 'sketch') for c in calls)
        invalid = sum(not o['valid'] for c in calls for o in c['observations'])
        summary = {'bug': ident, 'policy': row['policy'], 'status': row['status'],
                   'strict_gold_rank': metric.get('first_relevant_rank'), 'strict_mrr': metric.get('MRR'),
                   'verification_tokens': actual, 'calls': len(calls), 'stage_calls': dict(stages),
                   'invalid_observations': invalid, 'stop_reason': arm.get('stop_reason'), 'error': row.get('error')}
        summaries.append(summary)
        if not calls or ident not in inputs:
            continue
        data = inputs[ident]
        candidates = [PilotCandidate(**c) for c in data['candidates']]
        entities = [CodeEntity(**e) for e in data['entities']]
        gold = GoldLabels(**json.loads((folder / 'evaluation_only.json').read_text()))
        opts = manifest['config']['benchmark']
        fusion = Fusion(row['fusion'], opts['tie'], opts['rounding'], opts['rrf_k'])
        states = {c.function_id: CandidateState() for c in candidates}
        spent = 0
        def ranking():
            return [c.function_id for c in fusion.order(candidates, fusion.scores(candidates, states))]
        def metrics():
            return evaluate_profile({'ranked_functions': ranking()}, entities, gold, 'defects4j')['method'] or {}
        for step, call in enumerate(calls):
            before = metrics()
            stage = call.get('stage', 'method_full' if call.get('level') == 2 else 'sketch')
            level = 2 if stage == 'method_full' else 1
            for o in call['observations']:
                states[o['function_id']] = CandidateState(level, Observation(**o))
            after = metrics()
            spent += call['input_tokens'] + call['output_tokens']
            evolution.append({'bug': ident, 'policy': row['policy'], 'step': step, 'stage': stage,
                              'selected': call.get('candidate_ids'), 'rank_before': before.get('first_relevant_rank'),
                              'rank_after': after.get('first_relevant_rank'),
                              'mrr_delta': after.get('MRR', 0) - before.get('MRR', 0), 'spent': spent,
                              'call_tokens': call['input_tokens'] + call['output_tokens'], 'error': call.get('error')})
        assert ranking() == arm['ranked_functions'], 'Ranking reconstruction mismatch'
    budget_violations = [r for r in rows if r.get('arm', {}).get('budget_exceeded')]
    assert not budget_violations
    complete_bugs = {b for b in inputs if all(r['status'] == 'completed' for r in rows if r['instance_id'] == b)}
    summary = {'run': str(root), 'environment': json.loads((root / 'environment.json').read_text()),
               'selected_bugs': [i['instance_id'] for i in manifest['instances']], 'joint_complete_bugs': sorted(complete_bugs),
               'evidence_kinds': {b: d['online']['evidence']['kind'] for b, d in inputs.items()},
               'rows': summaries, 'spending': spending, 'journal_statuses': dict(Counter(r[2] for r in paid)),
               'cost_checks': checks, 'budget_violations': len(budget_violations),
               'source_results_sha256': file_hash(root / 'results.json'), 'new_inference_by_auditor': 0}
    (output / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'ranking_evolution.json').write_text(json.dumps(evolution, indent=2) + '\n')
    lines = ['# GRAFT local real-model smoke results', '',
             'Fresh local Llama3.1 8B development smoke; this is not an A100 measurement or independent held-out validation. Existing Time-1/Lang-1/Math-1 inputs were selected for availability before inference; the separate Lang-54/Math-80/Closure-61 protocol remains unexecuted.', '',
             f"The evidence is supplied trigger-test text, not newly generated observed failure logs. Official benchmark tests were not run. Math-1 source-span incompatibility remains documented in the initial run. Condition: {root.name}. All arms share the pool and configured fusion. Context: {settings['context_window']}; output per candidate: {settings['output_tokens']}; batch size: {settings['batch_size']}; logical verification budget: {settings['budget']}. Physical caps: {settings['physical_budget']} run / {settings['per_bug_physical_budget']} per bug / {settings['max_new_calls']} new requests.", '',
             '| Bug | Policy | Status | Strict gold rank | Verification tokens | Calls | Invalid observations |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['bug']} | {r['policy']} | {r['status']} | {r['strict_gold_rank']} | {r['verification_tokens']} | {r['calls']} | {r['invalid_observations']} |")
    lines += ['', 'Ranks from incomplete states are partial observations and must not be compared as completed accuracy. Missing labels are unavailable, not zero accuracy. The run REPORT.md separates operational and matched cohorts.', '',
              '## Accounting and reproducibility', '', '```json', json.dumps(spending, indent=2), '```', '',
              'All reconstructed final rankings and logical paid-call sums matched saved arms. Journal physical counters match accounting.json. No recorded logical-budget violation occurred. Exact requests/responses and selected/reused fingerprints remain in the source run. Logical policy costs include reused observations; physical totals count each paid request once. Retrieval/embedding/build costs are excluded.', '',
              '## Upgrade traces (offline interpretation)', '',
              '| Bug | Policy | Stage | Selected candidate | Rank before | Rank after | MRR change | Tokens |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for r in evolution:
        if r['stage'] == 'sketch' or r['policy'] == 'full':
            continue
        lines.append(f"| {r['bug']} | {r['policy']} | {r['stage']} | {', '.join(r['selected'] or [])} | {r['rank_before']} | {r['rank_after']} | {r['mrr_delta']:.4f} | {r['call_tokens']} |")
    lines += ['', 'These marginal changes use gold labels only after execution. They are descriptive outcomes of these states, not deployable utility features. More graph evidence need not improve ranking, and method Full is not an oracle. Two reused development defects cannot establish statistical superiority or justify training a calibrated utility model.', '',
              'Sources: manifest.json, environment.json, results.json, per-instance input/evaluation_only files and journal.sqlite3 in the recorded run. audit.json retains the provider/model identity and metric rows; ranking_evolution.json retains every paid transition.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if summaries and all(r['status'] == 'completed' for r in summaries):
        plot(summary, output)
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = summary['rows']
    settings = json.loads((Path(summary['run']) / 'manifest.json').read_text())['config']['benchmark']
    bugs = sorted({r['bug'] for r in rows})
    policies = list(dict.fromkeys(r['policy'] for r in rows))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    x = np.arange(len(policies))
    width = .8 / len(bugs)
    colors = ['#0072B2', '#E69F00', '#009E73']
    for n, bug in enumerate(bugs):
        group = {r['policy']: r for r in rows if r['bug'] == bug}
        offset = (n - (len(bugs) - 1) / 2) * width
        axes[0].bar(x + offset, [group[p]['verification_tokens'] for p in policies], width, label=bug, color=colors[n % 3])
        axes[1].bar(x + offset, [group[p]['strict_gold_rank'] or float('nan') for p in policies], width, color=colors[n % 3])
    axes[0].set_ylabel('Verification input + output tokens')
    axes[0].legend()
    axes[1].set_ylabel('Strict gold rank (lower is better)')
    axes[1].set_xticks(x, [p.replace('_', '\n') for p in policies])
    axes[1].set_yticks([1, 2, 3])
    axes[1].set_ylim(0, 3)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"Local {settings['model']}: {len(bugs)} previously used development defects")
    fig.text(.02, .01, f"Source: audit.json, complete arms; policy budget {settings['budget']}, context {settings['context_window']}.\nLogical verification cost includes exact reused responses. No held-out, A100 or statistical-superiority claim.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    for extension in ['png', 'svg']:
        fig.savefig(Path(output) / ('comparison.' + extension), dpi=220, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = audit(a.run_dir, a.output)
    print(json.dumps({'spending': result['spending'], 'rows': result['rows']}, indent=2))
