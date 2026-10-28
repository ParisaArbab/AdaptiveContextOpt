"""Verify a completed pilot against durable provider logs and offline metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from rave_fl.core import Action
from rave_fl.pilot import prepare_case, write_json
from rave_fl.pilot_backend import build_prompt
from rave_fl.pilot_evaluation import evaluate, summarize
from wp1.flexfl_data import read_ground_truth


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run_path, manifest_path):
    run = json.loads(run_path.read_text())
    require(run.get('status') == 'complete' and not run.get('dry_run'), 'Require a completed real experiment')
    manifest = {c['bug']: c for c in json.loads(manifest_path.read_text())['cases']}
    summary = {'status': 'verified', 'cases': [], 'total_paid_calls': 0, 'total_provider_tokens': 0}
    for case in run['cases']:
        manifest_case = manifest[case['bug']]
        candidates, context, provenance = prepare_case(manifest_case, case['top_per_method'])
        require(provenance['sources_sha256'] == case['sources_sha256'], 'Source changed since inference')
        require(provenance['evidence']['sha256'] == case['evidence']['sha256'], 'Evidence changed since inference')
        ids = {c.function_id for c in candidates}
        by_id = {c.function_id: c for c in candidates}
        truths = read_ground_truth(Path(manifest_case['flexfl_repo']), case['bug'])
        audited = {'bug': case['bug'], 'arms': {}}
        for name, arm in case['arms'].items():
            path = run_path.parent / case['bug'] / (name + '_calls.jsonl')
            events = [json.loads(s) for s in path.read_text().splitlines()]
            require(not any(e['event'] == 'error' for e in events), f'{name}: transport/provider error')
            requests = [e for e in events if e['event'] == 'request']
            responses = [e for e in events if e['event'] == 'response']
            require(len(requests) == len(responses) == arm['call_count'], f'{name}: incomplete call pairs')
            input_tokens = output_tokens = 0
            for request, response, call in zip(requests, responses, arm['calls']):
                require(request['call_id'] == response['call_id'], f'{name}: mispaired call IDs')
                require(request['request']['model'] == run['model'], f'{name}: model changed')
                system, user = build_prompt([by_id[i] for i in request['candidate_ids']], context,
                                             Action(request['action']), request['level'])
                require(request['request']['messages'] == [{'role': 'system', 'content': system},
                                                           {'role': 'user', 'content': user}], f'{name}: prompt/source/context mismatch')
                raw = response['response']
                require(call['input_tokens'] == raw['prompt_eval_count'] and call['output_tokens'] == raw['eval_count'],
                        f'{name}: provider usage mismatch')
                require(call['candidate_ids'] == request['candidate_ids'] and call['action'] == request['action']
                        and call['level'] == request['level'], f'{name}: action mismatch')
                require(call['raw_response'] == raw['message']['content'], f'{name}: response mismatch')
                input_tokens += raw['prompt_eval_count']
                output_tokens += raw['eval_count']
            require(arm['actual_input_tokens'] == input_tokens and arm['actual_output_tokens'] == output_tokens,
                    f'{name}: token totals disagree')
            require(arm['actual_tokens'] == input_tokens + output_tokens, f'{name}: total tokens disagree')
            require(arm['tokens_remaining'] == arm['config']['token_budget'] - arm['actual_tokens'], f'{name}: budget mismatch')
            require(len(arm['ranked_functions']) == len(ids) and set(arm['ranked_functions']) == ids,
                    f'{name}: ranking dropped or duplicated candidates')
            require(evaluate(arm['ranked_functions'], truths, arm['states']) == arm['accuracy'],
                    f'{name}: offline metrics do not match')
            paid_trace = [t for t in arm['trace'] if t['action'] != 'STOP']
            require(len(paid_trace) == arm['call_count'], f'{name}: missing decisions')
            for trace, call in zip(paid_trace, arm['calls']):
                require(trace['actual_tokens'] == call['input_tokens'] + call['output_tokens'], f'{name}: trace usage mismatch')
            audited['arms'][name] = {'calls': len(responses), 'actual_tokens': arm['actual_tokens'],
                                    'invalid_responses': sum(c['error'] is not None for c in arm['calls']),
                                    'baseline_complete': arm['full_verified_count'] == len(ids) and not arm['failed_full_ids']}
            summary['total_paid_calls'] += len(responses)
            summary['total_provider_tokens'] += arm['actual_tokens']
        summary['cases'].append(audited)
    require(summarize(run['cases']) == run['summary'], 'Paired summary differs from per-case outcomes')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    try:
        result = audit(a.run, a.manifest)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        p.error(str(exc))
    if a.output:
        write_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
