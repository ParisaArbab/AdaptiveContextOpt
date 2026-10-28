"""Bounded independent-bug feasibility, with exact within-run request reuse.

No new request is sent unless the global remaining physical token allowance
can reserve an entire context window plus capped output. Accounted per-policy
tokens still include every selected cached action's historical provider cost.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from rave_fl.adaptive import AdaptivePilot
from rave_fl.budget_reservation import ReservedBackend
from rave_fl.core import Action
from rave_fl.expanded_pool import union_pool
from rave_fl.pilot import digest, prepare_case, write_json
from rave_fl.pilot_backend import OllamaPilotBackend, build_prompt
from rave_fl.pilot_evaluation import evaluate
from rave_fl.pilot_types import PolicyConfig
from rave_fl.selection_controls import make_engine, RANKERS


class GlobalLimit(RuntimeError):
    pass


class SharedExecution:
    def __init__(self, cap, max_calls):
        self.cap, self.max_calls = cap, max_calls
        self.provider_tokens = self.new_calls = 0
        self.cache = {}
        self.events = []
        self.usage_unknown = False

    def execute(self, backend, candidates, action, level):
        system, user = build_prompt(candidates, backend.context, action, level)
        key = digest({'prompt': [system, user], 'ids': [c.function_id for c in candidates],
                      'action': action.value, 'level': level, 'model': backend.model_metadata['digest'],
                      'seed': backend.seed, 'temperature': 0, 'think': False, 'format': 'json',
                      'num_ctx': backend.context_window, 'num_predict': backend.output_per_candidate * len(candidates)})
        if key in self.cache:
            self.events.append({'event': 'selected_exact_cache_reuse', 'key': key, 'ids': [c.function_id for c in candidates]})
            return deepcopy(self.cache[key])
        reservation = backend.context_window + backend.output_per_candidate * len(candidates)
        if self.usage_unknown or self.new_calls >= self.max_calls or self.provider_tokens + reservation > self.cap:
            raise GlobalLimit('Global physical request/counter allowance cannot reserve the next whole context and output cap')
        self.new_calls += 1
        try:
            backend.call_number = self.new_calls - 1
            result = backend.execute(candidates, action, level)
        except Exception:
            self.usage_unknown = True
            self.events.append({'event': 'unknown_usage_abort', 'reservation': reservation, 'key': key})
            raise
        self.provider_tokens += result.total_tokens
        if result.input_tokens > backend.context_window or result.output_tokens > backend.output_per_candidate * len(candidates):
            self.usage_unknown = True
            raise GlobalLimit('Provider violated reserved prompt/output bounds; abort all further inference')
        self.events.append({'event': 'fresh_inference', 'key': key, 'reservation': reservation,
                            'actual_provider_tokens': result.total_tokens, 'cumulative_provider_tokens': self.provider_tokens,
                            'ids': [c.function_id for c in candidates]})
        self.cache[key] = deepcopy(result)
        return result


class SelectedActionBackend:
    """Planner exposes no cache answers, actual future costs or labels."""
    def __init__(self, backend, shared):
        self._backend, self._shared = backend, shared
        self.context_window = backend.context_window
        self.context = backend.context

    def estimate(self, candidates, action, level):
        return self._backend.estimate(candidates, action, level)

    def execute(self, candidates, action, level):
        return self._shared.execute(self._backend, candidates, action, level)


def run(manifest_path, output, live=False):
    if output.exists():
        raise ValueError('Require a new output directory')
    manifest = json.loads(manifest_path.read_text())
    if not 1 <= len(manifest['cases']) <= 3 or len({c['bug'] for c in manifest['cases']}) != len(manifest['cases']):
        raise ValueError('Smoke test permits 1..3 distinct preselected bugs')
    shared = SharedExecution(manifest['global_provider_token_cap'], manifest['max_new_calls'])
    output.mkdir(parents=True)
    write_json(output/'manifest.json', manifest)
    report = {'schema': 1, 'status': 'preparing', 'execution_kind': 'fresh_inference_with_exact_selected_action_cache' if live else 'dry_run_no_inference',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest': manifest,
              'manifest_sha256': digest(manifest), 'code_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'implementation_sha256': digest({p.name:p.read_text() for p in Path('rave_fl').glob('*.py')}),
              'cases': [], 'new_llm_calls': 0, 'fresh_provider_tokens': 0}
    ready = []
    # Validate all input sources without reading any gold label.
    for case in manifest['cases']:
        entry = {'bug':case['bug'], 'arms':{}, 'status':'input_validation'}
        report['cases'].append(entry)
        try:
            candidates, context, provenance = prepare_case(case, manifest['top_per_method'])
            if provenance['source_revision'] != case['buggy_revision']:
                raise ValueError('Buggy revision differs from predeclared manifest')
            _, metadata = union_pool(Path(case['flexfl_repo']), case['bug'], manifest['top_per_method'])
            depths = {m:manifest['top_per_method'] for m in RANKERS}
            entry.update(provenance=provenance, candidate_count=len(candidates), status='ready')
            write_json(output/case['bug']/'input_snapshot.json', {'candidates':[asdict(c) for c in candidates],
                         'context':context, 'provenance':provenance, 'rank_metadata':metadata, 'rank_depths':depths})
            b = OllamaPilotBackend(manifest['model'], context, output_per_candidate=manifest['output_per_candidate'],
                                   seed=manifest['seed'], context_window=manifest['context_window'])
            entry['plan'] = {'estimated_full':sum(b.estimate([c],Action.FULL_VERIFY,2) for c in candidates),
                             'estimated_l1':sum(b.estimate(candidates[n:n+manifest['batch_size']],Action.BATCH_SCREEN,1)
                                                for n in range(0,len(candidates),manifest['batch_size'])),
                             'maximum_next_request_reservation':manifest['context_window']+manifest['output_per_candidate']*manifest['batch_size']}
            ready.append((case, entry, candidates, context, metadata, depths))
        except (ValueError, FileNotFoundError) as exc:
            entry.update(status='unsupported_input', error=str(exc))
    write_json(output/'run.json', report)
    if live:
        for case, entry, candidates, context, metadata, depths in ready:
            entry['status'] = 'running'
            for policy in ['full', *manifest['policies']]:
                backend = OllamaPilotBackend(manifest['model'], context, output_per_candidate=manifest['output_per_candidate'],
                    ledger=output/case['bug']/'provider_calls.jsonl', seed=manifest['seed'], context_window=manifest['context_window'])
                backend.preflight()
                entry.setdefault('model_metadata', backend.model_metadata)
                if backend.model_metadata['digest'] != entry['model_metadata']['digest']:
                    raise ValueError('Model digest changed during paired comparison')
                wrapper = SelectedActionBackend(backend, shared)
                if manifest.get('admission_mode') == 'strict_reservation':
                    specification = json.loads(Path(manifest['reservation_specification']).read_text())
                    wrapper = ReservedBackend(wrapper,
                        manifest['full_reference_budget'] if policy == 'full' else manifest['policy_budget'],
                        specification, backend.model_metadata['digest'],
                        (manifest['model'], 0, manifest['seed'], manifest['context_window'], manifest['output_per_candidate']))
                start_tokens, start_calls = shared.provider_tokens, shared.new_calls
                try:
                    if policy == 'full':
                        engine = AdaptivePilot(candidates, wrapper, PolicyConfig(token_budget=manifest['full_reference_budget'], mode='full'))
                    else:
                        engine = make_engine(candidates, wrapper, manifest['policy_budget'], manifest['max_full'], policy,
                                             metadata, depths, manifest['batch_size'])
                    arm = engine.execute()
                    arm.update(selection_policy=policy, fresh_provider_tokens=shared.provider_tokens-start_tokens,
                               new_llm_calls=shared.new_calls-start_calls, execution_kind='selected_exact_cache_and_fresh')
                    arm['budget_is_estimated_admission'] = manifest.get('admission_mode') != 'strict_reservation'
                    arm['reservation_events'] = getattr(wrapper, 'events', [])
                    arm['all_l1_observed'] = policy == 'full' or arm['screened_count'] == len(candidates)
                    entry['arms'][policy] = arm
                    print(json.dumps({'bug':case['bug'], 'policy':policy, 'accounted_tokens':arm['actual_tokens'],
                                      'fresh_total':shared.provider_tokens, 'calls_total':shared.new_calls}), flush=True)
                except GlobalLimit as exc:
                    entry.update(status='global_limit_stopped', error=str(exc))
                    break
                except Exception as exc:
                    # Preserve failed and partially paid executions. Unknown usage
                    # remains unknown; it is never silently reported as zero cost.
                    entry.update(status='execution_failed', error=f'{type(exc).__name__}: {exc}')
                    write_json(output/'run.json', {**report, 'usage_unknown':shared.usage_unknown,
                               'fresh_provider_tokens':shared.provider_tokens, 'new_llm_calls':shared.new_calls})
                    break
                write_json(output/'run.json', report)
            else:
                entry['status'] = 'complete'
            write_json(output/'execution_events.json', shared.events)
            if shared.usage_unknown:
                break
        # Only now read labels, after all policy choices on all bugs.
        from wp1.flexfl_data import read_ground_truth
        labels = {}
        for case, entry, candidates, *_ in ready:
            truths = read_ground_truth(Path(case['flexfl_repo']), case['bug'])
            labels[case['bug']] = truths
            entry['candidate_pool_recall'] = evaluate([c.function_id for c in candidates], truths, {})['candidate_pool_recall']
            for arm in entry['arms'].values():
                arm['accuracy'] = evaluate(arm['ranked_functions'], truths, arm['states'])
                reference = entry['arms'].get('full')
                if reference and reference['full_verified_count'] == len(candidates):
                    arm['verification_token_savings'] = 1-arm['actual_tokens']/reference['actual_tokens']
        write_json(output/'evaluation_only.json', labels)
    report.update(status='complete_bounded_feasibility' if live else 'dry_run_complete',
                  finished_at_utc=datetime.now(timezone.utc).isoformat(), new_llm_calls=shared.new_calls,
                  fresh_provider_tokens=shared.provider_tokens, usage_unknown=shared.usage_unknown)
    write_json(output/'execution_events.json', shared.events)
    write_json(output/'run.json', report)
    return {'output':str(output), 'fresh_provider_tokens':shared.provider_tokens, 'new_llm_calls':shared.new_calls,
            'cases':[{k:c[k] for k in ['bug','status']} for c in report['cases']]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('configs/rave_independent_smoke.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live',action='store_true',help='Without this flag only prepare and validate inputs')
    a=p.parse_args()
    print(json.dumps(run(a.manifest,a.output,a.live),indent=2))


if __name__=='__main__':
    main()
