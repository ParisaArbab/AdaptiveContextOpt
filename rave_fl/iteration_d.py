"""Reproduce matched controls and offline diagnostics, without model calls."""
from dataclasses import asdict
import argparse
import gzip
import json
from pathlib import Path
import subprocess

from rave_fl.expanded_pool import union_pool
from rave_fl.pilot import digest, write_json
from rave_fl.pilot_evaluation import evaluate
from rave_fl.replay import ReplayCorpus, oracle_frontier
from rave_fl.selection_controls import cheap_features, make_engine, RANKERS

CONDITIONS = {
    "Original": ("results/rave_fl_pilot/time25_llama31_8b/run.json", "configs/rave_pilot_time25.json", "replay_final"),
    "Masked": ("results/rave_fl_rank_repair/live_masked/run.json", "results/rave_fl_rank_repair/masked_evidence/manifest.json", "replay_masked"),
    "Masked + Numeric": ("results/rave_fl_rank_repair/live_normalized_masked/run.json", "results/rave_fl_rank_repair/normalized_masked_evidence/manifest.json", "replay_normalized_masked")
}


def run_controls(protocol_path, output):
    if output.exists():
        raise ValueError("Use a new output directory; historical artifacts are never overwritten")
    protocol = json.loads(protocol_path.read_text())
    report = {"schema": 1, "execution_kind": "offline_exact_action_replay", "new_llm_calls": 0,
              "fresh_inference_tokens": 0, "independent_bugs": 1, "protocol": protocol,
              "protocol_sha256": digest(protocol), "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "implementation_sha256": digest({p.name: p.read_text() for p in Path('rave_fl').glob('*.py')}),
              "conditions": {}}
    for condition, (run, manifest, historical) in CONDITIONS.items():
        case = json.loads(Path(manifest).read_text())["cases"][0]
        corpus = ReplayCorpus(Path(run), case)
        _, metadata = union_pool(Path(case["flexfl_repo"]), case["bug"], 10)
        depths = {m: 10 for m in RANKERS}
        arms = []
        # No ground-truth read or counterfactual output access during this loop.
        for budget in protocol["budgets"]:
            for cap in protocol["max_full_values"]:
                for policy in protocol["policies"]:
                    a = make_engine(corpus.candidates, corpus.backend(), budget, cap, policy, metadata, depths,
                                    protocol["batch_size"]).execute()
                    a.update(selection_policy=policy, comparison_max_full=cap, execution_kind="offline_exact_action_replay",
                             fresh_inference_tokens=0, new_llm_calls=0, accounted_replayed_tokens=a["actual_tokens"])
                    arms.append(a)
        from wp1.flexfl_data import read_ground_truth
        truths = read_ground_truth(Path(case["flexfl_repo"]), case["bug"])
        for a in arms:
            a["accuracy"] = evaluate(a["ranked_functions"], truths, a["states"])
            a["complete_l1_start"] = a["initial_l1"] is not None
            a["verification_token_savings"] = 1 - a["actual_tokens"] / corpus.references["full"]["actual_tokens"]
        base = corpus.references['fixed_batch']
        from rave_fl.pilot_types import CandidateState, Observation
        status = {i: CandidateState(1, Observation(**s['observation'])) for i, s in base['states'].items()}
        f = cheap_features(corpus.candidates, status, metadata, depths)
        gold_ids = [i for i in base['ranked_functions'] if evaluate([i], truths, {})['MRR'] > 0]
        old = json.loads(Path('results/rave_fl_rank_repair', historical, 'replay.json').read_text())
        diagnostic = []
        utility = next(a for a in arms if a['selection_policy'] == 'v2_utility' and a['config']['token_budget'] == 16000 and a['comparison_max_full'] == 1)
        for ident in gold_ids:
            candidate = corpus.by_id[ident]
            proxy = next(r for r in utility['repair_diagnostics'][0]['options'] if r['function_id'] == ident)
            diagnostic.append({'candidate_id': ident, 'is_gold_offline_only': True, 'prior_rank': candidate.rank,
                               'prior': candidate.prior, 'input_ranks': metadata[ident]['ranks'],
                               'static_features': candidate.features, 'cheap_features': f[ident],
                               'l1': base['states'][ident]['observation'],
                               'full_offline_only': asdict(corpus.observed_full(ident)), 'utility_before_action': proxy,
                               'uncertainty_priority_position': sorted(f, key=lambda i: (-f[i]['reported_uncertainty'], f[i]['current_rank'])).index(ident)+1})
        report['conditions'][condition] = {'bug': case['bug'], 'provenance': corpus.provenance,
            'rank_metadata': metadata, 'rank_depths': depths, 'references': corpus.references, 'policy_runs': arms,
            'gold_diagnostics_offline_only': diagnostic,
            'oracle': oracle_frontier(corpus, truths, 4, protocol['primary_budget']),
            'historical_oracle': old['diagnostic_oracle']}
    rows = [json.loads(line) for line in Path('results/rave_fl_rank_repair/replay_final/action_values.jsonl').read_text().splitlines()]
    by_candidate = {}
    for r in rows:
        by_candidate.setdefault(r['candidate_id'], []).append(r)
    changes = []
    for ident, group in by_candidate.items():
        lo, hi = min(group, key=lambda r:r['targets']['delta_MRR']), max(group, key=lambda r:r['targets']['delta_MRR'])
        if hi['targets']['delta_MRR'] - lo['targets']['delta_MRR'] > 1e-12:
            changes.append({'candidate_id':ident, 'min_delta':lo['targets']['delta_MRR'], 'min_state':lo['already_full_ids'],
                            'max_delta':hi['targets']['delta_MRR'], 'max_state':hi['already_full_ids']})
    report['interactions'] = {'rows':len(rows), 'states':len({r['state_id'] for r in rows}), 'independent_bugs':1,
         'positive':sum(r['targets']['delta_MRR'] > 1e-12 for r in rows),
         'negative':sum(r['targets']['delta_MRR'] < -1e-12 for r in rows),
         'zero':sum(abs(r['targets']['delta_MRR']) <= 1e-12 for r in rows),
         'candidate_marginals_change_with_state': changes,
         'strongest_pair': report['conditions']['Original']['oracle']['strongest_pair_interaction'],
         'training_claim': 'none; conditional rows from one bug are not independent training examples'}
    output.mkdir(parents=True)
    (output/'controls.json.gz').write_bytes(gzip.compress(
        json.dumps(report, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    print(json.dumps({'output': str(output), 'new_llm_calls': 0, 'primary': {
        condition: [{'policy':a['selection_policy'], 'tokens':a['actual_tokens'],
                     'rank':a['accuracy']['first_relevant_rank'], 'full_calls':a['full_verified_count'],
                     'selected':[c['candidate_ids'][0] for c in a['calls'] if c['level']==2]}
                    for a in d['policy_runs'] if a['config']['token_budget']==16000 and a['comparison_max_full']==1]
        for condition,d in report['conditions'].items()}, 'interactions':report['interactions']}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('configs/rave_iteration_d.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run_controls(args.protocol, args.output)


if __name__ == '__main__':
    main()
