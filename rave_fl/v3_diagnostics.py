"""Deterministic Time-25 diagnostics, exact selected-action replay and audits."""
from copy import deepcopy
from dataclasses import asdict
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

from rave_fl.adaptive import fused_score
from rave_fl.budget_reservation import ReservedBackend
from rave_fl.core import Action
from rave_fl.iteration_d import CONDITIONS
from rave_fl.expanded_pool import union_pool
from rave_fl.pilot import digest, write_json
from rave_fl.pilot_types import CandidateState, Observation, PolicyConfig
from rave_fl.pilot_evaluation import evaluate
from rave_fl.rank_repair import RankRepairPilot, RepairConfig
from rave_fl.replay import ReplayCorpus
from rave_fl.selection_controls import SelectionControl, cheap_features, RANKERS


class DecisionAudit:
    """Record every candidate, including nonselected and unaffordable options."""
    def decide(self):
        d = super().decide()
        if all(s.level >= 1 for s in self.status.values()):
            features = cheap_features(self.candidates, self.status, self.audit_metadata, self.audit_depths)
            options = []
            for c in self.candidates:
                i = c.function_id
                estimate = self.backend.estimate([c], Action.FULL_VERIFY, 2)
                visited = self.status[i].level == 2
                affordable = estimate <= self.remaining and estimate <= self.backend.context_window
                proxy = self.ranking_proxy(c, estimate)
                reason = ('already_full' if visited else 'selected' if i in d.ids else
                          'budget_or_context' if not affordable else d.reason if d.action == Action.STOP else 'lower_priority')
                options.append({'function_id':i, 'predicted_tokens':estimate, 'affordable':affordable,
                                'selected':i in d.ids, 'decision':reason, **features[i], **proxy})
            self.option_audit.append({'spent_before':self.spent, 'remaining_before':self.remaining,
                                     'action':d.action.value, 'reason':d.reason, 'options':options})
        return d

    def execute(self):
        result = super().execute()
        result['option_audit'] = self.option_audit
        return result


class AuditedUtility(DecisionAudit, RankRepairPilot):
    pass


class AuditedControl(DecisionAudit, SelectionControl):
    pass


def engine_for(corpus, budget, cap, policy, metadata, admission, reservation):
    backend = corpus.backend()
    if admission == 'strict_reservation':
        backend = ReservedBackend(backend, budget, reservation, corpus.provenance['model_digest'], corpus.settings)
    cfg = PolicyConfig(token_budget=budget, batch_size=4, mode='repair_utility')
    repair = RepairConfig(cap,5)
    if policy == 'v2_utility':
        engine = AuditedUtility(corpus.candidates, backend, cfg, repair)
    else:
        engine = AuditedControl(corpus.candidates, backend, cfg, repair, policy, metadata, {m:10 for m in RANKERS})
    engine.option_audit, engine.audit_metadata, engine.audit_depths = [], metadata, {m:10 for m in RANKERS}
    return engine


def order_scores(candidates, states):
    scores = {c.function_id:fused_score(c,states[c.function_id]) for c in candidates}
    order = sorted(candidates,key=lambda c:(-scores[c.function_id],c.rank))
    return [c.function_id for c in order], scores


def quality(candidates, states, truths):
    order,_=order_scores(candidates,states)
    return evaluate(order,truths,{})


def write_csv(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def recorded_ranker_values(repo, bug):
    """Raw CSV diagnostics. Policies deliberately retain their top-list inputs."""
    from wp1.flexfl_data import flexfl_root
    root=flexfl_root(repo)
    signals,provenance={},{}
    for method in RANKERS:
        path=root/'data/FL_results'/method/'Defects4J'/f'{bug}_method-susps.csv'
        rows=list(csv.DictReader(path.open()));seen=set()
        for row in rows:
            if not row.get('File') or not row.get('Signature'):continue
            ident=row['File'].strip()+'.'+row['Signature'].strip()
            if ident in seen:continue
            seen.add(ident)
            signals.setdefault(ident,{})[method+'_full_list_rank']=len(seen)
            value=row.get('Suspiciousness')
            signals[ident][method+'_recorded_score']=float(value) if value not in (None,'') else None
        provenance[method]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'rows':len(rows),'unique_valid_candidates':len(seen),
                            'has_suspiciousness_column':bool(rows and 'Suspiciousness' in rows[0])}
    return signals,provenance


def run(protocol_path, output):
    if output.exists():raise ValueError('Require a new output directory')
    protocol=json.loads(protocol_path.read_text())
    reservation=json.loads(Path('configs/rave_v3_token_reservation.json').read_text())
    report={'schema':1,'execution_kind':'offline_exact_action_replay','fresh_inference_tokens':0,'new_llm_calls':0,
            'independent_bugs':1,'protocol':protocol,'protocol_sha256':digest(protocol),
            'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'implementation_sha256':digest({p.name:p.read_text() for p in Path('rave_fl').glob('*.py')}),
            'reservation_specification':reservation,'conditions':{}}
    candidates_table,policies_table,actions_table,decisions_table,pairs_table=[],[],[],[],[]
    for condition,(run_path,manifest,_) in CONDITIONS.items():
        case=json.loads(Path(manifest).read_text())['cases'][0]
        corpus=ReplayCorpus(Path(run_path),case)
        _,metadata=union_pool(Path(case['flexfl_repo']),case['bug'],10)
        recorded_signals,ranker_sources=recorded_ranker_values(Path(case['flexfl_repo']),case['bug'])
        arms=[]
        for admission,budget,cap,policy in itertools.product(protocol['admission_modes'],protocol['budgets'],protocol['max_full_values'],protocol['policies']):
            engine=engine_for(corpus,budget,cap,policy,metadata,admission,reservation)
            arm=engine.execute()
            arm.update(selection_policy=policy,admission_mode=admission,comparison_max_full=cap,
                       execution_kind='offline_exact_action_replay',new_llm_calls=0,fresh_inference_tokens=0,
                       strict_reservation=admission=='strict_reservation',
                       budget_is_estimated_admission=admission!='strict_reservation',
                       reservation_events=getattr(engine.backend,'events',[]))
            arms.append(arm)
        # Labels are read only after every action decision in this condition.
        from wp1.flexfl_data import read_ground_truth
        truths=read_ground_truth(Path(case['flexfl_repo']),case['bug'])
        fixed=corpus.references['fixed_batch']
        base_status={i:CandidateState(1,Observation(**s['observation'])) for i,s in fixed['states'].items()}
        features=cheap_features(corpus.candidates,base_status,metadata,{m:10 for m in RANKERS})
        base_accuracy=quality(corpus.candidates,base_status,truths)
        full_order=corpus.references['full']['ranked_functions']
        singleton={}
        for c in corpus.candidates:
            ident=c.function_id;full=corpus.observed_full(ident)
            status=deepcopy(base_status);status[ident]=CandidateState(2,full.observations[0])
            after=quality(corpus.candidates,status,truths)
            singleton[ident]=after['MRR']-base_accuracy['MRR']
            l1=base_status[ident].observation
            row={'condition':condition,'candidate_id':ident,'initial_rank':c.rank,'prior':c.prior,
                 'initial_fusion_score':.1+.5*c.prior, 'SBIR_rank':metadata[ident]['ranks'].get('SBIR'),
                 'Ochiai_rank':metadata[ident]['ranks'].get('Ochiai'),'BoostN_rank':metadata[ident]['ranks'].get('BoostN'),
                 **recorded_signals.get(ident,{}),
                 'missing_rank_is_censored':True,**features[ident],
                 'l1_suspicion':l1.suspicion,'l1_uncertainty':l1.uncertainty,'l1_explanation':l1.reason,
                 'l1_missing_evidence':l1.missing_evidence,'static_features':c.features,
                 'predicted_full_tokens':corpus.backend().estimate([c],Action.FULL_VERIFY,2),
                 'strict_full_reservation':ReservedBackend(corpus.backend(),20000,reservation,corpus.provenance['model_digest'],corpus.settings).estimate([c],Action.FULL_VERIFY,2),
                 'full_suspicion_offline':full.observations[0].suspicion,'full_explanation_offline':full.observations[0].reason,
                 'full_rank_offline':full_order.index(ident)+1,'full_input_tokens_offline':full.input_tokens,
                 'full_output_tokens_offline':full.output_tokens,'full_tokens_offline':full.total_tokens,
                 'singleton_delta_MRR_offline':singleton[ident],'singleton_gold_rank_offline':after['first_relevant_rank'],
                 'is_gold_offline':evaluate([ident],truths,{})['MRR']>0}
            candidates_table.append(row)
        # All pairs, not only prefixes visited by greedy policies. Diagnostic only.
        for a,b in itertools.combinations(corpus.candidates,2):
            status=deepcopy(base_status)
            for c in (a,b):status[c.function_id]=CandidateState(2,corpus.observed_full(c.function_id).observations[0])
            after=quality(corpus.candidates,status,truths)
            gain=after['MRR']-base_accuracy['MRR']
            pairs_table.append({'condition':condition,'first':a.function_id,'second':b.function_id,
                                'first_singleton_delta':singleton[a.function_id],'second_singleton_delta':singleton[b.function_id],
                                'pair_delta_MRR':gain,'interaction_MRR':gain-singleton[a.function_id]-singleton[b.function_id],
                                'pair_gold_rank':after['first_relevant_rank'],'diagnostic_only':True})
        for arm in arms:
            arm['accuracy']=evaluate(arm['ranked_functions'],truths,arm['states'])
            arm['complete_l1_start']=arm['screened_count']==len(corpus.candidates)
            if arm['strict_reservation'] and arm['budget_exceeded']:raise AssertionError('Strict replay exceeded budget')
            key={'condition':condition,'policy':arm['selection_policy'],'admission':arm['admission_mode'],
                 'budget':arm['config']['token_budget'],'max_full':arm['comparison_max_full']}
            policies_table.append({**key,'tokens':arm['actual_tokens'],'predicted_selected_tokens':sum(t['estimated_tokens'] for t in arm['trace'] if t['action']!='STOP'),
                                   'full_calls':arm['full_verified_count'],'rank':arm['accuracy']['first_relevant_rank'],
                                   'MRR':arm['accuracy']['MRR'],'top1':arm['accuracy']['top1'],'top5':arm['accuracy']['top5'],
                                   'complete_l1':arm['complete_l1_start'],'remaining':arm['tokens_remaining'],
                                   'budget_exceeded':arm['budget_exceeded'],'stop':arm['stop_reason']})
            status={c.function_id:CandidateState() for c in corpus.candidates}
            spent=0
            for step,call in enumerate(arm['calls']):
                before,bs=order_scores(corpus.candidates,status);bm=evaluate(before,truths,{})
                for obs in call['observations']:status[obs['function_id']]=CandidateState(call['level'],Observation(**obs))
                after,ss=order_scores(corpus.candidates,status);am=evaluate(after,truths,{})
                cost=call['input_tokens']+call['output_tokens'];spent+=cost
                for ident in call['candidate_ids']:
                    actions_table.append({**key,'step':step,'action':call['action'],'candidate_id':ident,
                                          'rank_before':before.index(ident)+1,'rank_after':after.index(ident)+1,
                                          'score_before':bs[ident],'score_after':ss[ident],
                                          'gold_rank_before_offline':bm['first_relevant_rank'],'gold_rank_after_offline':am['first_relevant_rank'],
                                          'delta_MRR_offline':am['MRR']-bm['MRR'],'predicted_call_tokens':arm['trace'][step]['estimated_tokens'],
                                          'actual_call_tokens_offline':cost,'batch_cost_shared':len(call['candidate_ids'])>1,
                                          'spent_after':spent,'remaining_after':arm['config']['token_budget']-spent})
            for step,state in enumerate(arm['option_audit']):
                for option in state['options']:
                    decisions_table.append({**key,'decision_step':step,'spent_before':state['spent_before'],
                                            'remaining_before':state['remaining_before'],**option})
        report['conditions'][condition]={'provenance':corpus.provenance,'ranker_csv_sources':ranker_sources,
                                        'references':corpus.references,'policy_runs':arms}
    rows=[json.loads(line) for line in Path('results/rave_fl_rank_repair/replay_final/action_values.jsonl').read_text().splitlines()]
    report['conditional_action_profile']={'rows':len(rows),'states':len({r['state_id'] for r in rows}),
        'independent_bugs':1,'positive':sum(r['targets']['delta_MRR']>1e-12 for r in rows),
        'negative':sum(r['targets']['delta_MRR']< -1e-12 for r in rows),'zero':sum(abs(r['targets']['delta_MRR'])<=1e-12 for r in rows)}
    output.mkdir(parents=True)
    for name,rows in [('candidates',candidates_table),('policies',policies_table),('actions',actions_table),('decisions',decisions_table),('pairs',pairs_table)]:
        write_csv(output/(name+'.csv'),rows)
    (output/'diagnostics.json.gz').write_bytes(gzip.compress(json.dumps(report,allow_nan=False).encode(),mtime=0))
    print(json.dumps({'output':str(output),'runs':len(policies_table),'new_llm_calls':0,'conditional_rows':report['conditional_action_profile']}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('configs/rave_v3_protocol.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.protocol,a.output)


if __name__=='__main__':main()
