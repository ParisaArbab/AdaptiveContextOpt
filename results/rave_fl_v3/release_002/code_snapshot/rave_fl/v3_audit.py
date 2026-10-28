"""Audit historical paired runs and pre-existing independent smoke ledgers."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from rave_fl.adaptive import fused_score
from rave_fl.audit_pilot import audit, require
from rave_fl.core import Action
from rave_fl.pilot import digest, prepare_case, write_json
from rave_fl.pilot_backend import build_prompt, parse_response
from rave_fl.pilot_evaluation import evaluate
from rave_fl.pilot_types import CandidateState, Observation

PAIRED = [
 ('results/rave_fl_pilot/time25_llama31_8b/run.json','configs/rave_pilot_time25.json'),
 ('results/rave_fl_pilot/time25_llama31_budget16k/run.json','configs/rave_pilot_time25.json'),
 ('results/rave_fl_rank_repair/live_original/run.json','configs/rave_pilot_time25.json'),
 ('results/rave_fl_rank_repair/live_masked/run.json','results/rave_fl_rank_repair/masked_evidence/manifest.json'),
 ('results/rave_fl_rank_repair/live_normalized_masked/run.json','results/rave_fl_rank_repair/normalized_masked_evidence/manifest.json')]


def audit_smoke(path):
    run=json.loads(path.read_text());manifest=run['manifest'];events=json.loads((path.parent/'execution_events.json').read_text())
    report={'status':'verified_saved_ledgers_with_evidence_limitations','executed_this_task':False,'cases':[],
            'new_calls':0,'provider_tokens':0,'invalid_outputs':0,
            'limitations':['No local Defects4J failing-test execution; saved FlexFL trigger descriptions.',
                           'No complete code snapshot of the uncommitted smoke implementation was saved.',
                           'Per-policy admission was estimated; global physical inference used whole-context reservations.',
                           'Two completed development/feasibility bugs, one unsupported input; not held-out validation.']}
    known={}
    from wp1.flexfl_data import read_ground_truth
    for entry,case in zip(run['cases'],manifest['cases']):
        if entry['status']=='unsupported_input':
            report['cases'].append({'bug':entry['bug'],'status':entry['status'],'error':entry['error']});continue
        cs,context,provenance=prepare_case(case,manifest['top_per_method']);by_id={c.function_id:c for c in cs}
        require(provenance['source_revision']==case['buggy_revision'],'Smoke revision mismatch')
        require(provenance['sources_sha256']==entry['provenance']['sources_sha256'],'Smoke source mismatch')
        require(provenance['evidence']['sha256']==entry['provenance']['evidence']['sha256'],'Smoke evidence mismatch')
        dirty=subprocess.check_output(['git','-C',case['checkout'],'status','--porcelain','--untracked-files=no'],text=True)
        require(not dirty,'Buggy checkout modified')
        ledger=path.parent/case['bug']/'provider_calls.jsonl'
        lines=[json.loads(l) for l in ledger.read_text().splitlines()]
        requests=[e for e in lines if e['event']=='request'];responses=[e for e in lines if e['event']=='response']
        require(len(requests)==len(responses),'Incomplete smoke ledger')
        require(not any(e['event']=='error' for e in lines),'Unknown smoke usage')
        for req,res in zip(requests,responses):
            require(req['call_id']==res['call_id'],'Smoke response pairing')
            require(digest(req['request'])==req['request_sha256'],'Smoke request checksum')
            ids=req['candidate_ids'];action=Action(req['action']);level=req['level'];p=req['request'];raw=res['response']
            system,user=build_prompt([by_id[i] for i in ids],context,action,level)
            require(p['messages']==[{'role':'system','content':system},{'role':'user','content':user}],'Smoke prompt changed')
            require(p['model']==manifest['model'] and p['options']=={'temperature':0,'seed':manifest['seed'],
                    'num_ctx':manifest['context_window'],'num_predict':manifest['output_per_candidate']*len(ids)},'Smoke model settings')
            parsed,error=parse_response(raw['message']['content'],[by_id[i] for i in ids],action)
            require([asdict(o) for o in parsed]==res['parsed']['observations'],'Smoke parse mismatch')
            require(raw['done'] is True,'Incomplete provider output')
            report['invalid_outputs']+=int(error is not None)
            require(raw['prompt_eval_count']<=manifest['context_window'] and raw['eval_count']<=p['options']['num_predict'],'Provider bound violated')
            report['new_calls']+=1;report['provider_tokens']+=raw['prompt_eval_count']+raw['eval_count']
            key=(case['bug'],action.value,level,tuple(ids))
            known[key]=(raw,parsed)
        truths=read_ground_truth(Path(case['flexfl_repo']),case['bug'])
        arm_audit={}
        for name,a in entry['arms'].items():
            total=0;status={c.function_id:CandidateState() for c in cs}
            for call in a['calls']:
                key=(case['bug'],call['action'],call['level'],tuple(call['candidate_ids']))
                require(key in known,'Arm cache entry unavailable')
                raw,obs=known[key]
                require(call['raw_response']==raw['message']['content'] and call['observations']==[asdict(o) for o in obs],'Selected cache mismatch')
                require(call['input_tokens']==raw['prompt_eval_count'] and call['output_tokens']==raw['eval_count'],'Smoke cache accounting')
                total+=call['input_tokens']+call['output_tokens']
                for o in obs:status[o.function_id]=CandidateState(call['level'],o)
            scores={c.function_id:fused_score(c,status[c.function_id]) for c in cs}
            order=sorted(cs,key=lambda c:(-scores[c.function_id],c.rank))
            require([c.function_id for c in order]==a['ranked_functions'] and scores==a['ranking_scores'],'Smoke frozen ranker mismatch')
            require(total==a['actual_tokens'] and not a['budget_exceeded'],'Smoke tokens mismatch or overrun')
            require(evaluate(a['ranked_functions'],truths,a['states'])==a['accuracy'],'Smoke evaluation mismatch')
            arm_audit[name]={'tokens':total,'calls':len(a['calls']),'rank':a['accuracy']['first_relevant_rank']}
        report['cases'].append({'bug':case['bug'],'status':'verified','source_revision':provenance['source_revision'],
                                'ledger_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest(),'arms':arm_audit})
    require(report['new_calls']==run['new_llm_calls'] and report['provider_tokens']==run['fresh_provider_tokens'],'Global smoke accounting mismatch')
    spent=0
    for event in events:
        if event['event']=='fresh_inference':
            require(spent+event['reservation']<=manifest['global_provider_token_cap'],'Global admission violated')
            spent+=event['actual_provider_tokens']
            require(spent==event['cumulative_provider_tokens'],'Global cumulative mismatch')
    require(spent==report['provider_tokens'],'Global event totals differ')
    return report


def run_audits(output):
    out={'paired_runs':{},'smoke':audit_smoke(Path('results/rave_fl_iteration_d/independent_live/run.json'))}
    for path,manifest in PAIRED:out['paired_runs'][path]=audit(Path(path),Path(manifest))
    tracked=subprocess.check_output(['git','ls-files','results/rave_fl_pilot','results/rave_fl_rank_repair','results/rave_fl_phase3'],text=True).splitlines()
    out['historical_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in tracked}
    out['historical_file_count']=len(tracked)
    write_json(output,out)
    return out
