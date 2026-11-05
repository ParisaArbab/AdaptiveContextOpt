"""Re-evaluate archived rankings without touching requests or calling a model."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from collections import defaultdict
import statistics
from rave_fl.eval.schema import CodeEntity,GoldLabels,file_hash
from rave_fl.eval.fusion import Fusion
from rave_fl.pilot_types import PilotCandidate,CandidateState,Observation
from rave_fl.pilot import write_json
from rave_fl.graft.evaluation import PROFILES,evaluate_profile


def analyze(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Choose a new archive evaluation output')
    rows=[];inputs=[]
    for path in sorted(source.glob('*/results.json')):
        folder=next(path.parent.glob('instances/*/input.json'));label=folder.parent/'evaluation_only.json'
        data=json.loads(folder.read_text());gold=GoldLabels(**json.loads(label.read_text()))
        entities=[CodeEntity(**e) for e in data['entities']];candidates=[PilotCandidate(**c) for c in data['candidates']]
        arms=json.loads(path.read_text());inputs.extend({'path':str(p),'sha256':file_hash(p)} for p in [path,folder,label])
        prior={'ranked_functions':[c.function_id for c in candidates]}
        for profile in PROFILES:
            for policy,arm,status,tokens in [('prior',prior,'completed',0)]+[(a['policy'],a.get('arm',{'ranked_functions':[]}),a['status'],a['accounted_tokens']) for a in arms]:
                metric=evaluate_profile(arm,entities,gold,'defects4j',profile)['method']
                rows.append({'bug':path.parent.name,'profile':profile,'policy':policy,'fusion':'historical','status':status,'tokens':tokens,'metric':metric})
                if 'states' in arm:
                    states={k:CandidateState(v['level'],Observation(**v['observation']) if v['observation'] else None) for k,v in arm['states'].items()}
                    fusion=Fusion('rrf');order=fusion.order(candidates,fusion.scores(candidates,states))
                    new={'ranked_functions':[c.function_id for c in order]}
                    rows.append({'bug':path.parent.name,'profile':profile,'policy':policy,'fusion':'rrf_frozen','status':status,'tokens':tokens,
                                 'metric':evaluate_profile(new,entities,gold,'defects4j',profile)['method']})
    groups=defaultdict(list)
    common={r['bug'] for r in rows if r['policy']=='full' and r['status']=='completed' and r['metric'] is not None}
    # Require completion and available labels for every historical arm.
    common={b for b in common if all(r['status']=='completed' for r in rows if r['bug']==b)}
    for r in rows:
        if r['metric'] is not None:groups[(r['profile'],r['policy'],r['fusion'])].append(r)
    summary=[]
    for (profile,policy,fusion),group in groups.items():
        for name,selected in [('known_labeled_operational',group),('matched_completed_labeled',[r for r in group if r['bug'] in common])]:
            if not selected:continue
            summary.append({'profile':profile,'policy':policy,'fusion':fusion,'cohort':name,'n':len(selected),
                **{m:statistics.mean(r['metric'][m] if r['status']=='completed' else 0 for r in selected) for m in ['Top-1','Top-5','MRR']}})
    output.mkdir(parents=True)
    write_json(output/'summary.json',{'schema':'graft-evaluation-v1','new_model_calls':0,'rows':summary,'input_files':inputs,
                                     'note':'Reference labels are supplied, not independently certified. RRF reranks frozen observations, not policy decisions.'})
    write_json(output/'per_bug.json',rows)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(analyze(a.source,a.output),indent=2))
