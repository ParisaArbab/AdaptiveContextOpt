"""English research reports and publication figures from preserved artifacts.

No model calls, result literals in plots, or randomized uncertainty intervals.
"""
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from rave_fl.pilot import write_json
from rave_fl.v3_diagnostics import write_csv

COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#332288','#666666']
NAMES={'fixed_l1':'Fixed L1','highest_suspicion':'Highest L1 suspicion','ranker_disagreement':'Ranker disagreement',
       'ranking_ambiguity':'Ranking uncertainty','v2_utility':'V2 Utility','uncertainty_aware':'Uncertainty-aware'}
COND=['Original','Masked','Masked + Numeric']


def md_table(rows,keys,labels=None):
    def cell(v):
        if v is None or v=='':return 'Unavailable'
        if isinstance(v,float):return f'{v:.5f}'.rstrip('0').rstrip('.')
        if isinstance(v,bool):return 'Yes' if v else 'No'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(labels or keys)+' |','|'+'|'.join(['---']*len(keys))+'|']+
                     ['| '+' | '.join(cell(r.get(k)) for k in keys)+' |' for r in rows])


def short(ident):
    bits=ident.split('(')[0].split('.')
    return '.'.join(bits[-2:])+'('+ident.split('(',1)[1] if '(' in ident else ident


def load_data(diagnostics):
    x=json.loads(gzip.decompress((diagnostics/'diagnostics.json.gz').read_bytes()))
    for name in ['candidates','policies','actions','decisions','pairs']:
        x[name]=list(csv.DictReader((diagnostics/(name+'.csv')).open()))
    # Phase A raw files are absent; parse the preserved contemporaneous README.
    x['phase_a']=[]
    for m in re.finditer(r'\|\s*\*{0,2}([AB]-(?:Full|Selective))\*{0,2}\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}([\d,]+)\*{0,2}\s*\|\s*\*{0,2}(\d+)',Path('README.md').read_text()):
        name,calls,tokens,rank=m.groups();rank=int(rank)
        x['phase_a'].append({'phase':'A','arm':name,'calls':int(calls),'tokens':int(tokens.replace(',','')),'rank':rank,
                             'MRR':1/rank,'top1':int(rank==1),'top5':int(rank<=5),
                             'source':'README.md historical table','model':'Qwen2.5 7B / AIScalar A100','evidence':'Short' if name[0]=='A' else 'Structured',
                             'execution':'Historical summary; raw ledgers unavailable locally'})
    if len(x['phase_a'])!=4:raise ValueError('Cannot extract four Phase A historical rows')
    x['v1']=[]
    for path,arms in [
        ('results/rave_fl_pilot/time25_llama31_8b/run.json',{'full':'Full','rank_threshold':'Rank gate','fixed_batch':'Fixed L1','confidence':'Confidence 60K','adaptive':'Adaptive 60K'}),
        ('results/rave_fl_pilot/time25_llama31_budget16k/run.json',{'confidence':'Confidence 16K','adaptive':'Adaptive 16K'})]:
        run=json.loads(Path(path).read_text())
        for key,label in arms.items():
            a=run['cases'][0]['arms'][key];q=a['accuracy']
            x['v1'].append({'phase':'B','arm':label,'calls':a['call_count'],'tokens':a['actual_tokens'],
                            'rank':q['first_relevant_rank'],'MRR':q['MRR'],'top1':q['top1'],'top5':q['top5'],
                            'source':path,'budget':a['config']['token_budget'],'model':run['model'],'raw':a})
    x['v2']=[]
    for condition,d in x['conditions'].items():
        for name,key in [('Full','full'),('Fixed L1','fixed_batch'),('V2 Utility','v2_utility')]:
            a=d['references'][key] if key!='v2_utility' else next(a for a in d['policy_runs'] if a['selection_policy']==key and a['config']['token_budget']==16000 and a['comparison_max_full']==1 and a['admission_mode']=='historical_estimate')
            q=a['accuracy']
            x['v2'].append({'condition':condition,'arm':name,'tokens':a['actual_tokens'],'rank':q['first_relevant_rank'],
                            'MRR':q['MRR'],'top1':q['top1'],'top5':q['top5'],'calls':a['call_count'],
                            'full_calls':a['full_verified_count'],'raw':a,
                            'source':d['provenance']['source_run'] if key!='v2_utility' else 'V3 exact replay (matches saved V2 Utility)',
                            'model':d['provenance']['model'],'budget':a['config']['token_budget']})
    x['smoke']=json.loads(Path('results/rave_fl_iteration_d/independent_live/run.json').read_text())
    return x


def primary(x,admission='historical_estimate',budget=16000,cap=1):
    return [r for r in x['policies'] if r['admission']==admission and r['budget']==str(budget) and r['max_full']==str(cap)]


def box(ax,xy,width,height,label,color=COLORS[0],fontsize=10,linestyle='solid'):
    rect=FancyBboxPatch(xy,width,height,boxstyle='round,pad=0.012',linewidth=1.2,
                       edgecolor=color,facecolor=color+'12',linestyle=linestyle,transform=ax.transAxes)
    ax.add_patch(rect);ax.text(xy[0]+width/2,xy[1]+height/2,label,ha='center',va='center',
                              transform=ax.transAxes,fontsize=fontsize,linespacing=1.5)


def arrow(ax,start,end):
    ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#555555','lw':1.3})


def figures(x,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,
                         'svg.fonttype':'none','svg.hashsalt':'rave-fl-v3','savefig.dpi':220})
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True);manifest=[]
    def save(fig,name,title,source,condition='Time-25, one development defect',budget='See labeled arms',model='llama3.1:8b, local',note=''):
        fig.suptitle(title,fontsize=15,fontweight='bold',y=.995)
        fig.text(.01,.008,'Verification tokens = provider prompt_eval_count + eval_count; replay incurs zero new inference.',fontsize=8,color='#555555')
        description=f'Source: {source}. Condition: {condition}. Model: {model}. Budget: {budget}. Verification tokens: provider input plus output; replay has no fresh inference. {note}'
        fig.savefig(folder/(name+'.png'),bbox_inches='tight',facecolor='white',metadata={'Description':description})
        fig.savefig(folder/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None,'Description':description});plt.close(fig)
        manifest.append({'figure':name,'title':title,'source':source,'condition':condition,'model':model,'budget':budget,
                         'token_definition':'Verification-stage input + output provider counters; replay uses accounted historical costs',
                         'note':note,'png':'figures/'+name+'.png','svg':'figures/'+name+'.svg'})
    # 1. Timeline from documented commits, not reconstructed run dates.
    fig,ax=plt.subplots(figsize=(16,5.3));ax.axis('off')
    results=[f"B-Full: {x['phase_a'][2]['tokens']:,} tokens; rank {x['phase_a'][2]['rank']}",
             f"Fixed L1: {x['v1'][2]['tokens']:,}; rank {x['v1'][2]['rank']}",
             f"Utility: {x['v2'][2]['tokens']:,}; rank {x['v2'][2]['rank']}",f"{len(x['policies'])} replay arms; no new inference"]
    text=[('A | Initial baseline','21 Oct: 2342adc–0f44096','Full vs fixed 0.60 gate\nShort vs structured evidence','Tail excluded by gate'),
          ('B | Adaptive pilot V1','22 Oct: 7aaa72a','L0 / L1 / Full feedback\nCost and confidence policies','Repeated inspection; weak ranking'),
          ('C | L1-first repair V2','22 Oct: 635a316','Global L1 then selective Full\nFrozen shared fusion ranker','Masked evidence collapses'),
          ('D | V3 diagnostic evaluation','Current working changes','Simple controls + action replay\nStrict reservations + audits','No advanced-policy superiority')]
    for i,(title,date,method,limit) in enumerate(text):
        left=.012+i*.25
        box(ax,(left,.16),.225,.70,f'{title}\n\n{date}\n\n{method}\n\n{results[i]}\n\nLimit: {limit}',COLORS[i],9)
        if i<3:arrow(ax,(left+.226,.51),(left+.246,.51))
    fig.subplots_adjust(top=.88,bottom=.13)
    save(fig,'01_research_timeline','Research evolution: changes, observations, and limitations','Git log; tables/historical_metrics.csv; tables/v2_metrics.csv',model='Separate Qwen/A100 and Llama/local generations',note='Commit dates are code/document dates; undocumented inference dates are not invented.')
    # 2. Independent model/environment panels.
    fig,axs=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1,1.6]})
    rows=x['phase_a'];axs[0].barh([r['arm'] for r in rows],[r['tokens'] for r in rows],color=COLORS[:4]);axs[0].invert_yaxis();axs[0].set_title('Qwen2.5 7B / AIScalar A100\nHistorical summaries')
    rows=x['v1']+[{'arm':'V2 Utility',**{k:x['v2'][2][k] for k in ['tokens']}}]
    axs[1].barh([r['arm'] for r in rows],[r['tokens'] for r in rows],color=COLORS);axs[1].invert_yaxis();axs[1].set_title('Llama3.1 8B / local\nOriginal structured evidence')
    for ax in axs:
        ax.set_xlabel('Verification tokens')
        for p in ax.patches:ax.text(p.get_width()+300,p.get_y()+p.get_height()/2,f'{int(p.get_width()):,}',va='center',fontsize=9)
        ax.margins(x=.22)
    fig.tight_layout(rect=[0,.04,1,.92]);save(fig,'02_verification_tokens','Measured verification token consumption','Historical README; V1 run.json; original V2 ledgers',model='Separated by model/environment')
    # 3. Single-defect indicators, not estimated population accuracy.
    fig,axs=plt.subplots(2,4,figsize=(19,8))
    for row,(data,title) in enumerate([(x['phase_a'],'Qwen / A100'),(x['v1']+[dict(x['v2'][2],arm='V2 Utility')],'Llama / local')]):
        for j,(key,label) in enumerate([('rank','Gold rank (lower is better)'),('top1','Top-1 indicator'),('top5','Top-5 indicator'),('MRR','Reciprocal gold rank / MRR')]):
            ax=axs[row,j];ax.barh([r['arm'] for r in data],[r[key] for r in data],color=COLORS[:len(data)]);ax.invert_yaxis();ax.set_title(f'{title}\n{label}')
            if key!='rank':ax.set_xlim(0,1.12)
            for p,r in zip(ax.patches,data):ax.text(p.get_width()+(.02 if key!='rank' else .3),p.get_y()+p.get_height()/2,f'{float(r[key]):.3g}',va='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.91]);save(fig,'03_fault_ranking','Fault ranking on Time-25: one defect, no confidence intervals','tables/historical_metrics.csv; tables/v2_metrics.csv',model='Separate Qwen and Llama panels',note='Top-1/5 are per-defect indicators, not population accuracy estimates.')
    # 4. Pareto frontiers are computed within compatible generations only.
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    for ax,data,title in [(axs[0],x['phase_a'],'Qwen / A100 (two evidence conditions)'),(axs[1],x['v1']+[dict(x['v2'][2],arm='V2 Utility')],'Llama / local (original evidence)')]:
        for n,r in enumerate(data):
            ax.scatter(r['tokens'],r['MRR'],color=COLORS[n%8],s=65)
            offset={'Rank gate':(-65,25),'Adaptive 16K':(20,25),'Confidence 16K':(28,-14),
                    'Adaptive 60K':(5,-14)}.get(r['arm'],(5,7))
            ax.annotate(r['arm'],(r['tokens'],r['MRR']),xytext=offset,textcoords='offset points',fontsize=8,
                        arrowprops={'arrowstyle':'-','color':'#777777','lw':.6} if abs(offset[0])>10 else None)
        if ax==axs[1]:
            frontier=[r for r in data if not any(o['tokens']<=r['tokens'] and o['MRR']>=r['MRR'] and (o['tokens']<r['tokens'] or o['MRR']>r['MRR']) for o in data)]
            frontier=sorted(frontier,key=lambda r:r['tokens']);ax.plot([r['tokens'] for r in frontier],[r['MRR'] for r in frontier],ls='--',color='#555555',label='Observed Pareto points');ax.legend(fontsize=8)
        ax.set(title=title,xlabel='Verification tokens',ylabel='MRR / reciprocal gold rank',ylim=(0,1.15));ax.margins(x=.16)
    fig.tight_layout(rect=[0,.04,1,.92]);save(fig,'04_cost_quality','Tokens versus ranking quality within experimental generations','tables/historical_metrics.csv; tables/v2_metrics.csv',model='Two environments; no cross-environment frontier',note='V1 budgets and V2 architecture differ; plotted comparisons are observational. No Qwen evidence-mixed frontier is claimed.')
    # 5. Evidence sensitivity.
    fig,axs=plt.subplots(1,2,figsize=(14,5.8));width=.23;xx=np.arange(3)
    for i,label in enumerate(['Full','Fixed L1','V2 Utility']):
        rows=[next(r for r in x['v2'] if r['condition']==c and r['arm']==label) for c in COND]
        for ax,key in zip(axs,['tokens','rank']):
            bars=ax.bar(xx+(i-1)*width,[r[key] for r in rows],width,color=COLORS[i],label=label)
            for b,r in zip(bars,rows):ax.text(b.get_x()+width/2,b.get_height()+(.45 if key=='rank' else 350),f'{r[key]:,}',ha='center',fontsize=8)
    for ax in axs:ax.set_xticks(xx,['Original','Masked','Masked + Numeric\n(post-hoc)']);ax.legend(fontsize=8);ax.margins(y=.2)
    axs[0].set_ylabel('Verification tokens');axs[1].set_ylabel('Gold rank (lower is better)')
    fig.tight_layout(rect=[0,.04,1,.93]);save(fig,'05_evidence_conditions','Evidence sensitivity: saved measured references and V2 outcomes','tables/v2_metrics.csv; raw ledgers for all three conditions',condition='Time-25: original / masked / post-hoc numeric',budget='Full 60K; repair 16K')
    # 6. Stage cost, no unsupported attribution to unnecessary reasoning.
    data=x['v1']+[dict(x['v2'][2],arm='V2 Utility')]
    stage=[]
    fig,ax=plt.subplots(figsize=(13,6));bottom=np.zeros(len(data))
    for level,label,color in [(0,'L0 discovery',COLORS[5]),(1,'L1 batching / expansion',COLORS[0]),(2,'Full inspection',COLORS[1])]:
        values=[sum(c['input_tokens']+c['output_tokens'] for c in r['raw']['calls'] if c['level']==level) for r in data]
        ax.barh([r['arm'] for r in data],values,left=bottom,color=color,label=label);bottom+=values
        for r,value in zip(data,values):stage.append({'arm':r['arm'],'stage':label,'tokens':value})
    ax.invert_yaxis();ax.set_xlabel('All paid input + output tokens, grouped by action level');ax.legend(loc='lower right');fig.tight_layout(rect=[0,.04,1,.93]);write_csv(out/'tables/stage_tokens.csv',stage)
    save(fig,'06_stage_costs','Stage-by-stage verification costs','V1/V2 calls and action levels; tables/stage_tokens.csv',note='L1 includes individual expansion in V1. Repeated context is charged. The schema cannot isolate which tokens were unnecessary.')
    # 7. All 24 candidates, offline gold annotation only.
    fig,axs=plt.subplots(1,2,figsize=(14,12))
    for panel,(ax,condition) in enumerate(zip(axs,['Original','Masked'])):
        rows=sorted([r for r in x['candidates'] if r['condition']==condition],key=lambda r:int(r['initial_rank']))
        arm=next(a for a in x['conditions'][condition]['policy_runs'] if a['selection_policy']=='v2_utility' and a['config']['token_budget']==16000 and a['comparison_max_full']==1 and a['admission_mode']=='historical_estimate')
        values=np.array([[int(r['initial_rank']),int(r['current_rank']),arm['ranked_functions'].index(r['candidate_id'])+1,int(r['full_rank_offline'])] for r in rows])
        im=ax.imshow(values,cmap='viridis_r',vmin=1,vmax=len(rows),aspect='auto')
        labels=[f"C{int(r['initial_rank']):02d} "+short(r['candidate_id']).split('(')[0] for r in rows] if panel==0 else [f"C{int(r['initial_rank']):02d}" for r in rows]
        ax.set_yticks(range(len(rows)),labels,fontsize=9)
        ax.set_xticks(range(4),['Initial','All L1','After Utility','Full-all\nreference']);ax.set_title(condition)
        for n,r in enumerate(rows):
            if r['is_gold_offline']=='True':ax.add_patch(Rectangle((-.5,n-.5),4,1,fill=False,ec=COLORS[1],lw=2.5))
            if arm['states'][r['candidate_id']]['level']==2:ax.add_patch(Rectangle((1.5,n-.5),1,1,fill=False,ec='#FFFFFF',lw=3))
            for j in range(4):ax.text(j,n,str(values[n,j]),ha='center',va='center',fontsize=8,color='white' if values[n,j]>13 else 'black')
    fig.subplots_adjust(wspace=.20,left=.32,right=.89,top=.91,bottom=.08)
    cax=fig.add_axes([.93,.25,.015,.48]);fig.colorbar(im,cax=cax,label='Candidate rank (1 is best)')
    save(fig,'07_ranking_evolution','All candidates: ranking evolution and mixed-evidence state','tables/candidates.csv; original/masked V2 Utility states',budget='Historical estimated admission 16K, at most one Full',note='Orange outline: gold (offline only). White cell outline: Full-upgraded candidate. Other Utility cells remain L1.')
    # 8. Input features and retrospective labels explicitly separated.
    rows=sorted([r for r in x['candidates'] if r['condition']=='Original'],key=lambda r:int(r['initial_rank']))
    arm=next(a for a in x['conditions']['Original']['policy_runs'] if a['selection_policy']=='v2_utility' and a['config']['token_budget']==16000 and a['comparison_max_full']==1 and a['admission_mode']=='historical_estimate')
    opts={o['function_id']:o for o in arm['option_audit'][0]['options']}
    fig,axs=plt.subplots(1,3,figsize=(14,9),sharey=True);yy=np.arange(len(rows));hh=.33
    axs[0].barh(yy,[opts[r['candidate_id']]['gain_proxy'] for r in rows],color=COLORS[0]);axs[0].set_xlabel('V2 gain proxy\nAvailable before Full')
    from matplotlib.ticker import MaxNLocator
    axs[0].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axs[1].barh(yy-hh/2,[int(r['predicted_full_tokens']) for r in rows],hh,label='Predicted (online)',color=COLORS[0]);axs[1].barh(yy+hh/2,[int(r['full_tokens_offline']) for r in rows],hh,label='Actual (offline)',color=COLORS[1]);axs[1].legend(fontsize=8);axs[1].set_xlabel('Marginal Full provider tokens')
    gains=[float(r['singleton_delta_MRR_offline']) for r in rows];axs[2].barh(yy,gains,color=[COLORS[2] if g>=0 else COLORS[4] for g in gains]);axs[2].set_xlabel('Observed singleton change in MRR\nRetrospective diagnostic label')
    axs[0].set_yticks(yy,[f"C{int(r['initial_rank']):02d} {short(r['candidate_id'])}" for r in rows],fontsize=8);axs[0].invert_yaxis()
    fig.tight_layout(rect=[0,.04,1,.93]);save(fig,'08_action_value_cost','Action utility proxies, costs, and realized ranking effects','tables/candidates.csv; tables/decisions.csv',condition='Original Time-25, all-L1 starting state',budget='16K historical admission; outcomes offline')
    # 9. Exact controller trace.
    fig,axs=plt.subplots(1,2,figsize=(15,7))
    for ax,condition in zip(axs,['Original','Masked']):
        ax.axis('off');a=next(a for a in x['conditions'][condition]['policy_runs'] if a['selection_policy']=='v2_utility' and a['config']['token_budget']==16000 and a['comparison_max_full']==1 and a['admission_mode']=='historical_estimate')
        base=x['conditions'][condition]['references']['fixed_batch'];selected=[c for c in a['calls'] if c['level']==2]
        box(ax,(.12,.72),.76,.17,f"{condition}: six fixed L1 batches\n{base['actual_tokens']:,} tokens; gold rank {base['accuracy']['first_relevant_rank']} (offline)",COLORS[0])
        if selected:
            call=selected[0];ident=call['candidate_ids'][0];opt=next(o for o in a['option_audit'][0]['options'] if o['function_id']==ident)
            text=f"Selected: {short(ident)}\nProxy gain {opt['gain_proxy']:.5f}; predicted cost {opt['predicted_tokens']:,}\nActual cost {call['input_tokens']+call['output_tokens']:,} (after selection)"
        else:text='All affordable proxy gains are nonpositive\nGold projection stays outside top-5 (offline identification)\nNo Full observation purchased'
        box(ax,(.06,.40),.88,.22,text,COLORS[1],9);arrow(ax,(.5,.71),(.5,.63))
        box(ax,(.12,.10),.76,.18,f"STOP: {a['stop_reason']}\n{a['actual_tokens']:,} tokens; final rank {a['accuracy']['first_relevant_rank']} (offline)",COLORS[2],10);arrow(ax,(.5,.39),(.5,.29))
    fig.subplots_adjust(top=.91,bottom=.08);save(fig,'09_decision_traces','V2 Utility decision traces: successful repair versus missed opportunity','Exact replay option_audit and trace; tables/decisions.csv',budget='16K historical estimate admission; one-Full control',note='The original recorded V2 cap-four run stops on no-positive-proxy after the same action; the cap-one control shown stops at its call limit.')
    # 10. Explicit completed / proposed architecture.
    fig,ax=plt.subplots(figsize=(16,7));ax.axis('off')
    nodes=[(.015,.65,.17,.17,'Existing rankers\nSBIR / Ochiai / BoostN'),(.225,.65,.15,.17,'Candidate union\nOfficial buggy spans'),(.415,.65,.16,.17,'Global L1 batching\nExact request ledgers'),(.615,.65,.17,.17,'Cheap features\nRanks / margins / support'),
           (.615,.32,.17,.17,'Controller + budget\nSelect / reject / stop'),(.415,.32,.16,.17,'Independent Full\nSelected action only'),(.225,.32,.15,.17,'Frozen shared ranker\nPrior + evidence'),(.015,.32,.17,.17,'Final ranking\nAll candidates retained')]
    for l,b,w,h,label in nodes:box(ax,(l,b),w,h,label,COLORS[0],10)
    for start,end in [((.185,.735),(.223,.735)),((.375,.735),(.413,.735)),((.575,.735),(.613,.735)),((.70,.65),(.70,.49)),((.615,.405),(.575,.405)),((.415,.405),(.375,.405)),((.225,.405),(.185,.405))]:arrow(ax,start,end)
    box(ax,(.825,.32),.16,.49,'Future research\n\nIndependent-bug\ninteraction / utility model\n\nLightGBM training +\nheld-out calibration',COLORS[7],9,'dashed')
    box(ax,(.225,.035),.56,.14,'Offline only: gold metrics, marginal action labels, oracle subsets\nNo backward path into inference-time features',COLORS[4],10)
    ax.plot([.30,.30,.80,.80,.70],[.49,.55,.55,.91,.91],transform=ax.transAxes,
            color='#666666',lw=1.1,ls='--')
    arrow(ax,(.70,.91),(.70,.82))
    ax.text(.54,.57,'Update scores; recompute after each Full',transform=ax.transAxes,fontsize=9,ha='center',color='#555555')
    arrow(ax,(.30,.32),(.30,.18));fig.subplots_adjust(top=.9,bottom=.08)
    save(fig,'10_implementation_architecture','Current implementation and proposed learned-policy extension','rave_fl/{expanded_pool,code_sketch,selection_controls,rank_repair,replay,adaptive,train_risk}.py',condition='Architecture, not an empirical result',budget='Reserved admission added in V3',note='Solid blue modules implemented. Dashed gray trained policy remains unvalidated. Training utilities exist; no real learned artifact is claimed.')
    # 11. Testing progress uses actual logs and commit milestones.
    tests=json.loads((out/'provenance/test_runs.json').read_text()) if (out/'provenance/test_runs.json').exists() else {'runs':[]}
    audits=json.loads((out/'provenance/artifact_audits.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(15,6));axs[0].axis('off')
    milestones='Verified code history\n\n21 Oct | 9e27fac: evidence regression tests\n22 Oct | 7aaa72a: adaptive + risk plumbing tests\n22 Oct | 635a316: replay + evidence ablations\nV3 | prompt-identity fix + reservation / control tests\n\nCurrent executions\n'+ '\n'.join(r['label']+': '+r['summary'] for r in tests['runs'])
    box(axs[0],(.02,.1),.95,.82,milestones,COLORS[0],10)
    counts=[v['total_paid_calls'] for v in audits['paired_runs'].values()]+[audits['smoke']['new_calls']]
    labels=['V1 initial','V1 final','V2 original','V2 masked','V2 numeric','Saved smoke']
    axs[1].barh(labels,counts,color=COLORS[:len(counts)]);axs[1].invert_yaxis();axs[1].set_xlabel('Raw provider request/response pairs audited')
    for i,n in enumerate(counts):axs[1].text(n+1,i,str(n),va='center')
    axs[1].set_xlim(0,max(counts)*1.15);fig.tight_layout(rect=[0,.04,1,.92]);save(fig,'11_validation_progress','Development milestones and current verification evidence','Git commits; provenance/test_runs.json; provenance/artifact_audits.json',condition='Audit coverage and current tests',model='No new model calls',budget='No inference',note='Historical tests are not presented as newly added. Known unrelated failures are retained.')
    # 12. Strict reservation results explicitly separate incomplete L1.
    fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for row,mode in enumerate(['historical_estimate','strict_reservation']):
        for col,condition in enumerate(COND):
            ax=axs[row,col];vals=[];complete=[]
            for policy in NAMES:
                rr=[next(r for r in x['policies'] if r['condition']==condition and r['policy']==policy and r['budget']==str(b) and r['max_full']=='1' and r['admission']==mode) for b in x['protocol']['budgets']]
                vals.append([int(r['rank']) for r in rr]);complete.append([r['complete_l1']=='True' for r in rr])
            ax.imshow(vals,vmin=1,vmax=24,cmap='viridis_r',aspect='auto');ax.set_title(condition+'\n'+mode.replace('_',' '))
            ax.set_xticks(range(4),['14K','16K','18K','20K']);ax.set_yticks(range(6),list(NAMES.values()),fontsize=9)
            for i in range(6):
                for j in range(4):ax.text(j,i,str(vals[i][j])+('†' if not complete[i][j] else ''),ha='center',va='center',color='white' if vals[i][j]>13 else 'black',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.92]);save(fig,'12_budget_admission','Gold ranks under matched call caps and different admission guarantees','tables/policies.csv',condition='Time-25 three evidence variants; conditional replay',budget='14K/16K/18K/20K; at most one Full',note='Dagger marks incomplete L1 coverage; those cells are not successful L1-first comparisons. Strict bounds rely on the pinned Llama 3 chat/tokenizer protocol.')
    # 13. Audited saved smoke, clearly not fresh V3 runs.
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    for ax,entry in zip(axs,[c for c in x['smoke']['cases'] if c['status']=='complete']):
        labels=['Full' if n=='full' else NAMES[n] for n in entry['arms']];values=[a['actual_tokens'] for a in entry['arms'].values()]
        bars=ax.barh(labels,values,color=COLORS[:len(values)]);ax.invert_yaxis();ax.set_title(entry['bug']+' | saved feasibility run');ax.set_xlabel('Accounted verification tokens')
        for b,a in zip(bars,entry['arms'].values()):ax.text(b.get_width()+100,b.get_y()+b.get_height()/2,f"rank {a['accuracy']['first_relevant_rank']}",va='center',fontsize=9)
        ax.margins(x=.26)
    fig.tight_layout(rect=[0,.04,1,.92]);save(fig,'13_saved_independent_smoke','Saved independent-bug feasibility: no selective ranking gain','results/rave_fl_iteration_d/independent_live/run.json + audited provider ledgers',condition='Time-1 and Lang-1; Math-1 rejected (span mismatch)',budget='Policies 20K; Full 60K; global physical cap 80K',note='Pre-existing uncommitted results audited in V3. Trigger text reused; tests not rerun locally. Different L1 pool size: 25. No new heuristic result is claimed here.')
    write_csv(out/'tables/figure_manifest.csv',manifest)
    return manifest


def history_table():
    commits=['2342adc','67c4f7e','9e27fac','50651f8','0f44096','7aaa72a','635a316']
    rows=[]
    for c in commits:
        data=subprocess.check_output(['git','show','-s','--format=%h%x09%aI%x09%s',c],text=True).strip().split('\t')
        files=subprocess.check_output(['git','diff-tree','--no-commit-id','--name-only','-r',c],text=True).splitlines()
        rows.append({'commit':data[0],'author_timestamp':data[1],'subject':data[2],
                     'source_files':[f for f in files if not f.startswith('results/')],'result_files_changed':sum(f.startswith('results/') for f in files)})
    return rows


def tables(x,out):
    folder=out/'tables';folder.mkdir(parents=True,exist_ok=True)
    historical=[{k:v for k,v in r.items() if k!='raw'} for r in x['phase_a']+x['v1']]
    v2=[{k:v for k,v in r.items() if k!='raw'} for r in x['v2']]
    write_csv(folder/'historical_metrics.csv',historical);write_csv(folder/'v2_metrics.csv',v2)
    for n in ['candidates','policies','actions','decisions','pairs']:write_csv(folder/(n+'.csv'),x[n])
    write_json(folder/'git_timeline.json',history_table())
    primary_rows=primary(x)
    (folder/'primary_comparison.md').write_text(md_table(primary_rows,['condition','policy','tokens','rank','MRR','full_calls','remaining','stop'])+'\n')
    rows=[]
    for c in x['smoke']['cases']:
        for policy,a in c.get('arms',{}).items():rows.append({'bug':c['bug'],'policy':policy,'tokens':a['actual_tokens'],
              'rank':a['accuracy']['first_relevant_rank'],'MRR':a['accuracy']['MRR'],'full_calls':a['full_verified_count'],
              'fresh_policy_calls':a['new_llm_calls'],'fresh_policy_tokens':a['fresh_provider_tokens']})
    write_csv(folder/'saved_smoke_metrics.csv',rows)
    return historical,v2,rows


def reports(x,out,manifest,historical,v2,smoke):
    replacements={
        '{{HISTORICAL}}':md_table(historical,['phase','arm','tokens','rank','MRR','source']),
        '{{V2}}':md_table(v2,['condition','arm','tokens','rank','MRR','full_calls']),
        '{{PRIMARY}}':md_table(primary(x),['condition','policy','tokens','rank','MRR','full_calls']),
        '{{STRICT}}':md_table(primary(x,'strict_reservation'),['condition','policy','tokens','rank','MRR','full_calls','complete_l1']),
        '{{SMOKE}}':md_table(smoke,['bug','policy','tokens','rank','MRR','full_calls']),
        '{{GIT}}':md_table(history_table(),['commit','author_timestamp','subject']),
        '{{FIGURES}}':'\n\n'.join(f"![{r['title']}]({r['png']})\n\n**Figure {n+1}. {r['title']}.** Source: `{r['source']}`. Condition: {r['condition']}. Model: {r['model']}. Budget: {r['budget']}. {r['token_definition']}. {r['note']} [SVG]({r['svg']})." for n,r in enumerate(manifest))
    }
    # Candidate-level compact tables for all candidates, each condition.
    candidate_sections=[]
    for condition in COND:
        rows=sorted([r for r in x['candidates'] if r['condition']==condition],key=lambda r:int(r['initial_rank']))
        for r in rows:r['short_id']=f"C{int(r['initial_rank']):02d} "+short(r['candidate_id'])
        candidate_sections.append('### '+condition+'\n\n'+md_table(rows,['short_id','initial_rank','current_rank','l1_suspicion','l1_uncertainty','current_score','full_rank_offline','full_suspicion_offline','predicted_full_tokens','full_tokens_offline']))
    replacements['{{CANDIDATES}}']='\n\n'.join(candidate_sections)
    goldrows=[];proxyrows=[]
    for condition,d in x['conditions'].items():
        g=next(r for r in x['candidates'] if r['condition']==condition and r['is_gold_offline']=='True')
        arm=next(a for a in d['policy_runs'] if a['selection_policy']=='v2_utility' and a['admission_mode']=='historical_estimate' and a['config']['token_budget']==16000 and a['comparison_max_full']==1)
        o=next(o for o in arm['option_audit'][0]['options'] if o['function_id']==g['candidate_id'])
        proxyrows.append({'condition':condition,'gain_proxy':o['gain_proxy'],'utility_per_token':o['utility_per_token'],
                          'projected_rank':o.get('projected_rank'),'predicted_tokens':o['predicted_tokens'],
                          'actual_tokens_offline':g['full_tokens_offline'],'strict_reservation':g['strict_full_reservation']})
        goldrows.append({**g,'input_ranks':'SBIR '+g['SBIR_rank']+', Ochiai absent, BoostN '+g['BoostN_rank']})
    replacements['{{GOLD}}']=md_table(goldrows,['condition','initial_rank','input_ranks','current_rank','l1_suspicion','l1_uncertainty','current_score','gap_above','gap_below','disagreement','evidence_support','full_rank_offline'])
    replacements['{{RANKSIGNALS}}']=md_table(goldrows,['condition','SBIR_full_list_rank','SBIR_recorded_score','Ochiai_full_list_rank','Ochiai_recorded_score','BoostN_full_list_rank','BoostN_recorded_score'])
    replacements['{{PROXY}}']=md_table(proxyrows,['condition','gain_proxy','utility_per_token','projected_rank','predicted_tokens','actual_tokens_offline','strict_reservation'])
    response_rows=[]
    for r in goldrows:response_rows.append({k:r[k] for k in ['condition','l1_explanation','full_explanation_offline']})
    replacements['{{RESPONSES}}']=md_table(response_rows,['condition','l1_explanation','full_explanation_offline'])
    changes=[]
    for condition in COND:
        rows=[r for r in x['candidates'] if r['condition']==condition];g=next(r for r in rows if r['is_gold_offline']=='True');gs=float(g['full_suspicion_offline'])
        changes.append({'condition':condition,'full_scores_above_gold':sum(float(r['full_suspicion_offline'])>gs for r in rows),
                        'equal_score_candidates_ahead_by_prior':sum(float(r['full_suspicion_offline'])==gs and int(r['initial_rank'])<int(g['initial_rank']) for r in rows),
                        'gold_full_score':gs,'l1_top2_margin':float(sorted(rows,key=lambda r:int(r['current_rank']))[0]['current_score'])-float(sorted(rows,key=lambda r:int(r['current_rank']))[1]['current_score'])})
    replacements['{{TIES}}']=md_table(changes,list(changes[0]))
    pairrows=[r for r in x['pairs'] if r['condition']=='Original'];pair=max(pairrows,key=lambda r:abs(float(r['interaction_MRR'])))
    both_positive=sum(float(r['first_singleton_delta'])>1e-12 and float(r['second_singleton_delta'])>1e-12 and float(r['pair_delta_MRR'])< -1e-12 for r in pairrows)
    replacements['{{PAIR}}']=md_table([pair],['first','second','first_singleton_delta','second_singleton_delta','pair_delta_MRR','interaction_MRR','pair_gold_rank'])
    p=x['conditional_action_profile'];replacements['{{ACTION_PROFILE}}']=f"{p['rows']} conditional action rows across {p['states']} partial states: {p['positive']} positive, {p['negative']} negative, {p['zero']} zero marginal MRR changes. All rows belong to one defect."
    replacements['{{BOTH_POSITIVE}}']=str(both_positive)
    # Exact observed changes under numeric augmentation, without attributing thought processes.
    masked={r['candidate_id']:r for r in x['candidates'] if r['condition']=='Masked'}
    numeric={r['candidate_id']:r for r in x['candidates'] if r['condition']=='Masked + Numeric'}
    nr={'l1_suspicion_changed':sum(masked[i]['l1_suspicion']!=numeric[i]['l1_suspicion'] for i in masked),
        'l1_uncertainty_changed':sum(masked[i]['l1_uncertainty']!=numeric[i]['l1_uncertainty'] for i in masked),
        'l1_fused_score_changed':sum(masked[i]['current_score']!=numeric[i]['current_score'] for i in masked),
        'full_suspicion_changed':sum(masked[i]['full_suspicion_offline']!=numeric[i]['full_suspicion_offline'] for i in masked)}
    replacements['{{NUMERIC_CHANGES}}']=md_table([nr],list(nr))
    tests=json.loads((out/'provenance/test_runs.json').read_text())
    replacements['{{TESTS}}']=md_table(tests['runs'],['label','command','summary','returncode'])
    for name in ['RAVE_FL_Research_Progress_Report','RAVE_FL_V2_Diagnostic_Analysis','RAVE_FL_V3_Experiment_Results','RAVE_FL_Reproducibility']:
        text=Path('reports/templates',name+'.md').read_text()
        for key,value in replacements.items():text=text.replace(key,value)
        unresolved=re.findall(r'\{\{[^}]+\}\}',text)
        if unresolved:raise ValueError('Unresolved report placeholders: '+str(unresolved))
        (out/(name+'.md')).write_text(text+'\n')


def generate(diagnostics,out):
    out.mkdir(parents=True,exist_ok=True);x=load_data(diagnostics)
    historical,v2,smoke=tables(x,out);manifest=figures(x,out);reports(x,out,manifest,historical,v2,smoke)
    paths=[p for folder in ['tables','figures'] for p in (out/folder).glob('*')]+list(out.glob('RAVE_FL_*.md'))
    write_json(out/'provenance/generated_sha256.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    print(json.dumps({'reports':str(out),'figures':len(manifest),'diagnostics':str(diagnostics),'new_llm_calls':0}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostics',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('reports'));a=p.parse_args();generate(a.diagnostics,a.output)


if __name__=='__main__':main()
