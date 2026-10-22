"""Controls, information boundaries, hard global reservation and frozen fusion."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
from rave_fl.adaptive import fused_score
from rave_fl.core import Action
from rave_fl.independent_smoke import GlobalLimit, SharedExecution
from rave_fl.pilot_types import CandidateState, Observation, CallResult
from rave_fl.selection_controls import make_engine, rank_percentiles, cheap_features
from test_rave_fl_rank_repair import candidates, RecordedBackend, make_disk_corpus
from rave_fl.replay import ReplayCorpus


def inputs():
    cs = candidates()
    meta = {c.function_id:{'ranks':{'SBIR':n+1}} for n,c in enumerate(cs)}
    return cs, meta, {'SBIR':3,'Ochiai':3,'BoostN':3}


def test_censored_disagreement_handles_missing_and_unequal_scales():
    assert rank_percentiles({'SBIR':5,'Ochiai':50}, {'SBIR':9,'Ochiai':99,'BoostN':4}) == [.5,.5,1]
    assert rank_percentiles({}, {'SBIR':9,'Ochiai':99,'BoostN':4}) == [1,1,1]
    with pytest.raises(ValueError):
        rank_percentiles({'SBIR':11}, {'SBIR':9,'Ochiai':99,'BoostN':4})


@pytest.mark.parametrize('policy',['fixed_l1','highest_suspicion','ranker_disagreement','ranking_ambiguity','v2_utility'])
def test_repeat_determinism_original_batches_and_exact_cost(policy):
    cs,meta,depths=inputs()
    def run():return make_engine(cs,RecordedBackend(),1000,1,policy,meta,depths,2).execute()
    a,b=run(),run()
    assert a==b
    assert [c['candidate_ids'] for c in a['calls'][:2]]==[[cs[0].function_id,cs[1].function_id],[cs[2].function_id]]
    assert a['actual_tokens']==sum(c['input_tokens']+c['output_tokens'] for c in a['calls'])
    assert a['full_verified_count'] <= 1
    assert a['ranker']=='shared_prior_evidence_fusion_v1'


def test_natural_stop_with_cap_four_does_not_force_four():
    cs,meta,depths=inputs()
    a=make_engine(cs,RecordedBackend(),1000,4,'highest_suspicion',meta,depths,2).execute()
    assert a['full_verified_count']==1
    assert a['stop_reason']=='full_incumbent_above_remaining_l1_heuristic'


@pytest.mark.parametrize('policy',['highest_suspicion','ranker_disagreement','ranking_ambiguity','v2_utility'])
def test_budget_uses_preflight_not_future_output_cost(policy):
    cs,meta,depths=inputs()
    b=RecordedBackend()
    a=make_engine(cs,b,75,4,policy,meta,depths,2).execute()
    assert a['full_verified_count']==0 and a['actual_tokens']==50
    assert all(level==1 for _,level,_ in b.actions)


def test_rank_ties_preserve_original_pool_order_and_interaction_is_nonadditive():
    cs,_,_=inputs()
    def mrr(full):
        ss={c.function_id:CandidateState(1,Observation(c.function_id,.4,.5,'')) for c in cs}
        for i in full:ss[cs[i].function_id]=CandidateState(2,Observation(cs[i].function_id,.8,.2,''))
        order=sorted(cs,key=lambda c:(-fused_score(c,ss[c.function_id]),c.rank))
        return 1/(order.index(cs[2])+1)
    assert mrr([2])==1 and mrr([0,2])==.5
    assert (mrr([0,2])-mrr([])) != pytest.approx((mrr([0])-mrr([]))+(mrr([2])-mrr([])))


def test_controls_read_only_cheap_fields():
    cs,meta,depths=inputs()
    states={c.function_id:CandidateState(1,Observation(c.function_id,.4,.6,'')) for c in cs}
    baseline=cheap_features(cs,states,meta,depths)
    contaminated=deepcopy(meta)
    for m in contaminated.values():m.update(gold=True,future_full=.99,future_output_tokens=99999)
    assert cheap_features(cs,states,contaminated,depths)==baseline
    assert not any(any(word in k for word in ['gold','future','observed_full']) for f in baseline.values() for k in f)


def test_replay_rejects_same_id_but_changed_source(tmp_path,monkeypatch):
    path,case=make_disk_corpus(tmp_path,monkeypatch)
    corpus=ReplayCorpus(path,case)
    c=corpus.candidates[0]
    with pytest.raises(ValueError,match='prompt changed'):
        corpus.backend().execute([replace(c,source=c.source+' // changed')],Action.FULL_VERIFY,2)


class FakeProvider:
    context_window=100
    output_per_candidate=10
    context='same failure'
    seed=0
    model_metadata={'digest':'same-weights'}
    def __init__(self):self.calls=0
    def execute(self,cs,action,level):
        self.calls+=1
        return CallResult([Observation(c.function_id,.1,.8,'paid') for c in cs],30,5,.1)


def test_global_reservation_refuses_before_call_and_cache_not_exposed_by_estimate():
    c=candidates()[0];b=FakeProvider();s=SharedExecution(140,10)
    first=s.execute(b,[c],Action.FULL_VERIFY,2)
    assert s.provider_tokens==35 and b.calls==1
    reused=s.execute(b,[c],Action.FULL_VERIFY,2)
    assert reused==first and b.calls==1 and s.provider_tokens==35
    with pytest.raises(GlobalLimit):s.execute(b,[candidates()[1]],Action.FULL_VERIFY,2)
    assert b.calls==1


def test_future_full_values_do_not_change_first_selected_action(tmp_path,monkeypatch):
    path,case=make_disk_corpus(tmp_path,monkeypatch)
    corpus=ReplayCorpus(path,case)
    cs,meta,depths=inputs()
    choices=[]
    for policy in ['highest_suspicion','ranker_disagreement','ranking_ambiguity','v2_utility']:
        first=[]
        for mutate in [False,True]:
            backend=corpus.backend()
            if mutate:
                for key,result in corpus._entries.items():
                    if key[1]==2:
                        result.observations=[replace(o,suspicion=.01) for o in result.observations]
                        result.output_tokens=999
            engine=make_engine(cs,backend,1000,1,policy,meta,depths,2)
            # Execute only L1; first Full decision must not inspect cached Full.
            while any(s.level<0 for s in engine.status.values()):
                d=engine.decide();r=backend.execute([engine.by_id[i] for i in d.ids],d.action,d.level)
                engine.spent+=r.total_tokens
                for o in r.observations:engine.status[o.function_id]=CandidateState(d.level,o)
            first.append(engine.decide())
        assert first[0]==first[1]
