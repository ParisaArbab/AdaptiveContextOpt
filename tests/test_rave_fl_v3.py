"""Focused V3 reservation, selection, complete decision audit and metric tests."""
from dataclasses import replace
from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from rave_fl.budget_reservation import ReservedBackend, ReservationViolation
from rave_fl.core import Action
from rave_fl.pilot_types import CallResult, Observation
from rave_fl.selection_controls import make_engine
from rave_fl.v3_diagnostics import AuditedUtility
from rave_fl.v3_diagnostics import recorded_ranker_values
from rave_fl.pilot_types import PolicyConfig
from rave_fl.rank_repair import RepairConfig
from test_rave_fl_rank_repair import candidates, RecordedBackend
from test_rave_fl_iteration_d import inputs


def spec():return json.loads(Path('configs/rave_v3_token_reservation.json').read_text())


class Backend(RecordedBackend):
    def __init__(self):
        super().__init__();self._planner=SimpleNamespace(context='failure');self.n=0
    def execute(self,cs,action,level):
        self.n+=1
        return super().execute(cs,action,level)


def reserved(budget=20000):
    b=Backend();s=spec()
    return b,ReservedBackend(b,budget,s,s['model_digest'],(s['model'],0,0,16384,160))


def test_reservation_denies_before_cache_revelation():
    b,r=reserved(1)
    with pytest.raises(ReservationViolation):r.execute(candidates()[:1],Action.FULL_VERIFY,2)
    assert b.n==0 and r.spent==0


def test_unknown_model_or_template_refused():
    b=Backend();s=spec()
    with pytest.raises(ValueError,match='digest'):ReservedBackend(b,20000,s,'unknown',(s['model'],0,0,16384,160))
    s['template']+='changed'
    with pytest.raises(ValueError,match='checksum'):ReservedBackend(b,20000,s,s['model_digest'],(s['model'],0,0,16384,160))


def test_utf8_reservation_and_paid_counter_violation_not_hidden():
    b,r=reserved();c=replace(candidates()[0],source='// '+('λ'*100))
    inp,out=r.bounds([c],Action.FULL_VERIFY,2)
    assert inp>200 and out==160
    b.execute=lambda cs,a,l:CallResult([Observation(cs[0].function_id,.8,.2,'')],inp+1,0,0)
    with pytest.raises(ReservationViolation,match='counters'):r.execute([c],Action.FULL_VERIFY,2)
    assert r.spent==inp+1 and r.events[-1]['actual_input_tokens']==inp+1


def test_highest_suspicion_does_not_substitute_affordable_runner_up():
    cs,meta,depths=inputs();b=RecordedBackend()
    # All batches fit; the highest suspicious candidate's Full does not.
    original=b.estimate
    b.estimate=lambda group,a,l: 5000 if l==2 and group[0]==cs[-1] else original(group,a,l)
    engine=make_engine(cs,b,1000,1,'highest_suspicion',meta,depths,2)
    a=engine.execute()
    assert a['full_verified_count']==0


def test_audit_records_all_candidates_even_rejected_by_budget():
    cs,meta,depths=inputs()
    engine=AuditedUtility(cs,RecordedBackend(),PolicyConfig(token_budget=75,batch_size=2,mode='repair_utility'),RepairConfig(1,5))
    engine.option_audit=[];engine.audit_metadata=meta;engine.audit_depths=depths
    a=engine.execute()
    assert len(a['option_audit'][-1]['options'])==3
    assert all(r['decision']=='budget_or_context' for r in a['option_audit'][-1]['options'])


@pytest.mark.parametrize('policy',['fixed_l1','highest_suspicion','ranker_disagreement','ranking_ambiguity','v2_utility','uncertainty_aware'])
def test_no_upgrade_cap_and_unverified_candidates_preserved(policy):
    cs,meta,depths=inputs()
    a=make_engine(cs,RecordedBackend(),1000,0,policy,meta,depths,2).execute()
    assert a['full_verified_count']==0 and a['screened_count']==3
    assert set(a['ranked_functions'])=={c.function_id for c in cs}
    assert a['actual_tokens']==50


def test_raw_ranker_csv_scores_are_optional_and_duplicate_rows_do_not_inflate_rank(tmp_path, monkeypatch):
    import wp1.flexfl_data
    monkeypatch.setattr(wp1.flexfl_data, 'flexfl_root', lambda _:tmp_path)
    for method in ['SBIR','Ochiai','BoostN']:
        p=tmp_path/'data/FL_results'/method/'Defects4J/Bug-1_method-susps.csv'
        p.parent.mkdir(parents=True)
        p.write_text('File,Signature,StartLine,EndLine'+(',Suspiciousness' if method=='BoostN' else '')+'\n'+
                     'p.C,a(),1,2'+(',.9' if method=='BoostN' else '')+'\n'+
                     'p.C,a(),1,2'+(',.8' if method=='BoostN' else '')+'\n'+
                     'p.C,b(),3,4'+(',.7' if method=='BoostN' else '')+'\n')
    values, provenance=recorded_ranker_values(tmp_path,'Bug-1')
    assert values['p.C.b()']['SBIR_full_list_rank']==2
    assert values['p.C.a()']['BoostN_recorded_score']==.9
    assert values['p.C.a()']['SBIR_recorded_score'] is None
    assert provenance['BoostN']['rows']==3 and provenance['SBIR']['unique_valid_candidates']==2
