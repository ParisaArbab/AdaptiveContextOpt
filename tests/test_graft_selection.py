import json
import pytest
from test_graft_graph import fixture_graph
from rave_fl.graft.selection import ByteCounter,SYSTEM,mandatory_ids,package,relevance_select,lexical_scores,query_text


def test_complete_input_budget_includes_wrapper_and_system():
    _,_,g=fixture_graph();target='x.A.a()';m=package(g,target,mandatory_ids(g,target),100000)
    assert m.input_units==len(m.text.encode())+len(SYSTEM.encode())+256
    assert package(g,target,m.ids,m.input_units).status=='ready'
    assert package(g,target,m.ids,m.input_units-1).status=='infeasible'
    assert package(g,target,[],10000).status=='infeasible'

def test_packing_is_deterministic_and_never_breaks_a_statement_to_fit():
    _,_,g=fixture_graph();a=relevance_select(g,'x.A.a()',2400);b=relevance_select(g,'x.A.a()',2400)
    assert a.hash==b.hash and a.input_units<=2400
    assert set(mandatory_ids(g,'x.A.a()'))<=set(a.ids)
    obj=json.loads(a.text)
    for unit in obj['units']:assert unit['text']==g.by_id[unit['id']].text
    assert a.counter.startswith('utf8_') and not a.exact_template_tokens

def test_unknown_candidates_and_invalid_budgets_fail():
    _,_,g=fixture_graph()
    with pytest.raises(ValueError):relevance_select(g,'gold-not-in-pool',2400)
    with pytest.raises(ValueError):relevance_select(g,'x.A.a()',0)

def test_gold_is_not_an_input_to_query_or_scoring():
    _,_,g=fixture_graph();q=query_text(g,'x.A.a()');scores=lexical_scores(g,q)
    assert set(scores)==set(g.by_id)
    assert all(v>=0 for v in scores.values())
