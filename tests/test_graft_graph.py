from dataclasses import replace
import hashlib
import pytest
from rave_fl.eval.schema import CodeEntity,Evidence,OnlineInstance
from rave_fl.graft.graph import build_graph,Graph,Edge,CoverageFact,Origin


def fixture_graph():
    text='TEST: fixture\nFAILURE: wrong answer\nat x.A.a(A.java:12)'
    i=OnlineInstance('custom','fixture','A-1','fixture/A','java','','','wrong answer',Evidence('observed_failure',text))
    es=[CodeEntity('x.A.a()','A.java',10,14,'int a() {\n int value = b();\n return value;\n}',.9,kind='method'),
        CodeEntity('x.A.b()','A.java',20,22,'int b() { return 2; }',.5,kind='method')]
    return i,es,build_graph(i,es)

def test_typed_graph_reproducible_and_online_only():
    i,es,g=fixture_graph();assert g.hash==build_graph(i,es).hash
    assert Graph.from_dict(g.to_dict()).hash==g.hash
    assert {'MAY_CALL','REQUIRES','STACK_AT'}<=set(e.relation for e in g.edges)
    assert not any(e.relation=='EXECUTES' for e in g.edges)
    assert set(g.targets)=={e.entity_id for e in es}
    with pytest.raises(TypeError):Graph.from_dict({**g.to_dict(),'gold':['secret']})

def test_invalid_edges_and_bounded_graph_are_rejected():
    i,es,g=fixture_graph()
    with pytest.raises(ValueError):replace(g,edges=g.edges+(Edge('unknown',g.nodes[0].id,'MAY_CALL','fixture'),))
    with pytest.raises(ValueError,match='limit'):build_graph(i,es,max_nodes=1)

def test_ambiguous_calls_are_not_recorded_as_execution():
    i,es,_=fixture_graph();es[0]=replace(es[0],source='int a() { return b(1); }')
    es.extend([CodeEntity('x.A.b(int)','A.java',30,31,'int b(int x) { return x; }',.3,kind='method'),CodeEntity('x.A.b(String)','A.java',40,41,'int b(String x) { return 1; }',.2,kind='method')])
    g=build_graph(i,es);assert any(d['kind']=='ambiguous_call' for d in g.diagnostics)
    assert not any(e.relation=='MAY_CALL' for e in g.edges)

def test_coverage_requires_actual_evidence_and_hash():
    i,es,_=fixture_graph();raw='fixture test executed x.A.a()';origin=Origin('coverage.txt',1,1,hashlib.sha256(raw.encode()).hexdigest(),'instrumentation')
    g=build_graph(i,es,[CoverageFact('x.A.a()','fixture test',raw,origin)])
    assert sum(e.relation=='EXECUTES' for e in g.edges)==1
    with pytest.raises(TypeError):build_graph(i,es,[{'Ochiai':1.}])

def test_python_structural_units_and_missing_parser_are_explicit():
    i,_,_=fixture_graph();i=replace(i,language='python')
    es=[CodeEntity('a.py::f','a.py',1,4,'def f(x):\n    value = x + 1\n    return value\n',.5)]
    g=build_graph(i,es);assert any(e.relation=='REQUIRES' for e in g.edges)
    assert not any(e.relation=='MAY_CALL' for e in g.edges)
