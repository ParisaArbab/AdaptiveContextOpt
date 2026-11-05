from dataclasses import replace
import pytest
from test_graft_graph import fixture_graph
from rave_fl.graft.graph import Edge
from rave_fl.graft.selection import mandatory_ids,package
from rave_fl.graft.paths import neighbors,reachable,shortest_path,preserve_paths


def test_shortest_paths_keep_hop_state_and_have_a_hard_limit():
    a={'s':{'a':1,'t':.1},'a':{'t':1},'t':{}}
    assert shortest_path(a,{'s'},'t',{},1)==('s','t')
    assert shortest_path(a,{'s'},'t',{},2)==('s','a','t')
    assert shortest_path({'s':{},'t':{}},{'s'},'t',{},2) is None

def test_path_recovery_is_explicit_and_fits_only_when_feasible():
    _,_,g=fixture_graph();target='x.A.a()';m=mandatory_ids(g,target)
    base=package(g,target,m,10000,scores={n.id:1. for n in g.nodes})
    augmented=preserve_paths(g,base,terminals=[g.targets['x.A.b()']],max_hops=4)
    assert augmented.status=='ready' and augmented.diagnostics['path_coverage_after']==1
    assert len(augmented.ids)>len(base.ids)
    tight=package(g,target,m,base.input_units,scores={n.id:1. for n in g.nodes})
    result=preserve_paths(g,tight,terminals=[g.targets['x.A.b()']],max_hops=4)
    assert result.status=='infeasible' and result.diagnostics['path_status']=='protected_path_infeasible'

def test_semantic_edges_are_never_trusted_path_shortcuts():
    _,_,g=fixture_graph()
    with pytest.raises(ValueError):neighbors(g,{'SEMANTIC'})
    base=package(g,'x.A.a()',mandatory_ids(g,'x.A.a()'),10000)
    modified=replace(g,revision='1'*40)
    with pytest.raises(ValueError,match='fingerprints'):preserve_paths(modified,base)

def test_unreachable_paths_do_not_masquerade_as_complete():
    _,_,g=fixture_graph();g=replace(g,edges=tuple(e for e in g.edges if e.relation!='STACK_AT'))
    s=package(g,'x.A.a()',mandatory_ids(g,'x.A.a()'),10000)
    r=preserve_paths(g,s,terminals=[g.targets['x.A.b()']])
    assert r.status=='ready' and r.diagnostics['path_coverage_after']==0 and r.diagnostics['unreachable_terminals']
