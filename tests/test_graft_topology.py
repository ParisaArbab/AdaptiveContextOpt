import hashlib
import re
import pytest
from dataclasses import replace
from test_graft_graph import fixture_graph
from rave_fl.graft.topology import EmbeddingBundle,TopologyConfig,normalize,propagate,text_hash,topo_select
from rave_fl.graft.selection import query_text


def synthetic_bundle(g):
    # Explicit fixture vectors, never presented as a real learned encoder.
    texts=[n.text for n in g.nodes]+[query_text(g,t) for t in g.targets]
    value={'schema':'graft-embeddings-v1','model_id':'synthetic-test-vectors','synthetic':True,'texts':{}}
    for text in texts:
        words=set(re.findall(r'\w+',text));v=[.1]*8
        for w in words:v[int(hashlib.sha256(w.encode()).hexdigest()[:4],16)%8]+=1
        value['texts'][text_hash(text)]={'dense':v,'sparse':{w:1. for w in words},'truncated':False}
    return EmbeddingBundle(value)


def test_propagation_preserves_mass_and_recovers_a_bridge():
    seed={'a':1.,'bridge':0.,'isolated':.5};adj={'a':{'bridge':1},'bridge':{'a':1},'isolated':{}}
    scores,d=propagate(seed,adj)
    assert scores['bridge']>0 and sum(scores.values())==pytest.approx(1.5)
    assert scores['isolated']==pytest.approx(.5) and d['converged']
    assert propagate({}, {})[0]=={}
    with pytest.raises(ValueError):propagate(seed,adj,eta=1.)

def test_missing_vectors_never_fall_back_to_fake_dense_scores():
    _,_,g=fixture_graph();b=synthetic_bundle(g);b.value['texts'].pop(text_hash(query_text(g,'x.A.a()')))
    with pytest.raises(ValueError,match='unavailable'):topo_select(g,'x.A.a()',b,2400)

def test_topology_and_relevance_ablation_share_inputs_and_budget():
    _,_,g=fixture_graph();b=synthetic_bundle(g)
    a=topo_select(g,'x.A.a()',b,2400);c=topo_select(g,'x.A.a()',b,2400,config=TopologyConfig(propagate=False))
    assert a.input_units<=2400 and c.input_units<=2400 and a.status==c.status=='ready'
    assert a.diagnostics['embedding_synthetic'] and c.diagnostics['iterations']==0
    assert a.hash==topo_select(g,'x.A.a()',b,2400).hash

@pytest.mark.parametrize('with_propagation', [False, True])
def test_same_line_statement_order_is_independent_of_node_iteration(with_propagation):
    from rave_fl.graft.graph import build_graph

    instance, entities, _ = fixture_graph()
    entities[0] = replace(entities[0], source='int a() { int value = b(); return value; }')
    graph = build_graph(instance, entities)
    bundle = synthetic_bundle(graph)
    config = TopologyConfig(propagate=with_propagation)
    original = topo_select(graph, 'x.A.a()', bundle, 2400, config=config)
    for nodes in (tuple(reversed(graph.nodes)), graph.nodes[2:] + graph.nodes[:2]):
        reordered = replace(graph, nodes=nodes)
        repeated = topo_select(reordered, 'x.A.a()', bundle, 2400, config=config)
        assert repeated.scores == original.scores
        assert (repeated.ids, repeated.text, repeated.input_units) == (
            original.ids, original.text, original.input_units
        )

def test_invalid_or_truncated_embedding_is_rejected():
    _,_,g=fixture_graph();b=synthetic_bundle(g);key=next(iter(b.value['texts']));b.value['texts'][key]['truncated']=True
    with pytest.raises(ValueError,match='Truncated'):EmbeddingBundle(b.value)
    assert normalize([0,0])==[0,0] and normalize([1,1])==[1,1]

def test_dense_export_never_loads_or_downloads_without_a_local_model(tmp_path):
    from rave_fl.graft.encode import export
    _,_,g=fixture_graph()
    with pytest.raises(ValueError,match='local'):export(g,tmp_path/'missing','pinned',tmp_path/'embedding.json')
    assert not (tmp_path/'embedding.json').exists()
