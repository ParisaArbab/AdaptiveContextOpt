from dataclasses import replace
from test_graft_graph import fixture_graph
from rave_fl.graft.graph import Edge
from rave_fl.graft.selection import mandatory_ids, package
from rave_fl.graft.repair import closure, missing_support, repair_support


def test_closure_restores_definitions_and_owner_without_increasing_budget():
    _, _, graph = fixture_graph()
    target = 'x.A.a()'
    fragment = next(n for n in graph.nodes if n.text == 'return value;')
    base = package(graph, target, set(mandatory_ids(graph, target)) | {fragment.id}, 9000)
    assert missing_support(graph, base.ids)
    result = repair_support(graph, base)
    assert result.status == 'ready' and not missing_support(graph, result.ids)
    assert result.budget == base.budget
    assert not result.diagnostics['semantic_completeness_guaranteed']


def test_cycles_terminate_and_eviction_cascades():
    _, _, graph = fixture_graph()
    a, b = [n.id for n in graph.nodes if n.kind == 'code'][:2]
    graph = replace(graph, edges=graph.edges + (Edge(a, b, 'REQUIRES', 'fixture cycle'), Edge(b, a, 'REQUIRES', 'fixture cycle')))
    assert {a, b} <= closure(graph, {a})
    ids = set(mandatory_ids(graph, 'x.A.a()')) | {b}
    base = package(graph, 'x.A.a()', ids, 10000)
    tight = replace(base, budget=base.input_units)
    result = repair_support(graph, tight)
    assert result.status == 'ready' and result.input_units <= tight.budget
    assert not missing_support(graph, result.ids)
    assert not ({a, b} & set(result.ids))


def test_protected_support_can_be_infeasible():
    _, _, graph = fixture_graph()
    fragment = next(n for n in graph.nodes if n.text == 'return value;')
    ids = set(mandatory_ids(graph, 'x.A.a()')) | {fragment.id}
    base = package(graph, 'x.A.a()', ids, 10000, diagnostics={'paths': {'protected': [fragment.id]}})
    tight = replace(base, budget=base.input_units)
    result = repair_support(graph, tight)
    assert result.status == 'infeasible' and result.diagnostics['observable_support_closed']
    assert result.input_units > result.budget


def test_failing_coverage_identity_is_exact_not_inferred():
    import hashlib
    from rave_fl.graft.graph import build_graph, CoverageFact, Origin
    instance, entities, _ = fixture_graph()
    text = 'coverage row'
    origin = Origin('coverage.txt', 1, 1, hashlib.sha256(text.encode()).hexdigest(), 'fixture instrumentation')
    matched = build_graph(instance, entities, [CoverageFact(entities[0].entity_id, 'fixture', text, origin)])
    unmatched = build_graph(instance, entities, [CoverageFact(entities[0].entity_id, 'another', text, origin)])
    assert any(e.relation == 'FAILS_TEST' for e in matched.edges)
    assert not any(e.relation == 'FAILS_TEST' for e in unmatched.edges)
