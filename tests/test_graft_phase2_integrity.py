"""Synthetic, provider-free tests of the fixed selector and real-bundle guards."""
from copy import deepcopy
from dataclasses import replace
import json
import os
import sys
from types import SimpleNamespace

import pytest

from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.encode import export
from rave_fl.graft.graph import build_graph
from rave_fl.graft.selection import mandatory_ids, query_text
from rave_fl.graft.topology import (
    BGE_M3_DENSE_DIMENSION, BGE_M3_HEAD_FILES, EmbeddingBundle, embedding_texts,
    propagate, text_hash, topo_select,
)
from test_graft_fixed_evidence_schedule import fixed_fixture
from test_graft_graph import fixture_graph
from test_graft_topology import synthetic_bundle


def test_propagation_changes_evidence_on_a_constructed_bridge():
    instance, entities, _ = fixture_graph()
    entity = replace(entities[0], source=(
        'int a() {\n int value = b();\n int next = value + 1;\n return next;\n}'
    ))
    graph = build_graph(instance, [entity])
    target = entity.entity_id
    query = query_text(graph, target)
    texts = embedding_texts(graph)
    # Orthogonal dense vectors remove semantic-neighbor traffic from this fixture.
    # Sparse alignment seeds only the first statement. The next statement is a bridge.
    entries = {
        text_hash(t): {
            'dense': [float(i == j) for j in range(len(texts))],
            'sparse': {'seed': 1.0} if t in {query, 'int value = b();'} else {},
            'truncated': False,
        }
        for i, t in enumerate(texts)
    }
    bundle = EmbeddingBundle({'schema': 'graft-embeddings-v1', 'model_id': 'bridge-fixture',
                              'synthetic': True, 'texts': entries})
    t0, t1 = [topo_select(graph, target, bundle, 3000, config=config, audit=True)
              for config in fixed.ARMS.values()]
    bridge = next(n for n in graph.nodes if n.text == 'int next = value + 1;')
    diff = fixed.compare_evidence(graph, target, t0, t1)
    assert bridge.id in diff['t1_only_ids']
    assert t0.scores[bridge.id] == 0 < t1.scores[bridge.id]
    assert set(mandatory_ids(graph, target)) <= set(diff['common_ids'])
    assert t0.diagnostics['iterations'] == 0 < t1.diagnostics['iterations']
    assert t1.diagnostics['converged'] and t1.diagnostics['residual'] <= fixed.ARMS[fixed.T1].tolerance
    assert t1.diagnostics['score_mass'] == pytest.approx(t0.diagnostics['seed_mass'])
    assert any(e['source'] == bridge.id and e['relation'] == 'REQUIRES' and 'lexical' in e['provenance']
               for e in diff['source_relations'])
    assert all(not s.diagnostics['config']['typed'] for s in (t0, t1))
    repeated = topo_select(graph, target, bundle, 3000, config=fixed.ARMS[fixed.T1], audit=True)
    assert (t1.ids, t1.text, t1.scores) == (repeated.ids, repeated.text, repeated.scores)


def test_nonconvergence_and_mandatory_budget_fail_closed():
    with pytest.raises(ValueError, match='did not converge'):
        propagate({'a': 1., 'b': 0.}, {'a': {'b': 1.}, 'b': {'a': 1.}}, max_iterations=1)
    _, entities, graph = fixture_graph()
    bundle = synthetic_bundle(graph)
    for config in fixed.ARMS.values():
        sel = topo_select(graph, entities[0].entity_id, bundle, 1, config=config)
        assert sel.status == 'infeasible'
        assert sel.input_units > sel.budget
        assert set(sel.ids) == set(mandatory_ids(graph, sel.target))


@pytest.mark.parametrize(('field', 'bad', 'error'), [
    ('dense', [0.] * 8, 'norm'),
    ('dense', [1.] * 7, 'norm'),
    ('dense', [1e308] * 8, 'norm'),
    ('dense', [True] * 8, 'dense'),
    ('dense', [float('nan')] * 8, 'dense'),
    ('sparse', {'x': -1}, 'lexical'),
    ('sparse', {'x': float('inf')}, 'lexical'),
    ('sparse', {'x': True}, 'lexical'),
    ('truncated', True, 'Truncated'),
])
def test_malformed_embeddings_are_rejected(field, bad, error):
    _, _, graph = fixture_graph()
    data = synthetic_bundle(graph).value
    next(reversed(data['texts'].values()))[field] = bad
    with pytest.raises(ValueError, match=error):
        EmbeddingBundle(data)


def test_exact_text_hash_checks_whitespace_and_unicode_bytes():
    _, entities, graph = fixture_graph()
    bundle = synthetic_bundle(graph)
    query = query_text(graph, entities[0].entity_id)
    bundle.lookup(query)
    assert text_hash('é') != text_hash('e\N{COMBINING ACUTE ACCENT}')
    with pytest.raises(ValueError, match='no synthetic fallback'):
        bundle.lookup(query + '\n')
    changed = replace(graph, nodes=tuple(replace(n, text=n.text + '\n') if n.kind == 'code' else n
                                         for n in graph.nodes))
    with pytest.raises(ValueError, match='missing, .* extra'):
        bundle.validate_for_graph(changed)
    bundle.value['texts'][text_hash('unrelated text')] = next(iter(bundle.value['texts'].values()))
    with pytest.raises(ValueError, match='0 missing, 1 extra'):
        bundle.validate_for_graph(graph)


def test_duplicate_embedding_keys_are_rejected(tmp_path):
    bundle = tmp_path / 'duplicate.json'
    bundle.write_text('{"texts": {"same-hash": {}, "same-hash": {}}}')
    with pytest.raises(ValueError, match='Duplicate key'):
        EmbeddingBundle.load(bundle)


def test_real_bundle_manifest_contract_without_claiming_real_vectors():
    instance, entities, _ = fixture_graph()
    graph = build_graph(replace(instance, revision='a' * 40), entities)
    # This is test data that exercises the non-synthetic metadata branch only.
    # It never leaves this test and must never be counted as a real-encoder result.
    data = {**synthetic_bundle(graph).value, 'synthetic': False, 'model_id': 'BGE-M3:pinned-fixture'}
    with pytest.raises(ValueError, match='provenance'):
        EmbeddingBundle(data).validate_for_graph(graph)
    data['provenance'] = {
        'graph_sha256': graph.hash, 'source_revision': graph.revision,
        'checkpoint_revision': 'pinned-fixture', 'max_length': 8192, 'max_observed_tokens': 30,
        'checkpoint_files': {p: text_hash(p) for p in (
            'config.json', 'tokenizer.json', 'model.safetensors', *sorted(BGE_M3_HEAD_FILES)
        )},
    }
    with pytest.raises(ValueError, match='dense dimension'):
        EmbeddingBundle(data).validate_for_graph(graph)
    for entry in data['texts'].values():
        entry['dense'].extend([0.] * (BGE_M3_DENSE_DIMENSION - len(entry['dense'])))
    EmbeddingBundle(data).validate_for_graph(graph)
    with pytest.raises(ValueError, match='graph hash or source revision'):
        EmbeddingBundle(data).validate_for_graph(replace(graph, revision='b' * 40))
    with pytest.raises(ValueError, match='graph hash or source revision'):
        EmbeddingBundle(data).validate_for_graph(replace(graph, nodes=tuple(reversed(graph.nodes))))
    for field, value, error in [
        ('checkpoint_revision', 'main', 'pinned'),
        ('checkpoint_files', {'config.json': 'x'}, 'pinned'),
        ('max_observed_tokens', 9000, 'non-truncation'),
    ]:
        invalid = deepcopy(data)
        invalid['provenance'][field] = value
        with pytest.raises(ValueError, match=error):
            EmbeddingBundle(invalid).validate_for_graph(graph)
    del next(iter(data['texts'].values()))['truncated']
    with pytest.raises(ValueError, match='non-truncation'):
        EmbeddingBundle(data).validate_for_graph(graph)


def test_real_plan_rejects_synthetic_bundle_before_selection(fixed_fixture, monkeypatch):
    value, _ = fixed_fixture
    value['benchmark']['backend'] = 'ollama'

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid real bundle reached selection or model execution')

    monkeypatch.setattr(fixed, 'topo_select', forbidden)
    monkeypatch.setattr(fixed, 'GraftProvider', forbidden)
    plan, pairs = fixed.prepare(fixed.configuration(value))
    assert not pairs and plan['new_model_calls'] == 0
    assert all(p['status'] == 'infeasible' and 'Synthetic embedding' in p['reasons'][0]
               for p in plan['pairs'])


def test_checkpoint_metadata_alone_never_starts_encoder(tmp_path):
    _, _, graph = fixture_graph()
    model = tmp_path / 'local-checkpoint'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'tokenizer.json').write_text('{}')
    out = tmp_path / 'embeddings.json'
    with pytest.raises(ValueError, match='Local BGE-M3 weights'):
        export(graph, model, 'fixture', out)
    (model / 'model.safetensors').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='heads'):
        export(graph, model, 'fixture', out)
    assert not out.exists()


def test_local_export_checks_token_lengths_before_loading_encoder(tmp_path, monkeypatch):
    instance, entities, _ = fixture_graph()
    graph = build_graph(replace(instance, revision='a' * 40), entities)
    model = tmp_path / 'local-checkpoint'
    model.mkdir()
    for name in ('config.json', 'tokenizer.json', 'model.safetensors', *BGE_M3_HEAD_FILES):
        (model / name).write_bytes(b'fixture')

    def tokenizer(path, **kwargs):
        assert path == str(model.resolve())
        assert kwargs == {'local_files_only': True, 'trust_remote_code': False}
        assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'

        def encode(text, *, truncation):
            assert not truncation
            assert text in embedding_texts(graph)
            return [1, 2, 3]

        return SimpleNamespace(encode=encode)

    def forbidden(*args, **kwargs):
        pytest.fail('An overlength input reached the encoder')

    monkeypatch.setenv('HF_HUB_OFFLINE', 'previous-value')
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer)
    ))
    monkeypatch.setitem(sys.modules, 'FlagEmbedding', SimpleNamespace(BGEM3FlagModel=forbidden))
    out = tmp_path / 'embeddings.json'
    with pytest.raises(ValueError, match='truncation would occur'):
        export(graph, model, 'fixture', out, max_length=2)
    assert not out.exists()
    assert os.environ['HF_HUB_OFFLINE'] == 'previous-value'
    assert 'TRANSFORMERS_OFFLINE' not in os.environ


def test_plan_reports_pair_identity_exact_evidence_and_provenance(fixed_fixture):
    _, config = fixed_fixture
    plan, pairs = fixed.prepare(config)
    for record in plan['pairs']:
        graph = pairs[record['instance_id']].graph
        assert record['graph_sha256'] == graph.hash
        assert record['revision'] == graph.revision
        assert record['embedding_synthetic'] is True
        assert len(record['evidence_comparisons']) == len(record['schedule'])
        for comparison, a, b in zip(record['evidence_comparisons'], record['arms'][fixed.T0],
                                    record['arms'][fixed.T1], strict=True):
            assert comparison['query_sha256'] == text_hash(query_text(graph, a['target']))
            common = set(a['selected_ids']) & set(b['selected_ids'])
            assert set(comparison['common_ids']) == common
            assert comparison['jaccard'] == len(common) / len(set(a['selected_ids']) | set(b['selected_ids']))
            assert a['input_allowance'] == b['input_allowance']
            for row in (a, b):
                messages = row['request']['messages']
                assert text_hash(messages[0]['content']) == plan['system_prompt_sha256']
                assert text_hash(messages[1]['content']) == row['evidence_message_sha256']
                assert row['input_units'] == sum(len(m['content'].encode()) for m in messages) + 256
                assert row['selected_node_count'] == len(row['selected_ids'])
                assert json.loads(messages[1]['content'])['target'] == comparison['target']
            for audit in comparison['evidence_nodes']:
                node = graph.by_id[audit['id']]
                assert audit['exact_text_sha256'] == text_hash(node.text)
                assert audit['origin']['source_hash'] == (
                    text_hash(graph.sources[node.owner]['source']) if node.owner else record['evidence_sha256']
                )
