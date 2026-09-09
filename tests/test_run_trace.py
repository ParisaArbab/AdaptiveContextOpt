import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wp1'))

from llm_backends import ChatResponse
from run_trace import RunTrace
from token_meter import TokenMeter


class Counter:
    method = 'test_counter'
    is_exact = False

    def count(self, text):
        return len(text or '')


def test_trace_persists_full_response_and_failed_request(tmp_path):
    path = tmp_path / 'events.jsonl'
    meter = TokenMeter(Counter(), RunTrace(path))
    large_response = 'x' * 9000
    chat = meter.wrap(lambda s, u: ChatResponse(large_response, usage={'prompt_tokens': 4}))
    with meter.stage('agent4sr'):
        assert chat('system', 'full prompt') == large_response

    def fail(system, user):
        raise RuntimeError('model disconnected')

    with pytest.raises(RuntimeError):
        with meter.stage('agent4lr'):
            meter.wrap(fail)('system2', 'prompt before failure')

    # Read before a run finishes: nothing depends on final result serialization.
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e['event'] for e in events] == ['llm_request', 'llm_response', 'llm_request', 'llm_error']
    assert events[1]['response'] == large_response
    assert events[1]['provider_metadata']['usage']['prompt_tokens'] == 4
    assert events[2]['user'] == 'prompt before failure'
    assert events[3]['stage'] == 'agent4lr'
    assert events[3]['error'] == 'model disconnected'


def test_provider_usage_replaces_estimate(tmp_path):
    meter = TokenMeter(Counter(), RunTrace(tmp_path / 'usage.jsonl'))
    chat = meter.wrap(lambda s, u: ChatResponse('answer', usage={'prompt_tokens': 12, 'completion_tokens': 3}))
    chat('much longer system message', 'user')
    assert meter.report()['llm_total_tokens'] == 15
    assert meter.report()['llm_token_source'] == 'provider_usage'


def test_cli_archives_capture_and_each_arm_without_changing_predictions(tmp_path, monkeypatch):
    import run_wp1_benchmark as runner
    from docker_harness import TestRunResult
    from leanctx_compressor import CompressionResult

    instance = {'instance_id': 'fixture__fixture-1', 'repo': 'fixture/fixture',
                'base_commit': 'test', 'problem_statement': 'bad function fails',
                'fail_to_pass': ['test_bad'], 'functions': ['code.py::bad'],
                'files': ['code.py'], 'language': 'python'}
    instances = tmp_path / 'instances.json'
    instances.write_text(json.dumps([instance]))
    raw = 'FAILED test_bad\n'
    capture = TestRunResult(instance['instance_id'], raw, '', 1, 'local_fallback')
    structure = {'code.py::bad()': {'file': 'code.py', 'line': 1, 'id': 'bad'}}
    monkeypatch.setattr(runner.docker_harness, 'run_local_fallback', lambda **kw: capture)
    monkeypatch.setattr(runner.graphify_structure, 'build_structure_map', lambda *a, **kw: structure)
    monkeypatch.setattr(runner.graphify_structure, 'build_call_graph', lambda *a: {})
    monkeypatch.setattr(runner.leanctx_compressor, 'compress', lambda text, **kw:
                        CompressionResult(text, 'cli', 5, 5, 0))
    monkeypatch.setattr(runner.token_meter, 'TokenCounter', lambda *a: Counter())
    monkeypatch.setattr(runner.llm_backends, 'build_chat_fn', lambda cfg:
                        lambda system, user: 'OK' if 'auditing' in system else 'Top_1 : code.py::bad')
    out = tmp_path / 'results' / 'wp1_results.json'
    monkeypatch.setattr(sys, 'argv', ['run_wp1_benchmark.py', '--instances', str(instances),
                                    '--out', str(out), '--repos-dir', str(tmp_path / 'repos'),
                                    '--arms', 'full,pure_flexfl', '--llm', 'ollama',
                                    '--model', 'llama3.1:8b', '--local-fallback'])
    runner.main()
    payload = json.loads(out.read_text())
    assert len(payload['outcomes']) == 2
    root = Path(payload['config']['trace_dir'])
    assert (root / 'source' / 'wp1' / 'agent_localizer.py').exists()
    per_instance = root / instance['instance_id']
    assert (per_instance / 'raw_test_output.txt').read_text() == raw
    assert json.loads((per_instance / 'capture.json').read_text())['stdout'] == raw
    for outcome in payload['outcomes']:
        assert outcome['predicted_functions'][0] == 'code.py::bad()'
        arm = per_instance / outcome['arm']
        assert json.loads((arm / 'outcome.json').read_text()) == outcome
        assert (arm / 'localization.json').exists()
        events = [json.loads(l) for l in (arm / 'events.jsonl').read_text().splitlines()]
        assert any(e['event'] == 'llm_request' and raw in e['user'] for e in events)
        assert any(e['event'] == 'llm_response' and 'Top_1' in e['response'] for e in events)
