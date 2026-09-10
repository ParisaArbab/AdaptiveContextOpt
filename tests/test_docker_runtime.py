import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('pipeline_docker_runtime', Path(__file__).resolve().parents[1] / 'docker/run.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def setup_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'APP_ROOT', tmp_path)
    packages = tmp_path / 'packages.txt'
    packages.write_text('graphifyy==0.9.32')
    monkeypatch.setattr(runtime, 'PACKAGES_FILE', packages)
    monkeypatch.setattr(sys, 'argv', ['run.py', '--instances', '/inputs/sample.json'])
    monkeypatch.setenv('MODEL', 'llama3.1:8b')
    monkeypatch.setenv('REQUIRE_GPU', '1')
    calls = []
    def api(base, path, payload=None, **kw):
        calls.append(path)
        if path in ('/api/tags', '/api/ps'):
            return {'models': [{'name': 'llama3.1:8b', 'size_vram': 8_000_000_000}]}
        return {}
    monkeypatch.setattr(runtime, 'request', api)
    return tmp_path, calls


def test_gpu_run_preserves_pipeline_arguments_and_existing_model(setup_runtime, monkeypatch):
    root, calls = setup_runtime
    def run(command):
        assert command[command.index('--base-url') + 1] == 'http://ollama:11434'
        assert '--local-fallback' in command
        assert command[-2:] == ['--instances', '/inputs/sample.json']
        output = Path(command[command.index('--out-dir') + 1])
        (output / 'wp1_results.json').write_text(json.dumps({'outcomes': [{}], 'skipped': []}))
        return 0
    monkeypatch.setattr(runtime.subprocess, 'call', run)
    assert runtime.main() == 0
    assert '/api/pull' not in calls
    evidence = json.loads(next((root / 'results').glob('*/environment.json')).read_text())
    assert evidence['pipeline_exit_code'] == 0
    assert evidence['loaded_models']['models'][0]['size_vram'] > 0


def test_gpu_mode_rejects_cpu_fallback_and_saves_error(setup_runtime, monkeypatch):
    root, _ = setup_runtime
    monkeypatch.setattr(runtime, 'request', lambda *a, **k: {'models': [{'name': 'llama3.1:8b', 'size_vram': 0}]})
    monkeypatch.setattr(runtime.subprocess, 'call', lambda *_: pytest.fail('Must not start the benchmark'))
    with pytest.raises(RuntimeError, match='no GPU allocation'):
        runtime.main()
    assert 'no GPU allocation' in next((root / 'results').glob('*/environment.json')).read_text()


@pytest.mark.parametrize('result', [None, {'outcomes': [], 'skipped': []}, {'outcomes': [{}], 'skipped': [{}]}])
def test_missing_or_incomplete_results_are_not_success(setup_runtime, monkeypatch, result):
    def run(command):
        if result is not None:
            output = Path(command[command.index('--out-dir') + 1])
            (output / 'wp1_results.json').write_text(json.dumps(result))
        return 0
    monkeypatch.setattr(runtime.subprocess, 'call', run)
    with pytest.raises(RuntimeError):
        runtime.main()


def test_model_override_cannot_bypass_gpu_check(setup_runtime, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run.py', '--model', 'different-model'])
    with pytest.raises(SystemExit, match='Set MODEL'):
        runtime.main()
