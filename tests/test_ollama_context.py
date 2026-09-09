import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wp1'))
from llm_backends import build_chat_fn, resolve


def test_native_ollama_requests_explicit_capacity_and_preserves_usage(monkeypatch):
    def send(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == 'http://localhost:11434/api/chat'
        assert payload['options']['num_ctx'] == 16384
        assert payload['messages'][1]['content'] == 'evidence'
        return io.BytesIO(json.dumps({'message': {'content': 'Top_1 : a.py::f'},
                                     'prompt_eval_count': 6000, 'eval_count': 12,
                                     'done_reason': 'stop'}).encode())
    monkeypatch.setattr('urllib.request.urlopen', send)
    response = build_chat_fn(resolve('ollama', model='llama3.1:8b'))('system', 'evidence')
    assert response.metadata['usage'] == {'prompt_tokens': 6000, 'completion_tokens': 12}


def test_context_saturation_is_not_silently_accepted(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(json.dumps({
        'message': {'content': 'ranking'}, 'prompt_eval_count': 16384, 'eval_count': 12}).encode()))
    with pytest.raises(RuntimeError, match='possible truncation'):
        build_chat_fn(resolve('ollama', model='llama3.1:8b'))('system', 'evidence')
