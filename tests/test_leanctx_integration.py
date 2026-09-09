import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wp1'))
import leanctx_compressor as lc


def test_missing_real_compressor_does_not_silently_use_reference(monkeypatch):
    monkeypatch.setenv('LEAN_CTX_BINARY', '/nonexistent/lean-ctx')
    with pytest.raises(RuntimeError, match='Real LeanCTX compression failed'):
        lc.compress('test output')


def test_official_binary_compresses_capture_and_records_provenance(tmp_path):
    binary = Path(__file__).resolve().parents[1] / '.tools/leanctx/3.10.1/lean-ctx'
    if not binary.is_file():
        pytest.skip('Run scripts/install_leanctx.py for real integration test')
    raw = '\n'.join(['repeated build progress'] * 200 + ['TypeError: broken comparison']) + '\n'
    result = lc.compress(raw, project_root=tmp_path, command='python bin/test')
    assert result.mode == 'cli'
    assert result.compressed_tokens_est <= result.original_tokens_est
    assert 'TypeError: broken comparison' in result.text
    assert result.detail['executable_sha256']
    assert 'compress preview' in result.detail['report']
    assert result.detail['compressed_bytes'] == len(result.text.encode())
