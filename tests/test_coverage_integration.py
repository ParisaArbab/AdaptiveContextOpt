import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wp1'))
from traditional_fl import run_coverage, ochiai_from_coverage


def test_native_wrapper_does_not_swallow_test_contexts(tmp_path):
    pytest.importorskip('coverage')
    (tmp_path / 'code.py').write_text('def bad():\n    return 2\n\ndef good():\n    return 1\n')
    (tmp_path / 'test_suite.py').write_text(
        'from code import bad, good\n'
        'def test_bad():\n    assert bad() == 1\n'
        'def test_good():\n    assert good() == 1\n')
    (tmp_path / 'runner.py').write_text(
        'import test_suite\n'
        'def test():\n'
        '    try:\n        test_suite.test_bad()\n'
        '    except AssertionError:\n        pass\n'
        '    test_suite.test_good()\n'
        'test()\n')
    data, error = run_coverage(tmp_path, Path(sys.executable), [sys.executable, 'runner.py'])
    assert not error
    contexts = {c for f in data['files'].values() for cs in f['contexts'].values() for c in cs}
    assert 'test_suite.py::test_bad' in contexts
    assert 'test_suite.py::test_good' in contexts
    assert not any(c.endswith('runner.test') for c in contexts)
    structure = {'code.py::bad()': {'file': 'code.py', 'line': 1, 'end_line': 2},
                 'code.py::good()': {'file': 'code.py', 'line': 4, 'end_line': 5}}
    ranked = ochiai_from_coverage(data, structure, tmp_path, ['test_bad'])
    assert ranked.entries == ['code.py::bad()']
    assert ranked.detail['failing_contexts'] == ranked.detail['passing_contexts'] == 1
    assert 'functions' not in data['files']['code.py']
