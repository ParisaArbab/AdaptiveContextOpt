from wp1.program_slicing import (
    backward_slice_python,
    parse_fail_to_pass,
    resolve_test_target,
)


def test_parse_fail_to_pass():
    path, function = parse_fail_to_pass(
        "sympy/core/tests/test_basic.py::test_immutable"
    )
    assert path == "sympy/core/tests/test_basic.py"
    assert function == "test_immutable"


def test_backward_slice_keeps_failure_dependency_chain():
    source = """from pkg import Basic
from other import unused

def test_immutable():
    x = 10
    b1 = Basic()
    y = x + 1
    assert not hasattr(b1, '__dict__')
"""

    result = backward_slice_python(
        source,
        source_file="tests/test_basic.py",
        function_name="test_immutable",
    )

    text = result.text

    assert "from pkg import Basic" in text
    assert "b1 = Basic()" in text
    assert "assert not hasattr(b1, '__dict__')" in text
    assert "x = 10" not in text
    assert "y = x + 1" not in text
    assert "from other import unused" not in text


def test_explicit_criterion_line_is_supported():
    source = """def test_value():
    a = 1
    b = a + 2
    assert b == 3
    assert a == 1
"""

    result = backward_slice_python(
        source,
        source_file="tests/test_value.py",
        function_name="test_value",
        criterion_line=4,
    )

    assert result.criterion_line == 4
    assert "a = 1" in result.text
    assert "b = a + 2" in result.text
    assert "assert b == 3" in result.text
    assert "assert a == 1" not in result.text


def test_resolve_bare_test_name(tmp_path):
    test_file = tmp_path / "pkg/tests/test_basic.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "def test_immutable():\n"
        "    b1 = object()\n"
        "    assert not hasattr(b1, '__dict__')\n"
    )

    path, function = resolve_test_target(tmp_path, "test_immutable")

    assert path == "pkg/tests/test_basic.py"
    assert function == "test_immutable"
