from wp1.program_slicing import (
    backward_slice_python,
    expand_production_dependencies,
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



def test_backward_slice_crosses_to_module_level_fixture():
    source = """from pkg.basic import Basic
from other import unused

b1 = Basic()
noise = 123

def test_immutable():
    assert not hasattr(b1, '__dict__')
"""

    result = backward_slice_python(
        source,
        source_file="tests/test_basic.py",
        function_name="test_immutable",
    )

    text = result.text
    assert "from pkg.basic import Basic" in text
    assert "b1 = Basic()" in text
    assert "assert not hasattr(b1, '__dict__')" in text
    assert "noise = 123" not in text
    assert "from other import unused" not in text
    assert result.mode == "static_backward_cross_scope"


class _FakeGraph:
    def __init__(self, repo):
        self.repo = repo

    def snippet(self, ref, radius=12):
        path, entity = ref.split("::", 1)
        source = (self.repo / path).read_text()
        return f"{ref}\n{source}"


def test_production_expansion_follows_inheritance(tmp_path):
    tests = tmp_path / "tests/test_basic.py"
    basic = tmp_path / "pkg/basic.py"
    printable = tmp_path / "pkg/printable.py"

    tests.parent.mkdir(parents=True)
    basic.parent.mkdir(parents=True)

    tests.write_text(
        "from pkg.basic import Basic\n"
        "b1 = Basic()\n"
        "def test_immutable():\n"
        "    assert not hasattr(b1, '__dict__')\n"
    )
    basic.write_text(
        "from .printable import Printable\n"
        "class Basic(Printable):\n"
        "    __slots__ = ('x',)\n"
    )
    printable.write_text(
        "class Printable:\n"
        "    pass\n"
    )

    result = backward_slice_python(
        tests.read_text(),
        source_file="tests/test_basic.py",
        function_name="test_immutable",
    )

    deps = expand_production_dependencies(
        tmp_path,
        _FakeGraph(tmp_path),
        result,
    )

    entities = [dep.entity for dep in deps]
    assert "pkg/basic.py::Basic" in entities
    assert "pkg/printable.py::Printable" in entities
