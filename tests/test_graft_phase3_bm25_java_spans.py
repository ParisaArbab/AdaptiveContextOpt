"""Regression tests for Java BM25 candidate source spans.

Pure local tests, no Defects4J, no Ollama, no Git checkout or model calls.
"""
from types import SimpleNamespace
from pathlib import Path

import pytest

from rave_fl.eval import representations as reprs
from rave_fl.eval.schema import CompatibilityError, CodeEntity


JAVA = (
    "package org.example;\n"
    "public class Example {\n"
    "  public int add(int x) { return x + 1; }\n"
    "  public void test() { add(3); }\n"
    "}\n"
)


def source_setup(tmp_path):
    checkout = tmp_path / "buggy"
    path = checkout / "src/main/java/org/example/Example.java"
    path.parent.mkdir(parents=True)
    path.write_text(JAVA)
    instance = SimpleNamespace(
        checkout=str(checkout), source_root="src/main/java", language="java"
    )
    return instance, path


def test_bm25_java_method_pool_has_valid_int_spans(tmp_path):
    instance, path = source_setup(tmp_path)
    entities = reprs.index_source(instance, "method")
    assert len(entities) == 2
    for e in entities:
        assert isinstance(e, CodeEntity)
        assert type(e.start_line) is int
        assert type(e.end_line) is int
        assert 1 <= e.start_line <= e.end_line <= len(JAVA.splitlines())
        assert path.relative_to(Path(instance.checkout)).as_posix() == e.path
    assert any("add(" in e.entity_id for e in entities)


def test_java_ast_does_not_read_point_coordinates(tmp_path, monkeypatch):
    """Any Point.row access is forbidden because of tree-sitter 0.26.0."""
    instance, _ = source_setup(tmp_path)
    original = reprs.java_tree

    class NoPoint:
        def __init__(self, node):
            self.node = node
            self.type = node.type
            self.start_byte = node.start_byte
            self.end_byte = node.end_byte
            self.has_error = node.has_error

        @property
        def start_point(self):
            raise AssertionError("native Point.row must never be accessed")

        @property
        def end_point(self):
            raise AssertionError("native Point.row must never be accessed")

        @property
        def named_children(self):
            return [NoPoint(child) for child in self.node.named_children]

        def child_by_field_name(self, key):
            return self.node.child_by_field_name(key)

    monkeypatch.setattr(reprs, "java_tree", lambda source: NoPoint(original(source)))
    entities = reprs.index_source(instance, "method")
    assert len(entities) == 2
    assert all(type(e.start_line) is int and type(e.end_line) is int for e in entities)


def test_invalid_parser_byte_span_reports_file_and_candidate(tmp_path, monkeypatch):
    instance, _ = source_setup(tmp_path)
    original = reprs.java_tree

    class InvalidMethodNode:
        def __init__(self, node):
            self.node = node
            self.type = node.type
            self.start_byte = (
                999999 if node.type == "method_declaration" else node.start_byte
            )
            self.end_byte = node.end_byte
            self.has_error = node.has_error

        @property
        def named_children(self):
            return [InvalidMethodNode(child) for child in self.node.named_children]

        def child_by_field_name(self, key):
            return self.node.child_by_field_name(key)

    monkeypatch.setattr(
        reprs, "java_tree", lambda source: InvalidMethodNode(original(source))
    )
    with pytest.raises(
        CompatibilityError, match="Invalid Java AST method byte span"
    ) as err:
        reprs.index_source(instance, "method")
    assert "Example.java" in str(err.value)
    assert "add" in str(err.value)


def test_large_realistic_java_source_line_number_above_256(tmp_path, monkeypatch):
    """Native Point.row on non-immortal ints can corrupt memory in 0.26.0."""
    instance, path = source_setup(tmp_path)
    source = (
        "package org.example;\n"
        + ("// café and ☕\n" * 300)
        + "public class Example {\n"
        + "  public void lateMethod() { System.out.println(2); }\n"
        + "}\n"
    )
    path.write_text(source, encoding="utf-8")
    entities = reprs.index_source(instance, "method")
    assert len(entities) == 1
    method = entities[0]
    assert "lateMethod" in method.entity_id
    assert method.start_line == 303
    assert method.end_line == 303
    assert "System.out.println(2)" in method.source
    assert method.path == "src/main/java/org/example/Example.java"



def test_deep_java_ast_is_indexed_without_python_recursion(tmp_path, monkeypatch):
    """A deeply nested parser tree must not overflow recursive walk_java."""
    instance, _ = source_setup(tmp_path)
    original = reprs.java_tree
    root = original(JAVA)

    class Wrapper:
        def __init__(self, child):
            self.type = "block"
            self.named_children = [child]

    for _ in range(1600):
        root = Wrapper(root)

    monkeypatch.setattr(reprs, "java_tree", lambda source: root)
    entities = reprs.index_source(instance, "method")
    assert len(entities) == 2
    assert any("add(" in e.entity_id for e in entities)
    assert all(type(e.start_line) is int and e.start_line > 0 for e in entities)


def test_java_index_trace_is_opt_in(tmp_path, monkeypatch, capsys):
    instance, _ = source_setup(tmp_path)
    monkeypatch.setenv("GRAFT_TRACE_JAVA_INDEX", "1")
    entities = reprs.index_source(instance, "method")
    stderr = capsys.readouterr().err
    assert len(entities) == 2
    assert "[java-index] parsing " in stderr
    assert "[java-index] parsed " in stderr
    assert "[java-index] finished " in stderr
    assert "org/example/Example.java" in stderr
