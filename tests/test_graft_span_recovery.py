"""Regression tests for narrowly corrected FlexFL source spans."""
from pathlib import Path

import pytest

from rave_fl.source_spans import extract_method


def source_tree(tmp_path, source):
    root = tmp_path / "checkout"
    path = root / "src/main/java/org/example/Fraction.java"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    return root


def test_recovers_unique_nearby_declaration_without_changing_source(tmp_path):
    source = "\n".join(
        [f"// header {i}" for i in range(404)]
        + ["    public Fraction negate() {", "       return this;", "    }"]
        + [f"// trailer {i}" for i in range(20)]
    )
    root = source_tree(tmp_path, source)
    function = "org.example.Fraction.negate()"
    info = {"file_class": "org.example.Fraction", "start_line": 406, "end_line": 415}
    path, body = extract_method(root, function, info)
    assert path.endswith("org/example/Fraction.java")
    assert "public Fraction negate()" in body
    assert (root / path).read_text() == source


def test_rejects_no_declaration_or_false_method_call(tmp_path):
    source = "\n".join(["// header"] * 405 + ["return negate();"] + ["// end"] * 20)
    root = source_tree(tmp_path, source)
    with pytest.raises(ValueError, match="nearby matching declarations: 0"):
        extract_method(
            root, "org.example.Fraction.negate()",
            {"file_class": "org.example.Fraction", "start_line": 407, "end_line": 415},
        )


def test_rejects_ambiguous_nearby_declarations(tmp_path):
    source = "\n".join(
        ["// header"] * 404
        + ["    public Fraction negate() {", "    }",
           "    private Fraction negate() {", "    }"]
        + ["// tail"] * 20
    )
    root = source_tree(tmp_path, source)
    with pytest.raises(ValueError, match="nearby matching declarations: 2"):
        extract_method(
            root, "org.example.Fraction.negate()",
            {"file_class": "org.example.Fraction", "start_line": 409, "end_line": 417},
        )
