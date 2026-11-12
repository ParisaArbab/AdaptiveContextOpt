"""Test real-checkout snapshot parsing without inventing benchmark results."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aitech.live_defects4j_snapshot import (
    _methods, compact_failure_text, extract_snapshot, run_failed_tests
)


class TestLiveDefects4jSnapshot(unittest.TestCase):
    def test_extracts_realistic_java_methods_and_provenance(self):
        source = """package org.example;
public class Widget {
  public boolean check(int value) {
    return value > 2;
  }
  public String describe() {
    return "ok";
  }
}
"""
        methods = _methods(source, "src/org/example/Widget.java",
                           "org.example.Widget")
        ids = [m["function_id"] for m in methods]
        self.assertIn("org.example.Widget.check(int)", ids)
        self.assertIn("org.example.Widget.describe()", ids)
        self.assertTrue(all(m["source_sha256"] for m in methods))

    def test_compaction_keeps_actual_failure_and_trace(self):
        text = "unrelated log\nFailing tests: 1\n" + (
            "progress report\n" * 200) + (
            "java.lang.AssertionError: mismatch\n"
            "at org.example.Widget.check(Widget.java:10)\n")
        result = compact_failure_text(text, max_chars=1000)
        self.assertIn("Failing tests: 1", result)
        self.assertIn("AssertionError", result)
        self.assertIn("org.example.Widget.check", result)
        self.assertLessEqual(len(result), 1000)

    def test_pre_gold_snapshot_from_checkout_like_tree(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / ".defects4j.config").write_text("project=Example\n")
            src = root / "src"
            src.mkdir()
            (src / "Widget.java").write_text(
                "package org.example;\n"
                "public class Widget {\n"
                "  public int check(int value) { return value + 1; }\n"
                "  public int helper() { return 1; }\n"
                "}\n")
            failure = ("Failing tests: 1\n"
                       "java.lang.AssertionError\n"
                       "at org.example.Widget.check(Widget.java:3)\n")
            with patch("aitech.live_defects4j_snapshot.export",
                       return_value="src"):
                output = extract_snapshot(root, failure, limit=5)
            self.assertFalse(output["provenance"]["gold_patches_loaded"])
            self.assertGreater(len(output["candidates"]), 0)
            self.assertEqual(output["provenance"]["frame_count"], 1)
            self.assertTrue(any(
                x["function_id"].endswith(".check(int)")
                for x in output["candidates"]))
            self.assertIn("Failing tests: 1", output["context"])

    def test_reads_real_defects4j_failing_tests_trace(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)
            (path / "failing_tests").write_text(
                "--- org.example.WidgetTest::failure\\n"
                "java.lang.AssertionError\\n"
                "at org.example.Widget.check(Widget.java:20)\\n")
            class Completed:
                stdout = "Failing tests: 1\\n"
                stderr = ""
                returncode = 0
            with patch("aitech.live_defects4j_snapshot.subprocess.run",
                       return_value=Completed()):
                result = run_failed_tests(path)
            self.assertIn("Widget.check(Widget.java:20)", result)
            self.assertIn("Failing tests: 1", result)

    def test_candidate_file_diversity_before_gold(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / ".defects4j.config").write_text("project=Example\\n")
            source_dir = root / "src"
            source_dir.mkdir()
            for name in ("Alpha", "Beta", "Gamma"):
                methods = "\\n".join(
                    f" public int m{i}() {{ return {i}; }}"
                    for i in range(8))
                (source_dir / f"{name}.java").write_text(
                    f"package org.example;\\nclass {name} {{\\n" +
                    methods + "\\n}\\n")
            with patch("aitech.live_defects4j_snapshot.export",
                       return_value="src"):
                result = extract_snapshot(
                    root, "Failing tests: 1\\n", limit=9)
            from collections import Counter
            per_file = Counter(x["source_path"] for x in result["candidates"])
            self.assertEqual(len(result["candidates"]), 9)
            self.assertTrue(all(n <= 3 for n in per_file.values()))
            self.assertEqual(len(per_file), 3)

    def test_checkout_required(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError, "Defects4J"):
                extract_snapshot(Path(t), "Failing tests: 1")


if __name__ == "__main__":
    unittest.main()
