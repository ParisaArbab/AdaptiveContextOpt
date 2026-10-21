"""Synthetic fixtures mirroring the real FlexFL CSV format and source ranges."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rave_fl.expanded_pool import union_pool
from rave_fl.source_spans import extract_pool_sources, extract_method


class ExpandedPoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for method in ("SBIR", "Ochiai", "BoostN"):
            folder = self.root / "data" / "FL_results" / method / "Defects4J"
            folder.mkdir(parents=True)
            with (folder / "Time-25_method-susps.csv").open("w", newline="") as out:
                w = csv.DictWriter(out, fieldnames=["File", "Signature", "StartLine", "EndLine"])
                w.writeheader()
                rows = [
                    {"File": "org.sample.Foo", "Signature": "ok()",
                     "StartLine": "1", "EndLine": "2"},
                    {"File": "org.sample.Foo", "Signature": "broken(int)",
                     "StartLine": "3", "EndLine": "4"},
                ]
                w.writerows(rows)
        folder = self.root / "src" / "main" / "java" / "org" / "sample"
        folder.mkdir(parents=True)
        (folder / "Foo.java").write_text(
            "void ok() {\n  return;\n"
            "void broken(int i) {\n  throw new IllegalStateException();\n"
        )

    def test_union_real_csv_ranges(self):
        candidates, metadata = union_pool(self.root, "Time-25", 10)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(metadata["org.sample.Foo.broken(int)"]["start_line"], 3)
        self.assertEqual(len(metadata["org.sample.Foo.broken(int)"]["methods"]), 3)

    def test_source_extraction(self):
        _, metadata = union_pool(self.root, "Time-25", 10)
        bodies, paths = extract_pool_sources(self.root, metadata)
        self.assertIn("throw new", bodies["org.sample.Foo.broken(int)"])
        self.assertTrue(paths["org.sample.Foo.broken(int)"].endswith("Foo.java"))

    def test_inner_class_resolves_outer_java_file(self):
        folder = self.root / "src" / "main" / "java" / "org" / "sample"
        (folder / "Outer.java").write_text("class Outer {\n void method() {}\n}")
        path, body = extract_method(
            self.root, "org.sample.Outer.Inner.method()",
            {"file_class": "org.sample.Outer.Inner", "start_line": 2, "end_line": 2})
        self.assertIn("method()", body)
        self.assertTrue(path.endswith("Outer.java"))

    def test_source_mismatch_fails_loudly(self):
        _, metadata = union_pool(self.root, "Time-25", 10)
        bad = dict(metadata["org.sample.Foo.broken(int)"], start_line=1, end_line=2)
        with self.assertRaisesRegex(ValueError, "method name absent"):
            extract_method(self.root, "org.sample.Foo.broken(int)", bad)


if __name__ == "__main__":
    unittest.main()
