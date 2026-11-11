"""Offline, no Defects4J installation needed: verify checkout preflight safety."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from aitech.defects4j_compiler_pilot import resolve_metadata, run
from aitech.java_compiler_graph import ResolutionAudit, ResolvedCall


class TestDefects4JCompilerPilot(unittest.TestCase):
    def make_checkout(self, root):
        checkout = Path(root)
        (checkout / ".defects4j.config").write_text("pid=Lang\n", encoding="utf-8")
        (checkout / "src").mkdir()
        (checkout / "src" / "Demo.java").write_text(
            "class Demo { void run() {} }", encoding="utf-8")
        return checkout

    def test_existing_buggy_checkout_classpath(self):
        with tempfile.TemporaryDirectory() as temp:
            checkout = self.make_checkout(temp)
            with patch("aitech.defects4j_compiler_pilot.export",
                       side_effect=lambda _checkout, key: {
                           "dir.src.classes": "src",
                           "cp.compile": "lib/a.jar:target/classes",
                       }[key]):
                meta = resolve_metadata(checkout)
            self.assertEqual(meta["compile_classpath_entries"], 2)
            self.assertIn(str(checkout / "lib/a.jar"), meta["compile_classpath"])

    def test_maps_compiler_targets_into_frozen_candidate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            checkout = self.make_checkout(temp)
            snapshot = checkout / "candidates.json"
            snapshot.write_text(json.dumps({"candidates": [
                {"function_id": "Demo.run()"},
                {"function_id": "Demo.helper()"},
            ]}), encoding="utf-8")
            fake = ResolutionAudit(
                calls=(ResolvedCall("Demo.java", "Demo", "run", "METHOD",
                                    "Demo", "helper", ""),),
                unresolved_count=0, source_files=1,
                source_hashes={"Demo.java": "testhash"},
            )
            with patch("aitech.defects4j_compiler_pilot.export",
                       side_effect=lambda _checkout, key: {
                           "dir.src.classes": "src", "cp.compile": "lib/example.jar",
                       }[key]):
                with patch("aitech.defects4j_compiler_pilot.audit_source_directory",
                           return_value=fake):
                    result = run(checkout, candidate_snapshot=snapshot)
            self.assertEqual(result["candidate_mapping"]["edges"], [[0, 1]])
            self.assertEqual(result["candidate_mapping"]["mapped_calls"], 1)

    def test_missing_checkout_rejected_before_external_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "existing, valid"):
                resolve_metadata(Path(temp))

    def test_disallows_partial_graph_above_file_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            checkout = self.make_checkout(temp)
            with patch("aitech.defects4j_compiler_pilot.export",
                       side_effect=lambda _checkout, key: {
                           "dir.src.classes": "src", "cp.compile": "lib/example.jar",
                       }[key]):
                with self.assertRaisesRegex(RuntimeError, "file limit"):
                    run(checkout, max_files=0)

    def test_passes_classpath_and_records_static_targets_only(self):
        with tempfile.TemporaryDirectory() as temp:
            checkout = self.make_checkout(temp)
            fake = ResolutionAudit(
                calls=(ResolvedCall("Demo.java", "Demo", "run", "METHOD",
                                    "Demo", "helper", ""),),
                unresolved_count=0, source_files=1,
                source_hashes={"Demo.java": "hash"},
            )
            with patch("aitech.defects4j_compiler_pilot.export",
                       side_effect=lambda _checkout, key: {
                           "dir.src.classes": "src",
                           "cp.compile": "build/classes",
                       }[key]):
                with patch("aitech.defects4j_compiler_pilot.audit_source_directory",
                           return_value=fake) as spy:
                    result = run(checkout, max_files=10)
            self.assertEqual(result["resolved_static_calls"], 1)
            self.assertFalse(result["gold_patch_used"])
            self.assertIn("build/classes", spy.call_args.kwargs["classpath"])


if __name__ == "__main__":
    unittest.main()
