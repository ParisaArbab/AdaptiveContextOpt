"""Evidence ablation tests; no Ollama or Defects4J dependency."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from rave_fl.compare_expanded import load_context
from rave_fl.failure_evidence import compress_failures


class FailureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_manual_structured_evidence_is_used_without_fallback(self):
        src = self.root / "time25_failure_evidence.txt"
        src.write_text("TEST: real failing test\nFAILURE: expected 4 actual 3", encoding="utf-8")
        context, meta = load_context(self.root, "Time-25", src)
        self.assertEqual(context, src.read_text())
        self.assertEqual(meta["source"], "structured_failure_file")
        self.assertEqual(meta["characters"], len(context))
        self.assertEqual(meta["sha256"], hashlib.sha256(context.encode()).hexdigest())

    def test_missing_or_empty_evidence_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_context(self.root, "Time-25", self.root / "missing.txt")
        empty = self.root / "empty.txt"
        empty.write_text(" \n")
        with self.assertRaisesRegex(ValueError, "Empty debugging evidence"):
            load_context(self.root, "Time-25", empty)

    def test_default_uses_original_flexfl_trigger_context(self):
        trigger = self.root / "data" / "input" / "trigger_tests" / "Defects4J"
        trigger.mkdir(parents=True)
        (trigger / "Time-25.txt").write_text("original short trigger")
        context, meta = load_context(self.root, "Time-25")
        self.assertEqual(context, "\noriginal short trigger")
        self.assertEqual(meta["source"], "flexfl_original")

    def test_compressor_keeps_assertion_and_project_frames(self):
        raw = (
            "--- org.joda.time.TestA::fails\n"
            "junit.framework.ComparisonFailure: expected +04 but got +03\n"
            "\tat junit.framework.Assert.assertEquals(Assert.java:1)\n"
            "\tat org.joda.time.TestA.fails(TestA.java:42)\n"
            "--- org.joda.time.TestB::fails\n"
            "junit.framework.ComparisonFailure: expected +04 but got +03\n"
            "\tat org.joda.time.TestB.fails(TestB.java:90)\n"
        )
        result = compress_failures(raw)
        self.assertEqual(result.count("TEST:"), 2)
        self.assertIn("expected +04 but got +03", result)
        self.assertIn("TestA.java:42", result)
        self.assertNotIn("at junit.framework", result)

    def test_compressor_does_not_silently_accept_unsupported_format(self):
        with self.assertRaises(ValueError):
            compress_failures("not a Defects4J failing_tests file")
        with self.assertRaisesRegex(ValueError, "No ComparisonFailure"):
            compress_failures("--- org.joda.time.TestA::fails\njava.lang.Error: failure")


if __name__ == "__main__":
    unittest.main()
