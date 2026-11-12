"""Conservative pre-gold class-hint STOP gate tests."""
import unittest

from aitech.model_free_stop_gate import decide_model_free_stop


def fixture(test_name="pkg.WidgetTests::testCase"):
    return {
        "context": "Failing tests: 1\n" + test_name,
        "provenance": {"gold_patches_loaded": False},
        "candidates": [
            {"function_id": "pkg.Other.unrelated()",
             "source_path": "Other.java",
             "levels": ["Other.unrelated", "unrelated source"]},
            {"function_id": "pkg.Widget.run()",
             "source_path": "Widget.java",
             "levels": ["Widget.run", "run source"]},
        ],
    }


class TestModelFreeStopGate(unittest.TestCase):
    def test_exact_conventional_test_owner_can_stop_at_class_level(self):
        result = decide_model_free_stop(fixture())
        self.assertEqual(result.action, "STOP_CLASS_HINT")
        self.assertEqual(result.predicted_class, "pkg.Widget")
        self.assertEqual(result.model_input_tokens, 0)
        self.assertEqual(result.model_output_tokens, 0)

    def test_missing_conventional_test_name_escalates(self):
        result = decide_model_free_stop(fixture("failure in framework"))
        self.assertEqual(result.action, "ESCALATE")
        self.assertIsNone(result.predicted_class)

    def test_tainted_source_is_refused(self):
        data = fixture()
        data["provenance"]["gold_patches_loaded"] = True
        with self.assertRaisesRegex(ValueError, "pre-gold"):
            decide_model_free_stop(data)

    def test_owner_partial_match_escalates_by_default(self):
        data = fixture("another.pkg.WidgetTests::testCase")
        result = decide_model_free_stop(data)
        self.assertEqual(result.action, "ESCALATE")


if __name__ == "__main__":
    unittest.main()
