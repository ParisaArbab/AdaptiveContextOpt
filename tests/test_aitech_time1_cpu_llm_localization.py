"""No-download tests for exact-prompt budget selection and safe output parsing."""
import unittest

from aitech.time1_cpu_llm_localization import (
    _extract_prediction, select_prompt,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "\n".join(m["content"] for m in messages)
        text += "\nASSISTANT:" if add_generation_prompt else ""
        return text.split() if tokenize else text


class TestCPUModelLocalization(unittest.TestCase):
    def fixture(self):
        snapshot = {
            "context": "test failed in execute()",
            "candidates": [
                {"function_id": "pkg.A.execute()",
                 "levels": ["", "METHOD: pkg.A.execute()\nreturn other();"]},
                {"function_id": "pkg.A.other()",
                 "levels": ["", "METHOD: pkg.A.other()\nreturn true;"]},
            ],
        }
        helper = {
            "target": "pkg.A.helper()",
            "method_source": "boolean helper() { return true; }",
        }
        return snapshot, helper

    def test_helper_is_not_eligible_prediction_id(self):
        snapshot, helper = self.fixture()
        result = select_prompt(FakeTokenizer(), snapshot, helper,
                               max_input_tokens=500, force_helper=True)
        self.assertTrue(result["source_helper_included"])
        self.assertEqual(len(result["selected_candidate_ids"]), 2)
        self.assertNotIn(helper["target"], result["selected_candidate_ids"])
        self.assertLessEqual(result["input_token_count"], 500)

    def test_gate_and_baseline_are_exactly_identical(self):
        snapshot, helper = self.fixture()
        a = select_prompt(FakeTokenizer(), snapshot, helper,
                          max_input_tokens=500, force_helper=False)
        b = select_prompt(FakeTokenizer(), snapshot, helper,
                          max_input_tokens=500, force_helper=False)
        self.assertEqual(a["prompt_sha256"], b["prompt_sha256"])

    def test_empty_prompt_budget_fails_closed(self):
        snapshot, helper = self.fixture()
        with self.assertRaises(ValueError):
            select_prompt(FakeTokenizer(), snapshot, helper,
                          max_input_tokens=1, force_helper=False)

    def test_inference_predictions_must_identify_selected_method(self):
        valid = ["pkg.A.execute()", "pkg.A.other()"]
        self.assertEqual(_extract_prediction("pkg.A.other()", valid),
                         "pkg.A.other()")
        self.assertIsNone(_extract_prediction("pkg.A.helper()", valid))


if __name__ == "__main__":
    unittest.main()
