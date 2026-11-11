"""No-model tests for frozen STOP vs L1-upgrade counterfactual design."""
import unittest

from aitech.cpu_action_counterfactual import freeze_upgrade_pair, score_pair


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Test-only deterministic proxy, NOT actual Qwen tokenizer.
        return text.split()

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return " ".join(x["role"] + ": " + x["content"] for x in messages) + " assistant:"

def candidate(name, l0, l1):
    return {"function_id": name,
            "levels": [l0, l1],
            "features": {"l0_tokens": 2, "l1_tokens": 3}}


class TestCpuActionCounterfactual(unittest.TestCase):
    def snapshot(self):
        return {
            "context": "failure() throws IllegalStateException",
            "candidates": [
                candidate("pkg.A.failure()", "failure signature",
                          "failure method body assertion error"),
                candidate("pkg.A.helper()", "helper signature",
                          "helper implementation with extra detail"),
                candidate("pkg.A.other()", "other signature",
                          "other implementation with many tokens"),
            ],
        }

    def test_frozen_prompts_use_same_failure_and_candidate_indices(self):
        frozen = freeze_upgrade_pair(self.snapshot(), TinyTokenizer(),
                                     evidence_budget=16, max_total_input=200)
        self.assertIsNone(frozen["skip_reason"])
        self.assertEqual(frozen["target_method"], "pkg.A.failure()")
        self.assertEqual([x["condition"] for x in frozen["prompts"]],
                         ["STOP", "UPGRADE_L1"])
        self.assertNotEqual(frozen["prompts"][0]["prompt_sha256"],
                            frozen["prompts"][1]["prompt_sha256"])
        self.assertEqual(set(frozen["prompts"][0]["selected_levels"]),
                         set(frozen["prompts"][1]["selected_levels"]))
        self.assertLessEqual(max(x["actual_total_input_tokens"]
                                 for x in frozen["prompts"]), 200)

    def test_skip_if_no_room_for_detail(self):
        frozen = freeze_upgrade_pair(self.snapshot(), TinyTokenizer(),
                                     evidence_budget=2, max_total_input=200)
        self.assertEqual(frozen["skip_reason"],
                         "no_positive_cost_L1_upgrade_fits_same_budget")

    def test_skip_if_common_input_ceiling_too_small(self):
        frozen = freeze_upgrade_pair(self.snapshot(), TinyTokenizer(),
                                     evidence_budget=16, max_total_input=10)
        self.assertEqual(frozen["skip_reason"],
                         "full_prompt_exceeds_matched_input_ceiling")

    def test_posthoc_outcome_uses_prediction_not_selected_gold(self):
        result = score_pair({
            "bug": "Lang-1", "skip_reason": None,
            "predictions": [
                {"condition": "STOP", "predicted_method": "pkg.A.helper()",
                 "actual_total_input_tokens": 90, "generation_seconds": 1.0},
                {"condition": "UPGRADE_L1", "predicted_method": "pkg.A.failure()",
                 "actual_total_input_tokens": 95, "generation_seconds": 1.2},
            ],
        }, ["pkg.A.failure()"])
        self.assertEqual(result["outcome"], 1)
        self.assertTrue(result["upgraded_correct"])

    def test_invalid_condition_order_rejected(self):
        with self.assertRaises(ValueError):
            score_pair({"bug": "Lang-1", "skip_reason": None,
                        "predictions": [{"condition": "UPGRADE_L1"},
                                        {"condition": "STOP"}]}, [])


if __name__ == "__main__":
    unittest.main()
