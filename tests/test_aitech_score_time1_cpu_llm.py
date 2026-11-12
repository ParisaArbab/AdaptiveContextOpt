"""No-model tests for clean post-selection CPU LLM scoring."""
import unittest

from aitech.score_time1_cpu_llm import evaluate


class TestScoreCPULocalization(unittest.TestCase):
    def make_data(self):
        pred = {
            "gold_patch_loaded": False,
            "model_id": "test-model",
            "model_revision_sha": "abc",
            "rows": [],
        }
        for method, predid, support, reuse in (
            ("anchor_first", "pkg.A.bad()", False, False),
            ("force_verified_source", "pkg.A.other()", True, False),
            ("gate_unverified_binding", "pkg.A.bad()", False, True),
        ):
            pred["rows"].append({
                "condition": method,
                "max_input_tokens": 100,
                "input_token_count": 90,
                "selected_candidate_ids": ["pkg.A.bad()", "pkg.A.other()"],
                "predicted_candidate_id": predid,
                "source_helper_included": support,
                "inference_seconds": 0.3,
                "reused_identical_prompt": reuse,
            })
        labels = {
            "scoring_only": True,
            "changed_method_candidate_ids": ["pkg.A.bad()"],
            "label_source": "official.patch",
        }
        return pred, labels

    def test_hit_impact_and_separate_label_loading(self):
        pred, labels = self.make_data()
        result = evaluate(pred, labels)
        self.assertEqual(result["paired"][0]["model_hit_delta_forced_minus_anchor"], -1)
        self.assertTrue(result["selected_before_gold_scoring"])

    def test_does_not_accept_predictions_outside_prompt(self):
        pred, labels = self.make_data()
        pred["rows"][0]["predicted_candidate_id"] = "pkg.A.unlisted()"
        with self.assertRaisesRegex(ValueError, "outside"):
            evaluate(pred, labels)

    def test_gold_leakage_is_rejected(self):
        pred, labels = self.make_data()
        pred["gold_patch_loaded"] = True
        with self.assertRaisesRegex(ValueError, "no-gold"):
            evaluate(pred, labels)

    def test_gate_must_match_baseline_prediction(self):
        pred, labels = self.make_data()
        pred["rows"][2]["predicted_candidate_id"] = "pkg.A.other()"
        with self.assertRaisesRegex(ValueError, "Abstain"):
            evaluate(pred, labels)


if __name__ == "__main__":
    unittest.main()
