"""Tests for leakage-safe paired LLM action-outcome contract."""
import unittest

from aitech.paired_action_outcomes import validate_pairs


def pair(project="Lang", split="train", outcome=1, action_id="a"):
    return {
        "project": project, "bug": project + "-1", "split": split,
        "model_id": "same-small-LLM@revision", "seed": 42,
        "decoding_hash": "decoding-same", "failure_sha256": "failure-same",
        "baseline_prompt_sha256": "prompt-before",
        "action_prompt_sha256": "prompt-after",
        "baseline_prediction_sha256": "prediction-before",
        "action_prediction_sha256": "prediction-after",
        "actual_baseline_input_tokens": 200,
        "actual_action_input_tokens": 210,
        "common_input_token_ceiling": 300,
        "action": {
            "action_id": action_id, "kind": "UPGRADE_L1",
            "target_id": "pkg.F.method()",
            "incremental_input_tokens": 10,
            "incremental_latency_ms": 50,
            "source_revision": "buggy-source-commit",
            "provenance": "java-source-and-test",
            "failure_relevance": 0.6,
        },
        "outcome": outcome,
        "outcome_definition": "defects4j_changed_class_hit_at_1",
        "gold_scoring_only": True,
        "predictions_frozen_before_gold": True,
    }


class TestPairedActionOutcomes(unittest.TestCase):
    def test_valid_outcomes_from_separate_projects(self):
        rows = [pair("Lang", "train"), pair("Time", "calibration", -1),
                pair("Math", "test", 0)]
        report = validate_pairs(rows)
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["projects"], 3)
        self.assertEqual(report["label_counts"]["test"]["0"], 1)

    def test_same_project_in_train_and_test_rejected(self):
        second = pair("Time", "test")
        second["project"] = "Lang"
        with self.assertRaisesRegex(ValueError, "split leakage"):
            validate_pairs([pair(), second])

    def test_mixed_decoding_settings_are_rejected(self):
        other = pair("Time", "calibration", -1)
        other["decoding_hash"] = "different-temperature"
        with self.assertRaisesRegex(ValueError, "decoding"):
            validate_pairs([pair(), other])

    def test_no_fabricated_freeze(self):
        value = pair()
        value["predictions_frozen_before_gold"] = False
        with self.assertRaisesRegex(ValueError, "freeze-before-scoring"):
            validate_pairs([value])

    def test_cannot_treat_noop_as_action(self):
        value = pair()
        value["action_prompt_sha256"] = value["baseline_prompt_sha256"]
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_pairs([value])

    def test_budget_is_on_actual_full_prompt(self):
        value = pair()
        value["actual_action_input_tokens"] = 301
        with self.assertRaisesRegex(ValueError, "exceeds common"):
            validate_pairs([value])

    def test_strict_action_features_reject_oracle_labels(self):
        value = pair()
        value["action"]["gold_method_is_present"] = True
        with self.assertRaisesRegex(ValueError, "unknown action feature"):
            validate_pairs([value])

    def test_unverified_graph_action_is_rejected(self):
        value = pair()
        value["action"]["kind"] = "GRAPH"
        value["action"]["requires_call_binding"] = True
        value["action"]["compiler_binding_verified"] = False
        with self.assertRaisesRegex(ValueError, "unverified"):
            validate_pairs([value])

    def test_graph_cannot_bypass_binding_by_omitting_flag(self):
        value = pair()
        value["action"]["kind"] = "GRAPH"
        value["action"]["requires_call_binding"] = False
        value["action"]["compiler_binding_verified"] = False
        with self.assertRaisesRegex(ValueError, "unverified"):
            validate_pairs([value])

    def test_empty_outcome_data_cannot_train(self):
        with self.assertRaisesRegex(ValueError, "actual paired"):
            validate_pairs([])


if __name__ == "__main__":
    unittest.main()
