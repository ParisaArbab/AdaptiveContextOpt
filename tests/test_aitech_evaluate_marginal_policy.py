"""Offline policy ablation tests, synthetic fixtures never claimed as results."""
import unittest

from aitech.evaluate_marginal_policy import evaluate
from aitech.marginal_action_policy import FEATURE_NAMES


def fake_pair(project, split, bug_num, outcome, action_id="upgrade"):
    return {
        "project": project, "bug": f"{project}-{bug_num}", "split": split,
        "model_id": "Qwen-tiny@sha", "seed": 42,
        "decoding_hash": "greedy-24", "failure_sha256": f"{project}-{bug_num}-failure",
        "baseline_prompt_sha256": f"{project}-{bug_num}-before",
        "action_prompt_sha256": f"{project}-{bug_num}-after-{action_id}",
        "baseline_prediction_sha256": f"{project}-{bug_num}-pred0",
        "action_prediction_sha256": f"{project}-{bug_num}-pred1-{action_id}",
        "actual_baseline_input_tokens": 200,
        "actual_action_input_tokens": 210,
        "common_input_token_ceiling": 300,
        "action": {
            "action_id": action_id, "kind": "UPGRADE_L1",
            "target_id": "pkg.Class.method()", "incremental_input_tokens": 10,
            "incremental_latency_ms": 50, "source_revision": "buggy-sha",
            "provenance": "audited-repo-snapshot",
            "failure_relevance": 0.7,
        },
        "outcome": outcome,
        "outcome_definition": "defects4j_changed_class_hit_at_1",
        "gold_scoring_only": True,
        "predictions_frozen_before_gold": True,
    }


def fake_artifact():
    n = len(FEATURE_NAMES)
    return {
        "schema": 1,
        "independent_project_calibration_verified": True,
        "source_pairs_sha256": "frozen-training-data",
        "project_split": {
            "Lang": "train", "Time": "calibration", "Math": "test"},
        "model_id": "Qwen-tiny@sha",
        "outcome_definition": "defects4j_changed_class_hit_at_1",
        "common_input_token_ceiling": 300,
        "feature_names": list(FEATURE_NAMES),
        "mean": [0.0] * n, "scale": [1.0] * n,
        "coef": [[0.0] * n for _ in range(3)],
        "intercept": [-1.0, 0.0, 1.0], "temperature": 1.0,
        "classes": [-1, 0, 1],
    }


class TestEvaluateMarginalPolicy(unittest.TestCase):
    def rows(self):
        return [
            fake_pair("Lang", "train", 1, 1),
            fake_pair("Time", "calibration", 1, -1),
            fake_pair("Math", "test", 1, 1),
            fake_pair("Math", "test", 2, -1),
        ]

    def test_oracle_abstains_on_harm_and_baseline_never_acquires(self):
        result = evaluate(self.rows(), fake_artifact())
        policies = result["policies"]
        self.assertEqual(policies["stop_always"]["metrics"]["actions_acquired"], 0)
        self.assertEqual(policies["force_cheapest"]["metrics"]["harmed_bugs"], 1)
        self.assertEqual(policies["posthoc_oracle"]["metrics"]["harmed_bugs"], 0)
        self.assertEqual(policies["posthoc_oracle"]["metrics"]["helped_bugs"], 1)
        self.assertTrue(result["oracle_is_posthoc_not_deployable"])

    def test_only_test_projects_are_evaluated(self):
        result = evaluate(self.rows(), fake_artifact())
        self.assertEqual(result["policies"]["stop_always"]["metrics"]["bugs"], 2)

    def test_uncalibrated_model_is_rejected(self):
        artifact = fake_artifact()
        artifact["independent_project_calibration_verified"] = False
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            evaluate(self.rows(), artifact)

    def test_train_test_project_mislabeling_is_rejected(self):
        artifact = fake_artifact()
        artifact["project_split"]["Math"] = "train"
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate(self.rows(), artifact)

    def test_unmatched_baseline_in_same_bug_is_rejected(self):
        rows = self.rows()
        other = fake_pair("Math", "test", 1, 1, action_id="graph")
        other["action"]["kind"] = "RETRIEVE"
        other["baseline_prompt_sha256"] = "different-baseline"
        rows.append(other)
        with self.assertRaisesRegex(ValueError, "unmatched baseline"):
            evaluate(rows, fake_artifact())


if __name__ == "__main__":
    unittest.main()
