"""Tests for multinomial marginal-outcome trainer and artifact inference."""
import unittest

from aitech.marginal_action_policy import (
    EvidenceAction, ActionEstimate, decide_next_action, FEATURE_NAMES
)
from aitech.train_marginal_outcomes import (
    LearnedMarginalEstimator, train
)


def candidate():
    return EvidenceAction(
        action_id="upgrade", kind="UPGRADE_L1", target_id="pkg.A.foo()",
        incremental_input_tokens=20, incremental_latency_ms=100,
        source_revision="buggy-revision", provenance="verified-src",
        failure_relevance=0.8, semantic_similarity=0.7,
        rank_fraction=0.3, failure_anchor=True,
        context_fill_fraction=0.5,
    )


class TestMarginalOutcomeTraining(unittest.TestCase):
    def test_no_data_cannot_be_called_a_trained_controller(self):
        with self.assertRaisesRegex(ValueError, "actual paired"):
            train([])

    def test_tiny_calibration_is_rejected_before_sklearn_import(self):
        def row(project, split, label):
            return {
                "project": project, "bug": project + "-1", "split": split,
                "model_id": "same-model@sha", "seed": 42,
                "decoding_hash": "constant",
                "failure_sha256": "failure",
                "baseline_prompt_sha256": "before",
                "action_prompt_sha256": "after",
                "baseline_prediction_sha256": "pred0",
                "action_prediction_sha256": "pred1",
                "actual_baseline_input_tokens": 200,
                "actual_action_input_tokens": 210,
                "common_input_token_ceiling": 300,
                "action": {
                    "action_id": "upgrade", "kind": "UPGRADE_L1",
                    "target_id": "pkg.A.foo()",
                    "incremental_input_tokens": 10,
                    "incremental_latency_ms": 100,
                    "source_revision": "buggy-revision",
                    "provenance": "verified"
                },
                "outcome": label,
                "outcome_definition": "defects4j_changed_class_hit_at_1",
                "gold_scoring_only": True,
                "predictions_frozen_before_gold": True,
            }
        rows = [
            row("Lang", "train", 1), row("Math", "train", 0),
            row("Time", "calibration", -1), row("Closure", "test", 0),
        ]
        with self.assertRaisesRegex(ValueError, "insufficient independent calibration"):
            train(rows)

    def test_artifact_predictions_sum_to_one_and_do_not_use_gold(self):
        d = len(FEATURE_NAMES)
        artifact = {
            "independent_project_calibration_verified": True,
            "feature_names": list(FEATURE_NAMES),
            "mean": [0.0] * d, "scale": [1.0] * d,
            "coef": [[0.0] * d for _ in range(3)],
            "intercept": [-1.0, 0.0, 1.0],
            "temperature": 1.0, "classes": [-1, 0, 1],
        }
        model = LearnedMarginalEstimator(artifact)
        estimate = model.predict(candidate())
        self.assertIsInstance(estimate, ActionEstimate)
        self.assertGreater(estimate.p_improve, estimate.p_harm)
        self.assertLessEqual(estimate.p_improve + estimate.p_harm, 1.0)
        decision = decide_next_action([candidate()], model, 100)
        self.assertEqual(decision.decision, "ACQUIRE")

    def test_uncalibrated_artifact_abstains(self):
        model = LearnedMarginalEstimator({
            "independent_project_calibration_verified": False
        })
        self.assertEqual(decide_next_action([candidate()], model, 100).decision,
                         "STOP_UNCALIBRATED")


if __name__ == "__main__":
    unittest.main()
