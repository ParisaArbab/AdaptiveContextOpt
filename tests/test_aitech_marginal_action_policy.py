"""Unit tests for calibrated, provenance-gated acquisition decisions."""
import unittest

from aitech.marginal_action_policy import (
    ActionEstimate, EvidenceAction, decide_next_action
)


def action(key="method", kind="UPGRADE_L1", tokens=15,
           requires_binding=False, verified=False, latency=100):
    return EvidenceAction(
        action_id=key, kind=kind, target_id="pkg.F.method()",
        incremental_input_tokens=tokens, incremental_latency_ms=latency,
        source_revision="buggy-commit-abc",
        provenance="frozen-test-and-repo",
        requires_call_binding=requires_binding,
        compiler_binding_verified=verified,
    )


class FakeCalibratedEstimator:
    calibrated_on_independent_projects = True

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, candidate):
        gain, harm = self.predictions[candidate.action_id]
        return ActionEstimate(candidate.action_id, gain, harm)


class FakeUncalibratedEstimator(FakeCalibratedEstimator):
    calibrated_on_independent_projects = False


class TestMarginalActionPolicy(unittest.TestCase):
    def test_uncalibrated_or_missing_estimator_abstains(self):
        for model in (None, FakeUncalibratedEstimator({"a": (0.9, 0.0)})):
            outcome = decide_next_action([action("a")], model, 100)
            self.assertEqual(outcome.decision, "STOP_UNCALIBRATED")
            self.assertIsNone(outcome.action_id)

    def test_harmful_expansion_stops(self):
        estimator = FakeCalibratedEstimator({"a": (0.1, 0.5)})
        outcome = decide_next_action([action("a")], estimator, 100)
        self.assertEqual(outcome.decision, "STOP_NONPOSITIVE_VALUE")

    def test_budget_prevents_expansion_even_with_high_gain(self):
        estimator = FakeCalibratedEstimator({"a": (1.0, 0.0)})
        outcome = decide_next_action([action("a", tokens=101)], estimator, 100)
        self.assertEqual(outcome.decision, "STOP_NO_ELIGIBLE_ACTIONS")
        self.assertIn(("a", "token_budget_exceeded"), outcome.rejection_reasons)

    def test_unverified_call_cannot_acquire(self):
        estimator = FakeCalibratedEstimator({"a": (1.0, 0.0)})
        result = decide_next_action(
            [action("a", kind="GRAPH", requires_binding=True)],
            estimator, 100)
        self.assertEqual(result.decision, "STOP_NO_ELIGIBLE_ACTIONS")
        self.assertIn(("a", "unverified_call_binding"), result.rejection_reasons)

    def test_graph_requires_verified_binding_even_if_caller_omits_flag(self):
        estimator = FakeCalibratedEstimator({"graph": (1.0, 0.0)})
        decision = decide_next_action(
            [action("graph", kind="GRAPH", requires_binding=False,
                    verified=False)], estimator, 100)
        self.assertEqual(decision.decision, "STOP_NO_ELIGIBLE_ACTIONS")
        self.assertIn(("graph", "unverified_call_binding"),
                      decision.rejection_reasons)

    def test_best_positive_utility_selected(self):
        estimator = FakeCalibratedEstimator({
            "expensive": (0.6, 0.1),
            "cheap": (0.5, 0.05),
        })
        options = [action("expensive", tokens=150),
                   action("cheap", tokens=10)]
        result = decide_next_action(options, estimator, 200)
        self.assertEqual(result.decision, "ACQUIRE")
        self.assertEqual(result.action_id, "cheap")
        self.assertEqual(result.eligible_action_count, 2)

    def test_ties_are_deterministic(self):
        estimator = FakeCalibratedEstimator({"b": (0.6, 0.1),
                                             "a": (0.6, 0.1)})
        result = decide_next_action(
            [action("b"), action("a")], estimator, 200)
        self.assertEqual(result.action_id, "a")

    def test_improve_harm_cannot_exceed_one(self):
        estimator = FakeCalibratedEstimator({"a": (0.8, 0.6)})
        with self.assertRaisesRegex(ValueError, "exclusive"):
            decide_next_action([action("a")], estimator, 100)

    def test_duplicate_actions_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            decide_next_action([action("a"), action("a")], None, 100)


if __name__ == "__main__":
    unittest.main()
