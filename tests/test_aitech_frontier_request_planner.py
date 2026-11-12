"""Tests for failure-prioritized missing Java definition retrieval requests."""
import unittest

from aitech.frontier_request_planner import plan_missing_definitions


def item(signature, calls):
    return {"function_id": signature,
            "levels": ["", "\n".join("CALL " + c for c in calls)]}


class TestFrontierRequestPlanner(unittest.TestCase):
    def test_prioritizes_failure_anchor_missing_method(self):
        snapshot = {
            "context": "test failed at Partial(...), then Partial()",
            "candidates": [
                item("pkg.Partial.Partial()", [
                    "Helpers.verify(x)", "Helpers.verify(x)"]),
                item("pkg.Unrelated.m()", ["Helpers.extra(x)"]),
                item("pkg.Helpers.already(String)", []),
            ],
        }
        requests = plan_missing_definitions(snapshot)
        self.assertEqual(requests[0].owner, "pkg.Helpers")
        self.assertEqual(requests[0].method, "verify")
        self.assertTrue(requests[0].failure_anchor_caller)
        self.assertEqual(requests[0].evidence_callers, (0,))
        self.assertEqual(requests[0].binding_status,
                         "unverified_receiver_name_and_arity")

    def test_does_not_request_methods_already_in_pool(self):
        snapshot = {
            "context": "run()",
            "candidates": [
                item("pkg.Caller.run()", ["Helpers.verify(x)"]),
                item("pkg.Helpers.verify(String)", []),
            ],
        }
        self.assertEqual(plan_missing_definitions(snapshot), ())

    def test_abstains_on_simple_class_name_collision(self):
        snapshot = {
            "context": "run()",
            "candidates": [
                item("pkg.Caller.run()", ["Helpers.verify(x)"]),
                item("a.Helpers.other()", []),
                item("b.Helpers.other()", []),
            ],
        }
        self.assertEqual(plan_missing_definitions(snapshot), ())

    def test_request_limit_and_negative_limit(self):
        snapshot = {
            "context": "run()",
            "candidates": [
                item("pkg.Caller.run()", ["Helpers.verify(x)", "Helpers.check(x)"]),
                item("pkg.Helpers.other()", []),
            ],
        }
        self.assertEqual(len(plan_missing_definitions(snapshot, max_requests=1)), 1)
        with self.assertRaises(ValueError):
            plan_missing_definitions(snapshot, max_requests=-1)


if __name__ == "__main__":
    unittest.main()
