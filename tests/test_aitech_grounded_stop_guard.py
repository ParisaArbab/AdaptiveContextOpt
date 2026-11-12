"""Risk-gated source-evidence selection never uses gold or model scores."""
import unittest

from aitech.grounded_stop_guard import decide_pre_gold


def record(name, description):
    return {"function_id": name,
            "levels": [name, name + "\n" + description]}


class TestGroundedStopGuard(unittest.TestCase):
    def test_java_production_frame_protects_from_unneeded_upgrade(self):
        snap = {
            "context": "java.lang.AssertionError\nat pkg.Core.check(Core.java:19)",
            "candidates": [
                record("pkg.Helper.a()", "unrelated()"),
                record("pkg.Core.check()", "throws IllegalStateException;"),
            ]
        }
        out = decide_pre_gold(snap, shortlist_size=2)
        self.assertEqual(out.action, "STOP")
        self.assertEqual(out.reason, "selected_production_trace_class")

    def test_test_name_class_is_a_weak_stop_clue(self):
        snap = {
            "context": "--- pkg.junit.RendererTests::testFailure\n"
                       "junit.framework.AssertionFailedError",
            "candidates": [
                record("pkg.Other.help()", "helper"),
                record("pkg.Renderer.draw()", "draw image"),
            ],
        }
        out = decide_pre_gold(snap, shortlist_size=2)
        self.assertEqual(out.action, "STOP")
        self.assertEqual(out.reason, "selected_test_name_class_clue")

    def test_missing_source_match_allows_upgrade_action(self):
        snap = {
            "context": "generic failure, no production stack",
            "candidates": [
                record("pkg.Other.help()", "helper"),
                record("pkg.Renderer.draw()", "draw image"),
            ],
        }
        out = decide_pre_gold(snap, shortlist_size=2)
        self.assertEqual(out.action, "UPGRADE_L1")
        self.assertTrue(out.target_method)

    def test_no_gold_data_or_model_dependency(self):
        snap = {
            "context": "--- pkg.junit.RendererTests::testFailure",
            "candidates": [
                record("pkg.Renderer.draw()", "draw"),
                record("pkg.Other.help()", "help"),
            ],
        }
        snapshot_before = repr(snap)
        self.assertEqual(decide_pre_gold(snap, shortlist_size=2).action, "STOP")
        self.assertEqual(repr(snap), snapshot_before)


if __name__ == "__main__":
    unittest.main()
