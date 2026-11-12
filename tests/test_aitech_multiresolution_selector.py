"""Multi-resolution allocation safeguards and identity/fidelity distinctions."""
import unittest

from aitech.multiresolution_selector import select_tiered


def method(name, cost_l0, cost_l1, prior=0.7, relevance=0.6):
    return {
        "function_id": name,
        "prior": prior,
        "features": {
            "l0_tokens": cost_l0,
            "l1_tokens": cost_l1,
            "failure_relevance": relevance,
            "trace_match": 0.0,
            "ranker_agreement": 0.5,
        },
    }


class TestMultiResolutionSelector(unittest.TestCase):
    def test_expensive_anchor_keeps_identity_without_forcing_expensive_code(self):
        items = [
            method("pkg.Demo.failure(String)", 30, 560),
            method("pkg.Demo.helper()", 25, 100),
            method("pkg.Demo.other()", 20, 70),
        ]
        result = select_tiered(items, "failure(s) failed", 200)
        self.assertEqual(result.anchor_identities_retained, 1)
        self.assertEqual(result.anchors_total, 1)
        self.assertNotEqual(result.levels[0], 2)
        self.assertLessEqual(result.proxy_tokens, 200)

    def test_distinguishes_code_logic_from_identity(self):
        items = [method("pkg.A.fail()", 10, 70)]
        result = select_tiered(items, "fail() failed", 10)
        self.assertEqual(result.anchor_identities_retained, 1)
        self.assertEqual(result.anchor_l1_retained, 0)
        self.assertEqual(result.l0_count, 1)

    def test_upgrades_anchor_when_room_exists(self):
        items = [method("pkg.A.fail()", 10, 40)]
        result = select_tiered(items, "fail() failed", 40)
        self.assertEqual(result.levels, (2,))
        self.assertEqual(result.anchor_l1_retained, 1)

    def test_cost_and_determinism(self):
        items = [
            method("pkg.A.x()", 15, 120),
            method("pkg.A.y()", 30, 80),
            method("pkg.A.z()", 20, 90),
        ]
        first = select_tiered(items, "x() failed", 130, edges={(0, 1)})
        self.assertEqual(first, select_tiered(items, "x() failed", 130,
                                             edges={(0, 1)}))
        self.assertLessEqual(first.proxy_tokens, 130)
        self.assertEqual(first.l0_count + first.l1_count,
                         sum(level > 0 for level in first.levels))

    def test_bad_edge_is_rejected(self):
        with self.assertRaises(ValueError):
            select_tiered([method("A.x()", 10, 20)], "", 100, edges={(0, 4)})


if __name__ == "__main__":
    unittest.main()
