"""Test-name-to-production-class hints never require patch labels."""
import unittest

from aitech.test_name_anchored_reranker import (
    test_hints, hinted_shortlist, rank_with_test_hints
)


def snapshot():
    return {
        "context": (
            "Failing tests: 1\n"
            "- org.jfree.chart.renderer.category.junit."
            "AbstractCategoryItemRendererTests::test2947660\n"
            "junit.framework.AssertionFailedError\n"
            "at junit.framework.Assert.assertEquals(Assert.java:78)\n"
            "at org.jfree.chart.renderer.category.junit."
            "AbstractCategoryItemRendererTests.test2947660"
            "(AbstractCategoryItemRendererTests.java:409)\n"
        ),
        "candidates": [
            {
                "function_id": "org.jfree.chart.plot.PolarPlot.getLegendItems()",
                "source_path": "PolarPlot.java",
                "levels": ["PolarPlot.getLegendItems", "getLegendItems chart"],
            },
            {
                "function_id": (
                    "org.jfree.chart.renderer.category."
                    "AbstractCategoryItemRenderer.hitTest()"
                ),
                "source_path": "AbstractCategoryItemRenderer.java",
                "levels": ["AbstractCategoryItemRenderer.hitTest",
                           "hitTest rectangle chart"],
            },
            {
                "function_id": (
                    "org.jfree.chart.renderer.category."
                    "BarRenderer.render()"
                ),
                "source_path": "BarRenderer.java",
                "levels": ["BarRenderer.render", "render chart"],
            },
        ],
    }


class TestTestNameAnchoredReranker(unittest.TestCase):
    def test_junit_subpackage_resolves_conventional_test_name(self):
        hints = test_hints(snapshot()["context"])
        self.assertIn((
            "org.jfree.chart.renderer.category.AbstractCategoryItemRenderer",
            "AbstractCategoryItemRenderer"
        ), hints)

    def test_test_name_ranks_production_class_above_junit_noise(self):
        ranked = rank_with_test_hints(snapshot())
        self.assertEqual(ranked[0]["index"], 1)
        self.assertEqual(ranked[0]["test_name_boost"], 30)
        self.assertEqual(hinted_shortlist(snapshot(), 3)[0], 1)

    def test_non_conventional_test_does_not_invent_owner(self):
        self.assertEqual(test_hints("at com.example.Helper.invoke(Helper.java:11)"),
                         ())

    def test_no_gold_patches_required(self):
        x = snapshot()
        x["provenance"] = {"gold_patches_loaded": False}
        self.assertEqual(hinted_shortlist(x, 3)[0], 1)


if __name__ == "__main__":
    unittest.main()
