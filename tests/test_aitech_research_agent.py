"""No paid compute / no network test suite for AITECH research runner."""
import tempfile
import unittest
from pathlib import Path

from aitech.research_agent import (
    run_ablation, choose_next_action, run_agent, render_report,
)
from aitech.sympy_trace_pilot import ANCHORS

TRACE = """sympy/core/tests/test_basic.py:37
  19: b1 = Basic()
  37: def test_immutable():
  38:     assert not hasattr(b1, '__dict__')
\\n[Agent4SR] STEP 4/20
sympy/core/basic.py:28
   6: from .cache import cacheit
   7: from .sympify import _sympify, sympify, SympifyError
  10: from ._print_helpers import Printable
  28: class Basic(Printable, metaclass=ManagedProperties):
"""


class TestResearchAgent(unittest.TestCase):
    def test_budget_sweep_is_reproducible_and_no_gold(self):
        result = run_ablation(TRACE)
        self.assertEqual(len(result["rows"]), 10)
        self.assertEqual(result["raw_tokens_proxy"], 49)
        self.assertFalse(result["gold_used_for_selection"])
        self.assertTrue(all(r["tokens_proxy"] <= r["budget_proxy"] for r in result["rows"]))
        arms = {(r["budget_pct"], r["method"]): r for r in result["rows"]}
        self.assertEqual(arms[(75, "relevance_only")]["known_dangling"], 1)
        self.assertEqual(arms[(75, "dependency_aware")]["known_dangling"], 0)
        self.assertEqual(arms[(75, "dependency_aware")]["tokens_proxy"], 35)

    def test_next_action_acknowledges_loss_of_assertion(self):
        rows = run_ablation(TRACE)["rows"]
        self.assertIn("anchor-preservation", choose_next_action(rows))

    def test_writes_truthful_report_in_offline_mode(self):
        with tempfile.TemporaryDirectory() as root:
            trace = Path(root) / "trace.log"
            trace.write_text(TRACE, encoding="utf-8")
            dest = Path(root) / "run"
            record = run_agent(trace, dest, search=False)
            self.assertTrue((dest / "results.json").exists())
            report = (dest / "report.md").read_text(encoding="utf-8")
            self.assertIn("No LLM was run", report)
            self.assertIn("not independent SWE-bench evaluation", report)
            self.assertEqual(record["discovery"]["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
