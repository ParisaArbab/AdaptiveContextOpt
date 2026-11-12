"""End-to-end javac semantic graph -> candidate IDs, without benchmark labels."""
import shutil
import unittest

from aitech.compiler_attribution_acceptance import run


@unittest.skipUnless(shutil.which("javac") and shutil.which("java"), "JDK required")
class TestCompilerAttributionAcceptance(unittest.TestCase):
    def test_resolves_both_caller_and_callee_overloads(self):
        result = run()
        self.assertEqual(result["mapped_edges"], [[0, 2], [1, 3]])
        self.assertEqual(result["overloaded_calls_examined"], 2)
        self.assertFalse(result["mapping_ambiguity"]["callers"])
        self.assertFalse(result["mapping_ambiguity"]["targets"])
        self.assertFalse(result["llm_executed"])


if __name__ == "__main__":
    unittest.main()
