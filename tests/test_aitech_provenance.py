"""Provenance extraction tests, no external services or gold-label oracle."""
import unittest

from aitech.provenance import (
    capture_locations, render_with_provenance, resolve_relative_import,
)
from aitech.sympy_trace_pilot import extract, ANCHORS


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


class TestProvenance(unittest.TestCase):
    def test_provenance_from_numbered_source_not_gold(self):
        evidence = extract(TRACE)
        locations = capture_locations(TRACE, evidence)
        self.assertEqual(len(locations), len(ANCHORS))
        self.assertEqual(locations["printable_import"].path, "sympy/core/basic.py")
        self.assertEqual(locations["printable_import"].line, 10)
        self.assertEqual(locations["test_assertion"].path, "sympy/core/tests/test_basic.py")
        self.assertEqual(locations["test_assertion"].line, 38)

    def test_selected_lines_render_with_verified_origins(self):
        evidence = extract(TRACE)
        locations = capture_locations(TRACE, evidence)
        context = render_with_provenance(
            evidence, ["basic_definition", "printable_import"], locations)
        self.assertIn("[source: sympy/core/basic.py:10]", context)
        self.assertNotIn("sympy/core/_print_helpers.py", context)
        self.assertEqual(context.count("[source:"), 2)

    def test_no_verified_location_rejected(self):
        with self.assertRaisesRegex(ValueError, "No verified snippet location"):
            render_with_provenance(extract(TRACE), ["test_assertion"], {})

    def test_derive_relative_import_for_offline_audit_only(self):
        self.assertEqual(
            resolve_relative_import("sympy/core/basic.py", "._print_helpers"),
            "sympy/core/_print_helpers.py",
        )


if __name__ == "__main__":
    unittest.main()
