"""Regression tests for retrieval recall across producer/consumer API pairs."""
import unittest

from aitech.related_api_retrieval import (
    owner_of, related_shortlist, sibling_class, choose_behavior_method
)


def candidate(method, detail="int value = 0;"):
    return {"function_id": method, "levels": [method, method + "\n" + detail]}


class TestRelatedApiRetrieval(unittest.TestCase):
    def test_source_present_output_partner_is_retained(self):
        items = [
            candidate("pkg.CpioArchiveInputStream.readFully()"),
            candidate("pkg.CpioArchiveInputStream.getNextEntry()"),
            candidate("pkg.CpioArchiveInputStream.read()"),
            candidate("pkg.Other.alpha()"),
            candidate("pkg.Another.beta()"),
            candidate("pkg.Foo.gamma()"),
            candidate("pkg.Bar.delta()"),
            candidate("pkg.Baz.epsilon()"),
            candidate("pkg.CpioArchiveOutputStream.write()"),
        ]
        snap = {
            "context": "java.io.EOFException\n"
                       "at pkg.CpioArchiveInputStream.readFully(File.java:20)",
            "candidates": items,
        }
        result = related_shortlist(snap, count=8)
        selected = [items[i]["function_id"] for i in result.original_indices]
        self.assertIn("pkg.CpioArchiveOutputStream.write()", selected)
        self.assertEqual(
            result.related_class_proposals,
            (("pkg.CpioArchiveInputStream", "pkg.CpioArchiveOutputStream"),))
        self.assertFalse(result.binding_verified)
        self.assertEqual(len(result.original_indices), 8)

    def test_test_named_production_classes_are_preserved(self):
        items = [
            candidate("pkg.Unrelated.alpha()"),
            candidate("pkg.Alpha.compress()"),
            candidate("pkg.Beta.decode()"),
            candidate("pkg.Gamma.encode()"),
            candidate("pkg.Other.run()"),
        ]
        snap = {
            "context": "--- pkg.AlphaTests::testBad\n"
                       "--- pkg.BetaTest::testBad\n"
                       "--- pkg.TestGamma::testBad\n",
            "candidates": items,
        }
        chosen = related_shortlist(snap, count=4)
        owners = {owner_of(items[i]["function_id"])
                  for i in chosen.original_indices}
        self.assertIn("pkg.Alpha", owners)
        self.assertIn("pkg.Beta", owners)
        self.assertIn("pkg.Gamma", owners)

    def test_sibling_avoids_constructor_when_behavior_method_exists(self):
        items = [
            candidate("pkg.ArchiveOutputStream.ArchiveOutputStream(OutputStream)"),
            candidate("pkg.ArchiveOutputStream.ArchiveOutputStream(OutputStream,int)"),
            candidate("pkg.ArchiveOutputStream.close()"),
        ]
        self.assertEqual(choose_behavior_method(items, [0, 1, 2]), 2)

    def test_missing_sibling_is_never_invented(self):
        snap = {
            "context": "at pkg.Reader.read(File.java:21)",
            "candidates": [
                candidate("pkg.Reader.read()"),
                candidate("pkg.Helper.help()"),
            ],
        }
        chosen = related_shortlist(snap, count=2)
        self.assertEqual(chosen.related_class_proposals, ())
        self.assertTrue(all(i in (0, 1) for i in chosen.original_indices))

    def test_reciprocal_names_only_are_heuristic(self):
        self.assertEqual(sibling_class("pkg.CpioArchiveInputStream"),
                         "pkg.CpioArchiveOutputStream")
        self.assertEqual(sibling_class("pkg.CpioArchiveOutputStream"),
                         "pkg.CpioArchiveInputStream")
        self.assertEqual(sibling_class("pkg.JsonEncoder"), "pkg.JsonDecoder")
        self.assertEqual(sibling_class("pkg.Unrelated"), None)

    def test_bad_budget_rejected(self):
        snap = {"context": "", "candidates": [candidate("pkg.A.m()")]}
        with self.assertRaises(ValueError):
            related_shortlist(snap, count=0)


if __name__ == "__main__":
    unittest.main()
