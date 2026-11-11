"""Focused no-network tests for conservative Java call-candidate discovery."""
import unittest

from aitech.java_call_audit import count_args, parts, sketch_call_audit


def item(signature, calls):
    return {"function_id": signature,
            "levels": ["", "METHOD: " + signature + "\n" + "\n".join("CALL " + x for x in calls)]}


class TestJavaCallAudit(unittest.TestCase):
    def test_counts_nested_arguments(self):
        self.assertEqual(count_args("foo(a,b), new X(1,2), \"x,y\""), 3)
        self.assertEqual(count_args(""), 0)
        self.assertEqual(count_args("Map<String,Integer>, int[]"), 2)

    def test_signature_parsing(self):
        self.assertEqual(parts("pkg.Num.method(String,int)"), ("pkg.Num", "method", 2))

    def test_resolves_unique_arity_and_same_class(self):
        candidates = [
            item("pkg.Num.createNumber(String)", ["createInteger(str)", "createInteger(str, 10)"]),
            item("pkg.Num.createInteger(String)", []),
            item("pkg.Num.createInteger(String,int)", []),
        ]
        audit = sketch_call_audit(candidates)
        self.assertEqual(set(audit.edges), {(0, 1), (0, 2)})
        self.assertEqual(audit.ambiguous, ())

    def test_keeps_same_arity_overloads_unresolved(self):
        candidates = [
            item("pkg.Num.use(String)", ["convert(value)"]),
            item("pkg.Num.convert(String)", []),
            item("pkg.Num.convert(Object)", []),
        ]
        audit = sketch_call_audit(candidates)
        self.assertEqual(audit.edges, ())
        self.assertIn("0:convert/1", audit.ambiguous)

    def test_does_not_infer_unresolved_receiver_type(self):
        candidates = [
            item("pkg.Num.createNumber(String)", ["external.createInteger(str)"]),
            item("pkg.Num.createInteger(String)", []),
        ]
        self.assertEqual(sketch_call_audit(candidates).edges, ())

    def test_named_same_class_qualifier(self):
        candidates = [
            item("pkg.Num.createNumber(String)", ["Num.createInteger(str)"]),
            item("pkg.Num.createInteger(String)", []),
        ]
        self.assertEqual(sketch_call_audit(candidates).edges, ((0, 1),))


if __name__ == "__main__":
    unittest.main()
