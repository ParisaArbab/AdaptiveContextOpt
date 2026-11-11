"""Real JDK compiler AST tests, skipped only when JDK is unavailable."""
import shutil
import unittest

from aitech.java_ast_bridge import extract_ast_calls, verified_syntax_candidates


def candidate(identifier, source):
    return {"function_id": identifier, "source": source}


@unittest.skipUnless(shutil.which("java") and shutil.which("javac"), "JDK not installed")
class TestJavaASTBridge(unittest.TestCase):
    def test_parses_actual_method_calls_not_string_occurrences(self):
        records = [
            candidate("pkg.Num.createNumber(String)",
                      'public int createNumber(String s) { '
                      'String text = "createInteger(fake)"; '
                      'return createInteger(s); }'),
            candidate("pkg.Num.createInteger(String)",
                      "public int createInteger(String s) { return 1; }"),
        ]
        result = extract_ast_calls(records)
        audit = verified_syntax_candidates(records, result)
        self.assertEqual(audit["edges"], [(0, 1)])
        self.assertEqual(audit["parse_error_indices"], [])
        self.assertGreaterEqual(audit["calls_extracted"], 1)

    def test_constructor_is_parsed_as_constructor(self):
        records = [
            candidate("pkg.Num.Num(String)",
                      "public Num(String s) { helper(s); }"),
            candidate("pkg.Num.helper(String)",
                      "void helper(String s) { }"),
        ]
        result = extract_ast_calls(records)
        graph = verified_syntax_candidates(records, result)
        self.assertEqual(graph["edges"], [(0, 1)])
        self.assertEqual(graph["parse_error_indices"], [])

    def test_type_ambiguous_overloads_not_guessed(self):
        records = [
            candidate("pkg.Num.use(String)",
                      "void use(String s) { convert(s); }"),
            candidate("pkg.Num.convert(String)",
                      "void convert(String s) {}"),
            candidate("pkg.Num.convert(Object)",
                      "void convert(Object s) {}"),
        ]
        result = extract_ast_calls(records)
        graph = verified_syntax_candidates(records, result)
        self.assertEqual(graph["edges"], [])
        self.assertIn("0:convert/1", graph["ambiguous"])


if __name__ == "__main__":
    unittest.main()
