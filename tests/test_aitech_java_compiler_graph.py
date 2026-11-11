"""Compiler-attributed call graph tests with real JDK type resolution.

This is a *controlled compiler validation fixture*, not a benchmark bug.
No network access, LLM or gold fault labels.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from aitech.java_compiler_graph import audit_source_directory


@unittest.skipUnless(shutil.which("javac") and shutil.which("java"),
                     "JDK is required")
class TestCompilerGraph(unittest.TestCase):
    def test_resolves_overloads_with_identical_arity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Example.java").write_text(
                """
                package testpkg;
                public class Example {
                    public static String convert(String x) { return x; }
                    public static String convert(Object x) { return x.toString(); }
                    public static String use(String s) {
                        String a = convert(s);
                        String b = convert((Object) s);
                        return a + b;
                    }
                    public static Example create() {
                        return new Example();
                    }
                }
                """, encoding="utf-8")
            result = audit_source_directory(root)
            target_calls = [c for c in result.calls
                            if c.caller_method == "use" and c.target_method == "convert"]
            self.assertEqual(len(target_calls), 2)
            self.assertEqual({c.parameter_types for c in target_calls},
                             {"java.lang.String", "java.lang.Object"})
            self.assertTrue(all(c.target_owner == "testpkg.Example" for c in target_calls))
            self.assertTrue(any(c.kind == "CONSTRUCTOR" and
                                c.target_owner == "testpkg.Example"
                                for c in result.calls))
            self.assertEqual(result.unresolved_count, 0)
            self.assertEqual(result.source_files, 1)
            self.assertIn("Example.java", result.source_hashes)

    def test_compiler_emits_distinct_overloaded_caller_signatures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Overloads.java").write_text(
                """
                class Overloads {
                    static String convert(String x) { return x; }
                    static String convert(Object x) { return x.toString(); }
                    String use(String value) { return convert(value); }
                    String use(Object value) { return convert(value); }
                }
                """, encoding="utf-8")
            audit = audit_source_directory(root)
            calls = [c for c in audit.calls
                     if c.caller_method == "use" and c.target_method == "convert"]
            self.assertEqual(len(calls), 2)
            self.assertEqual({c.caller_parameter_types for c in calls},
                             {"java.lang.String", "java.lang.Object"})
            self.assertEqual({c.parameter_types for c in calls},
                             {"java.lang.String", "java.lang.Object"})

    def test_cross_file_static_call_target(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "A.java").write_text(
                "package sample; class A { int use() { return B.answer(); } }",
                encoding="utf-8")
            (root / "B.java").write_text(
                "package sample; class B { static int answer() { return 42; } }",
                encoding="utf-8")
            audit = audit_source_directory(root)
            self.assertTrue(any(c.target_owner == "sample.B" and
                                c.target_method == "answer" for c in audit.calls))

    def test_external_compilation_classpath_changes_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ext = root / "external"
            pkg = ext / "foo"
            pkg.mkdir(parents=True)
            (pkg / "Library.java").write_text(
                "package foo; public class Library { "
                "public static int answer() { return 42; } }",
                encoding="utf-8")
            binary = root / "classes"
            binary.mkdir()
            subprocess.run(
                ["javac", "-d", str(binary), str(pkg / "Library.java")],
                capture_output=True, text=True, check=True)
            app = root / "app"
            app.mkdir()
            (app / "User.java").write_text(
                "class User { int run() { return foo.Library.answer(); } }",
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Compiler attribution failed"):
                audit_source_directory(app)
            audit = audit_source_directory(app, classpath=str(binary))
            self.assertTrue(any(c.target_owner == "foo.Library" and
                                c.target_method == "answer" for c in audit.calls))

    def test_missing_types_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Broken.java").write_text(
                "class Broken { void run() { MissingClass.someMethod(); } }",
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Compiler attribution failed"):
                audit_source_directory(root)

    def test_missing_directory_fails(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                audit_source_directory(Path(td))


if __name__ == "__main__":
    unittest.main()
