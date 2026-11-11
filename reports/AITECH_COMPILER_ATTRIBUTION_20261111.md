# AITECH Research Stage: Compiler-attributed Java call targets

## Objective

Move from method-name and argument-count heuristics toward
compiler-verified **static method binding**. Java overloads with the
same number of arguments but different parameter types are an important
source of graph errors.

## New implementation

- `aitech/java/ResolvedCallExtractor.java` uses the JDK
  `JavacTask.parse()` and `JavacTask.analyze()` APIs.
  `Trees.getElement()` provides the compiler-selected declaration for
  method calls and constructor calls on complete, compilable Java source.
- `aitech/java_compiler_graph.py` starts the compiler, records source
  SHA-256 hashes, saves explicit target owner/name/parameter types,
  counts unresolved calls, and **fails closed** if required dependencies
  are missing or the compiler reports errors.
- `tests/test_aitech_java_compiler_graph.py` covers same-arity overloads,
  cross-file calls, constructor targets and failure on missing types.
- `aitech/fixtures/java_compiler_verified/OverloadDemo.java` is a
  small controlled, pinned Java program for reproducibility.

## What this establishes

Unlike lexical call extraction, the Java compiler can distinguish:

```java
translate(String value)
translate(Object value)
```

for the calls `translate(value)` and
`translate((Object) value)`. A daily GitHub Actions job now records
the attributed call graph and checks both target signatures.

## Reproduce

Requires Python 3.11 and JDK 17 or newer, without paid API calls:

```bash
python -m unittest discover -s tests -p 'test_aitech_*.py' -v

python -m aitech.java_compiler_graph \
  --sources aitech/fixtures/java_compiler_verified \
  --output results/aitech/compiler_verified_fixture.json
```

The GitHub workflow stores the JSON under its downloadable artifact.
The unit-test workflow has passed with the new compiler-attribution tests;
a passing full experiment artifact must be verified separately.

## Scientific limitations

**This is not yet a full Defects4J experiment.** Archived method snippets
are incomplete compilation units and often omit classes, imports and
dependencies. They cannot be treated as a complete compile-ready
repository. This compiler-attributed method currently validates only
controlled complete Java sources, not real bug-localization accuracy.

The javac-resolved method is the **static declared target**, not
necessarily the runtime method invoked through dynamic dispatch.
Reflection, missing classpath dependencies and generated code can still
prevent accurate graph construction.

No gold patch labels, fault-localization predictions, trained policy or
actual LLM token counts were measured.

## Next experimental requirements

1. Pin a Defects4J buggy revision and obtain its complete source tree
   and dependency classpath.
2. Run the compiler-backed extractor and record unresolved/errors.
3. Compare compiler-attributed edges with the existing lexical and
   name-plus-arity graphs.
4. Freeze graph-derived evidence selections before looking at gold
   patch labels.
5. Measure downstream fault-localization file/function rank, evidence
   retention, token cost and latency against relevance-only and anchor-first.

The objective is not maximizing the number of graph edges. It is
maximizing **reliable, useful evidence under a strict context budget**.
