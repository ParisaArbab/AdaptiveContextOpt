# AITECH: Java AST call-evidence integration

## Objective

Improve reliability of AITECH's program-dependency retrieval. The previous
prototype matched method names in compressed source sketches and produced
false or ambiguous links, especially on overloaded constructors.

## Implemented components

1. `aitech/java_call_audit.py`: conservative name+arity checks on archived
   L1 `CALL` records, abstaining on uncertain overloads.
2. `aitech/java/CallAstExtractor.java`: JDK compiler AST parser for real
   Java method source snippets. It emits actual syntactic method and
   constructor invocations plus argument counts.
3. `aitech/java_ast_bridge.py`: wraps individual archived methods as
   Java compilation units, invokes local JDK parsing, and reports syntax
   diagnostics per snippet.
4. `aitech/java_ast_real_bug.py`: compares anchor-first selection with
   AST-derived, same-owner, unique name+arity **candidate edges** on the
   archived Lang-1 and Time-1 Defects4J inputs.

The Java compiler parser does NOT perform type attribution or resolve
semantic method bindings. Missing imports, inheritance, dynamic dispatch,
overloads and cross-class edges therefore remain unresolved. Any observed
AST-derived edge is a candidate grounded in **real source call syntax**,
not necessarily the correct callee.

## Reproduction

Requires JDK 17+ and Python 3.11, no API key and no paid compute.

```bash
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.java_ast_real_bug \
  --output results/aitech/java_ast_real_bug.json
```

The GitHub workflow `.github/workflows/aitech-research-agent.yml`
is configured to run the new experiment, retain the raw JSON as an
artifact and fail visibly on parser or execution errors.

## Verification and research limitations

- Unit tests explicitly test syntactic calls, constructor parsing,
  ambiguous same-arity overloads, and strings that resemble calls.
- A previous arity-only archived-sketch experiment inspected 98 CALL lines
  on Lang-1 and 131 on Time-1. It created 7 and 0 same-owner unique
  name+arity edges respectively. These values are **not** results of
  the new JDK AST implementation.
- Do not claim completed end-to-end AST experiment results until its
  GitHub Actions artifact or local output is confirmed.
- The available inputs contain isolated source methods, not full
  dependency-complete Java project checkouts.
- The same two bugs have been used for development, not an unseen final test.
- No LLM fault localization was run in this phase.

## Next research step

Use a pinned full Defects4J checkout with `javac` Trees/Elements
attribution to resolve target method signatures where possible. Label
each edge `verified`, `ambiguous`, or `unresolved`. Measure edge
precision on an independently manually audited sample before testing
localization accuracy or training an adaptive graph-expansion policy.
