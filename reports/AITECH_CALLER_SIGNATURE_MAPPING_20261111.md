# AITECH: Compiler-resolved caller and callee identity, November 11, 2026

## Research objective

Previous compiler-backed code in AITECH included the **target method's**
parameter types but only the **calling method's name**. If the caller was
overloaded, the compiler graph could not be matched uniquely to AITECH's
ranked evidence candidates, even when the compiler knew the exact caller.

This stage closes that identity gap, keeping conservative fallback behavior.

## New implementation

- `aitech/java/ResolvedCallExtractor.java` emits parameter types for the
  calling method as an additional column to every resolved call.
- `aitech/java_compiler_graph.py` stores
  `caller_parameter_types` in each record. Old 8-column records are still
  supported, but their missing caller signature is explicitly unknown.
- `aitech/verified_graph_evidence.py` matches caller owner, method and
  normalized parameter types, then matches the target's complete signature.
  If the caller signature is missing, the tool only maps an otherwise unique
  caller and abstains on overloaded names.
- `aitech/fixtures/java_compiler_verified/CallerOverloadDemo.java`
  contains two `use` overloads and two `convert` overloads.
- `aitech/compiler_attribution_acceptance.py` runs compiler attribution
  and checks the full caller-to-callee identity mapping, saving hashes,
  ambiguity counts and accepted edges.

## Controlled verified result

On the complete source fixture with compiled Java types, the JDK
compiler identifies the following two static links:

| Calling method | Compiler target | Candidate edge |
|---|---|---|
| `use(String)` | `convert(String)` | `0 → 2` |
| `use(Object)` | `convert(Object)` | `1 → 3` |

The acceptance test requires exactly these two links. The relevant
GitHub Actions **unit test completed successfully** with that check.

This is a **controlled end-to-end type-binding validation**, not evidence
of improvement on Defects4J, SWE-bench or an LLM debugging benchmark.

## Reproduction

```bash
git switch AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v

python -m aitech.compiler_attribution_acceptance \
  --output results/aitech/compiler_attribution_acceptance.json
```

The daily GitHub Actions research workflow also includes the acceptance
runner and uploads the JSON with the other research artifacts.

## Important limitations

- The fixture is a simple Java program, not a real buggy repository.
- The compiler resolves static method declarations, not reflection or
  runtime polymorphic dispatch.
- Normalizing fully qualified Java types to short candidate names is
  not always identity-safe when imported classes share names.
- The real Defects4J `Lang-1b` one-off workflow has been configured,
  but its compiler graph result has **not been verified in this report**.
- No LLM inference, localization recall, token-cost change or patch
  success was measured in this stage.

## Next acceptance criteria

1. Confirm the separate Lang-1b GitHub Actions checkout/compile artifact.
2. Inspect failures if its legacy dependency classpath or JDK is incompatible.
3. Produce a pinned compiler-resolved graph and audit unmatched candidates
   and overloaded caller/target identities.
4. Freeze matched-budget selections, then measure patch-label recall and
   later actual LLM localization at equal total inference budgets.
5. Expand to independent bugs, keep test data separate from development.

Scientific objective: improve trustworthy context selection, **not**
maximize edge counts, commits or the number of artificial green squares.
