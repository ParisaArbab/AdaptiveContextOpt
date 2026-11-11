# AITECH: Do not confuse Java types with identical short names

## Hypothesis

Evidence graph retrieval can introduce incorrect call links when a
ranked candidate's parameter type is written using only its short name.
For example, `java.util.Date` and `java.sql.Date` both become `Date`.
Even Java compiler-resolved call targets become unreliable if the
downstream mapping discards the package identity.

## Implementation

`aitech/verified_graph_evidence.py` now retains full package names for
candidate parameter types when present. Where a candidate uses a short
name, the mapper checks the entire set of compiler-resolved signatures
that match it. If more than one distinct full signature matches the
same shortened candidate, it **refuses to create that evidence edge**.

This safety check applies to both calling and called methods.
The mapping report exposes `ambiguous_type_identities` and
`ambiguous_short_type_identity_count` rather than hiding the loss.

## Controlled compiler validation

The repository contains a complete, compilable fixture:

- `aitech.fixture.first.Date`
- `aitech.fixture.second.Date`
- Two `PackageCollisionDemo.use(...)` overloads
- Two `PackageCollisionDemo.convert(...)` overloads

With all four candidate signatures fully qualified, the compiler
mapping resolves two separate graph edges:

| Caller | Callee |
|---|---|
| `use(first.Date)` | `convert(first.Date)` |
| `use(second.Date)` | `convert(second.Date)` |

When candidate signatures use `use(Date)` and `convert(Date)`, the
mapper reports an ambiguous type identity and accepts neither edge.
This is intentional abstention, not a localization failure.

The tests `test_aitech_verified_graph_evidence.py` and
`test_aitech_java_compiler_graph.py` cover both the short-name
confusion and full type resolution.

## Meaning and limits

This is an engineering correctness improvement for graph-to-evidence
identity matching. The controlled fixture is not a real bug and does
not measure Recall@k, token savings, LLM accuracy or patch correctness.
Compiler attribution remains static, not runtime dynamic dispatch.

Next experiment: run on a pinned complete Defects4J checkout, save
the counts of resolved, absent, overloaded and package-ambiguous
candidate mappings, then compare rank-fill and graph-aware selection
at matched budgets before and after gold-label evaluation.
