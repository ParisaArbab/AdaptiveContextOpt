# Phase 1 - Typed evidence graph

Added stable, checksummed nodes/edges and Java/Python AST statement units. Every code unit records supplied source hash, path and lines. Stack-to-method mapping requires a unique file/line span. Same-owner static calls are MAY_CALL with explicit unresolved dispatch; rank scores never become coverage. Lexical-definition REQUIRES edges are structural interpretation support, not reaching-definition or causal proof. Coverage is accepted only as supplied checksummed instrumentation evidence.

Archive diagnostic: 97 graphs, 13,794 nodes, zero AST parse failures, 11,165 statement containment edges, 555 conservative static call links, 343 supplied stack frames and 90 unique stack/method links. No coverage was supplied. The archive contains candidate sources only, not the full 100 buggy checkouts: graph scope is expressly limited to those sources. No model calls occurred.

Validation: 12 cumulative focused tests (7 evaluation, 5 graph). Ambiguous overload calls, out-of-scope IDs, dangling edges, graph size bounds, absent coverage and Python extraction are covered.

Decision: continue with bounded statement evidence and explicit missing data. Do not invent repository-wide runtime edges or treat graph reachability as a causal explanation.
