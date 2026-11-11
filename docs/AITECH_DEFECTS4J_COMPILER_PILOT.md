# AITECH: Compiler-attributed calls on a real Defects4J checkout

## What changed

Previous AITECH experiments ran on archived isolated method snippets.
`aitech/defects4j_compiler_pilot.py` now reads a *real existing buggy
Defects4J checkout*, exports its class source directory and compilation
classpath, and runs javac's semantic attribution over complete source files.

`aitech/java/ResolvedCallExtractor.java` now accepts an optional
compile classpath. `aitech/java_compiler_graph.py` forwards the classpath.
Tests check that the exported paths are used, invalid checkouts are rejected,
and incomplete oversized source sets fail instead of silently generating
partial call graphs.

## How to run on a checked-out Defects4J bug

Requires Python 3.11+, JDK 17, and the Defects4J CLI on PATH. No GPU or
paid model is required.

Use the **buggy** revision, for example Lang-1b:

```bash
cd ~/RAVE_FL_Research/AdaptiveContextOpt_AITECH
git pull --ff-only origin AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v

# Only if Lang-1b has NOT previously been checked out:
mkdir -p data/rave_fl_checkouts
defects4j checkout -p Lang -v 1b -w "$PWD/data/rave_fl_checkouts/Lang-1"

# Source/classpath verification is performed by the program:
python -m aitech.defects4j_compiler_pilot \
  --checkout data/rave_fl_checkouts/Lang-1 \
  --max-files 1000 \
  --timeout 600 \
  --candidate-snapshot results/rave_fl_iteration_d/independent_live/Lang-1/input_snapshot.json \
  --output results/aitech/lang1_compiler_graph.json
```

The command does not delete, modify, or auto-download the checkout.
The graph output records file hashes, static method target signatures,
unresolved calls and failures. If source dependencies cannot be compiled
using the exported Defects4J classpath and installed JDK, it fails
explicitly. JDK compatibility may depend on bug revision and project.

The `--max-files` guard avoids accidentally attempting expensive
whole-repository compilation; increase only when the source count and
available memory justify it. This is a complete-source policy rather
than silently selecting only a subset.

## Important limitations

- **Not yet executed on Lang-1 in this GitHub session.** The pinned
  checkout, its exact Java build toolchain and runtime graph results
  must be validated in an environment with Defects4J installed.
- Even a resolved javac method target is *static*, and may not be
  the runtime method reached through polymorphism.
- Java version and classpath compatibility may prevent compilation,
  especially on older Defects4J bugs. Do not disable errors or label
  incomplete graphs as verified.
- Source SHA-256 and `.defects4j.config` SHA-256 are saved in results;
  keep these alongside the Defects4J framework version and checkout logs.
- No LLM, patch-generation, fault-localization accuracy, or gold-label
  evaluation is performed by this step.

## Acceptance gate

Only advance to graph-informed evidence selection when a real bug's
complete source tree can be attributed successfully and the number
of unresolved/ambiguous calls is reported. Then compare at equal
model-token budgets against rank-fill and Anchor-First, preserving a
separate unseen test set and freezing selections before gold scoring.

## Compiler graph to ranked evidence (added)

When `--candidate-snapshot` is supplied, the pilot matches compiler-resolved
target owner, method and parameter types to pre-existing ranked evidence
candidate IDs. An edge is accepted only with a unique caller candidate and
a unique target candidate. Ambiguous callers, ambiguous target overloads and
targets absent from the candidate pool are counted, not invented.

The snapshot's SHA-256 is recorded. Matching short/simple parameter type names
to compiler-qualified names is an **approximate normalization step**,
not independent proof that the candidate source is identical. The original
compiler target is still preserved in the graph so this can be audited.

This stage does not use gold patch labels and does not make LLM calls. Actual
fault-localization gains require separate paired evaluation.
