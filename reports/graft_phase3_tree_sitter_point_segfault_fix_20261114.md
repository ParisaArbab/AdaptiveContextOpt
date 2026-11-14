# Phase 3 AIScalar Java preflight SIGSEGV: native Point workaround

Date: 2026-11-14
Branch: graft-fl-aiscalar-20261113
Prepared real cohort: data/graft_phase3_official_20261113_215102/manifest.json

## Observed crash

Preflight died with process status 139 (SIGSEGV) before calling Qwen. All 58
unit tests passed at the previous revision. Per-bug faulthandler diagnostics
showed crashes *after* Java parsing and during CandidateEntity construction
for the following files:

- Lang-1: src/main/java/org/apache/commons/lang3/AnnotationUtils.java
- Math-1: src/main/java/org/apache/commons/math3/RealFieldElement.java
- Time-1: src/main/java/org/joda/time/Chronology.java

Python's stack showed rave_fl/eval/schema.py:134 (constructing the error
message for an invalid CodeEntity source span). That Python frame is likely
where prior native memory corruption was detected, not necessarily the source.

## Likely root cause, upstream evidence

Upstream py-tree-sitter issue #472 identifies a 0.26.0 reference-counting
bug: reading Node.start_point.row or Node.end_point.row may prematurely free a
non-immortal integer, usually on real source files with line numbers >256.
Later execution segfaults in unrelated Python operations. The upstream
maintainers confirmed the Point getter bug and reported that 0.25.2 did not
reproduce their workload.

- https://github.com/tree-sitter/py-tree-sitter/issues/472
- https://github.com/tree-sitter/py-tree-sitter/issues/500

**The actual installed AIScalar tree-sitter version was not included in the
reported log. Verify with**:

    python -c "from importlib.metadata import version; print(version('tree-sitter'))"

If AIScalar is not on affected version 0.26.0, retain the diagnostics and
investigate further; do not claim root cause was independently confirmed.

## Patch

In rave_fl/eval/representations.py, Java method spans now use native
node.start_byte / node.end_byte offsets, and a precomputed table of UTF-8
line starts. bisect_right on this table produces the same one-based line
coordinates without accessing the unsafe native Point getters.

The indexer preserves method identity, source bytes, source order, candidate
pool selection, ranking settings, and all source/budget constraints. It fails
closed for an out-of-range or reversed byte span. The graph builder and Java
sketch helper were checked for Point getter usage and had none.

Regression tests cover:
- No access to start_point or end_point (a wrapper raises if called)
- Realistic >300-line Java code, with multibyte UTF-8 content before a method
- Reversed/out-of-range byte span failure
- Prior candidate identity and short-source behavior
- Optional per-file native crash trace and iterative AST traversal

No Qwen calls, Defects4J tests, or original checkouts were modified during
the fix. No new observed accuracy result is asserted by this note.

## AIScalar verification

    cd ~/RAVE_FL_Research/GRAFT_FL_New_Run
    conda activate rave-fl
    git fetch origin graft-fl-aiscalar-20261113
    git merge --ff-only FETCH_HEAD
    python -c "from importlib.metadata import version; print('tree-sitter', version('tree-sitter'))"
    python -m pytest -q tests/test_graft_phase3_bm25_java_spans.py tests/test_graft_phase3_real_bugs.py
    python -m scripts.diagnose_graft_phase3_preflight \
      --config results/graft_phase3_real_bugs_20261113_220831/config.json \
      --output results/graft_phase3_real_bugs_20261113_220831/native_diagnostics_byte_offsets

Only after all three isolated bug diagnostics print "passed":

    bash scripts/aiscalar_graft_phase3_multibug.sh \
      data/graft_phase3_official_20261113_215102/manifest.json

The experiment script still requires a successful three-bug preflight before
Qwen inference and writes each attempted experiment into a new results folder.
