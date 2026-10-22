# RAVE-FL single SWE-bench development run, 22 Oct 2026

**Measured smoke test**, not an accuracy claim or held-out experiment.

- Official SWE-bench Lite instance: `sympy__sympy-20590`.
- Confirmed buggy SymPy base commit: `cffd4e0f86fefd4802349a9f9b19ed70934ea354`.
- GitHub Actions run: https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/39814347577
- Raw, audited per-call ledgers in workflow artifact `swebench-sympy20590-rave-fl`.
- Model: local CPU `qwen2.5:0.5b` through Ollama, **not** the intended AIScalar Qwen 7B GPU model.
- Evidence: **`issue_only`**, official problem statement plus failing test identifiers. **No real test stdout/stderr** was available in this cloud smoke run.
- Candidate source: gold-free lexical retrieval over the official buggy Python source, top **8 functions**, with Python AST L0/L1/L2 representations.
- Controller: same verification/repair framework; `repair_grounded` untrained. Full and policy runs see the same candidate set and evidence.

## Directly measured Ollama usage

| Method | Ollama input + output tokens | LLM calls | Saved vs Full |
|---|---:|---:|---:|
| Full | 11,472 | 8 | 0% |
| Fixed Batch | 4,288 | 2 | 62.6% |
| Grounded Repair | 5,573 | 4 | 51.4% |

**All 3 arms failed candidate-level localization.** A class-level change was required but the initial retriever only returned functions. The model could not verify a candidate that was absent.

## Correction after independently inspecting the official merged PR

[SymPy PR #20590](https://github.com/sympy/sympy/pull/20590) fixes the regression by inserting **`__slots__ = ()` into class `Printable`** in `sympy/core/_print_helpers.py`.

The original ad-hoc function-level gold parser mistakenly labeled the adjacent `Printable.__str__` as changed because it assigned the whole diff *context hunk* to overlapping functions. That is **not the edited method** and should never be used as gold. The corrected extractor maps **actual changed lines** to the most specific Python AST entity, including class declarations. Any earlier rank labels for `Printable.__str__` are invalid; the original token counts remain measured Ollama counts.

This was a **source-entity granularity/retrieval failure**, not evidence that token selection independently caused a false negative. Since the bug was already inspected before correcting the extractor, any repeated run on this instance is **post-hoc development**, not held-out validation.

## Changes after the initial run

- Python indexer extended to classes as well as functions.
- Python AST sketches for classes now preserve class declaration, bases, assignments, method signatures and the complete original class source for L2.
- Gold evaluator changed from hunk-span intersection to actual changed old-side lines, choosing the smallest enclosing indexed source entity.
- A regression test checks that an inserted class-level attribute immediately before a method is attributed to the class, **not** that method.
- Existing source-pool option `--candidate-source existing_pool` can use an already generated Agent4LR candidate list, when actual Python source resolves for each entity.

**Required next step:** Review a corrected rerun's candidate recall, Full baseline completion, Top-1/Top-5 and measured tokens, then repeat a frozen approach on independent bugs. Do not present this development bug as evidence that the policy generalizes.
