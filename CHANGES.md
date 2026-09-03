# What changed in this version

Compared against `76fea69`, the last version before this pass.

The short version: FlexFL and GraphLocator were already implemented properly.
The machinery around them — the part that feeds them data, measures what they
do, and compares configurations — was not, and that is what this pass
rebuilt. Six areas, roughly in order of how much they mattered.

---

## 1. The trigger test was never actually running

This is the one that invalidated everything downstream, so it goes first.

The orchestrator called the test harness with `test_patch=""`, and the
harness then ran a plain `pytest -v` across the whole repository at the buggy
commit. Two things went wrong there. The FAIL_TO_PASS test — the test that
demonstrates the bug — ships inside the gold test patch, so with an empty
patch that test did not exist in the checkout at all. And running the entire
suite instead of the one failing test meant the captured text was mostly
unrelated output.

FlexFL's Stage 1 reasons from stack traces and trigger-test evidence. So the
text the whole compression experiment was compressing largely did not contain
the evidence it was supposed to be preserving. The Docker path had the same
problem in a different form: it ran `docker run --rm <image>` with no command
at all, which starts the image's default entrypoint and runs no tests.

Now the gold test patch is applied before anything runs (with `git apply`,
then `--3way`, then `patch -p1` as fallbacks), and only the FAIL_TO_PASS ids
are executed. The Docker path applies the patch inside the container,
activates the image's conda environment, and runs the same selected tests.
Checkouts are reused across runs rather than re-cloned per instance, and
reset to the pristine commit each time so a leftover test patch can't leak
test-file entities into the structure map.

A capture that produced no failure evidence is now flagged and the instance
is skipped with a recorded reason, rather than quietly feeding an evidence-free
string into the pipeline and scoring the result as a localization failure.

## 2. Tokens were never measured

For a project whose headline claim is a token reduction, this was the awkward
one. `InstanceOutcome` had no token fields. The compressor computed a
before/after count and the orchestrator threw it away. What counting did exist
was `len(text) // 4`.

There are now two independent views of token cost, because they answer
different questions:

- **Per stage.** Every model call is attributed to the stage that made it —
  feedback loop, Agent4SR, Agent4LR, GraphLocator confirmation. This says
  where tokens go inside a single configuration.
- **Per arm, by differencing.** Removing an element and re-running captures
  its knock-on effects, not just its direct cost. Compression that saves 3,000
  tokens of context but causes the agent to make four extra exploratory tool
  calls has not saved 3,000 tokens, and only differencing shows that.

Counting uses a real tokenizer: the served model's own HuggingFace tokenizer
when the model id is an HF repo, otherwise tiktoken. The chars/4 estimate
still exists as a last resort, but any report built on it is tagged
`tokenizer_exact: false` and the analyzer says so out loud rather than
presenting an estimate as a measurement.

One thing worth knowing: compression is applied to the captured test output,
which is not the largest thing in the prompt. The structure map and the code
snippets Stage 2 retrieves are also substantial. The instrumentation now makes
that visible instead of leaving it as an assumption.

## 3. Accuracy was measured too coarsely, and the metric rewarded guessing wide

The old score was: did any predicted file appear in the ground-truth file
list. That ignores rank entirely — a correct file ranked 20th scored the same
as one ranked 1st. Worse, GraphLocator expansion *adds* entities to the
prediction set, so a wider prediction could only ever help. The metric had no
way to penalize noise, which meant the graph expansion step could not be
evaluated fairly by it.

Scoring now uses FlexFL's own metrics from `eval_FL.py` — Top-1, Top-3, Top-5,
MAP, MRR — computed at both method level (the paper's granularity, and the one
that separates a real localization from a lucky file hit) and file level (kept
because gold-patch symbol extraction is regex-based and can miss a symbol,
whereas the file list cannot be missed). Precision and recall are reported
alongside, so expansion that adds noise now costs something.

Making rank matter meant fixing the code that destroyed it: several
`sorted(set(...))` calls were alphabetizing the prediction list before it was
returned.

Symbol matching is deliberately forgiving about name shape but strict about
file. Graphify labels an entity `Class.method` or `method` depending on the
grammar, while gold-patch parsing yields the bare name. Requiring the file to
match and the name to agree on its last dotted segment is the tightest rule
that doesn't systematically undercount real hits.

## 4. There was no ablation

`CONDITIONS = ("raw", "leanctx")` — two arms. The plan called for five. There
was no way to run without Graphify (a Graphify failure just skipped the
instance), and the feedback loop wasn't independently switchable; it was
hardwired to run whenever the condition wasn't `raw`.

The arm matrix is now explicit: a factorial over Graphify, lean-ctx, and the
feedback loop, with the all-off corner (`pure_flexfl`) serving as the control.
`--arms default` runs the five arms the plan asks for; `--arms all` runs all
six valid combinations, which additionally separates interaction effects from
main effects — useful if two elements turn out to overlap in what they remove.

Six rather than eight, because a feedback loop with no compressor has nothing
to act on: there is no pruned text to restore and no compression decision to
second-guess. Those two combinations are excluded by construction rather than
run as duplicates of the arms they would be identical to.

One judgment call worth flagging: the "no Graphify" arm disables the
*graph-based* reasoning (the structural briefing before search, and the
call-graph expansion after) but keeps the structure index itself, because
FlexFL's own function-call tools require an index to query and the real FlexFL
builds one. Removing the index entirely would ablate FlexFL, not Graphify.

## 5. The feedback loop only worked in one direction

The requirement was two-way: restore what was wrongly pruned, and filter what
turned out to be useless. Only the first half existed. The loop could reveal
lines the compressor had cut, which means it could only ever *add* tokens —
making it a pure cost against the project's own headline metric.

Both directions now work. `MISSING: L<a>-L<b>` restores a range of the
original capture; `USELESS: C<a>-C<b>` deletes a range of the working text.
Two coordinate systems on purpose: `L` addresses the raw capture, so the agent
can ask for something it cannot see, and `C` addresses the numbered text in
front of it. Mixing them was the easiest way to get silently wrong edits, so
the prefixes are required and a mismatch is rejected rather than guessed at.
To make restore requests answerable at all, the agent is shown a menu of what
was actually omitted instead of being asked to guess line numbers of text it
never saw.

On not getting stuck, which was called out explicitly — there are four
independent guards, and they are not redundant:

1. A hard round cap, as before.
2. **No-progress stop.** If a round applies zero edits, the loop exits
   immediately instead of burning its remaining rounds re-asking.
3. **An idempotence ledger.** A range already restored cannot be restored
   again, and a range restored during this session cannot then be pruned. This
   is the guard that makes oscillation impossible rather than merely
   unlikely — without it, a model that disagrees with itself can ping-pong one
   region back and forth within any round budget.
4. **A prune budget**, capping how much can be removed per round, so one
   over-confident verdict cannot collapse the evidence to nothing.

There is also a deterministic stand-in verdict generator, so the feedback-loop
ablation arm is a real variable on the key-free backend instead of a no-op.

## 6. Benchmark and language were baked in

`fetch_instances.py` hardcoded `princeton-nlp/SWE-bench_Lite`, and the
gold-patch parser only understood Python's `def` and `class`. Nothing about
SWE-bench-Java would have worked.

Dataset and language are now data rather than code. A registry maps aliases
(`swe-bench-lite`, `swe-bench-verified`, `swe-bench`, `swe-bench-java`) to
HuggingFace ids, and any unregistered id can be passed directly. Where a fork
uses different column names, `--field-map logical=column` handles it without
an edit. Language adapters own the four things that actually differ: how to
read symbols out of a diff, which file suffixes count as source, which
tree-sitter grammar to parse with, and how to invoke the tests — pytest node
ids for Python, and for Java, Maven or Gradle selectors chosen by detecting
the build system in the checkout.

Honest caveat: the SWE-bench-Java registry entry points at a multi-language
fork whose schema I could not confirm from here. If it fails to load, the
column names are the thing to check first, and `--field-map` fixes it without
touching code.

---

## Also in this pass

**Every LLM, one contract.** The four hardcoded backend constructors became a
provider registry covering models served on your own hardware (Ollama, vLLM,
LM Studio, TGI, llama.cpp), hosted open-weight endpoints (DeepSeek, Qwen,
OpenRouter, Together, Groq, Mistral, Ollama Cloud) and proprietary APIs
(OpenAI, Anthropic, Gemini), plus a `custom` row for anything else that speaks
the OpenAI shape. Providers differ only in base URL, key variable, and default
model, so adding one is a registry line.

Three behaviours are handled there specifically because open-weight models
need them:

- Reasoning tags are stripped. R1, QwQ and Qwen3-thinking emit
  `<think>…</think>`, and DeepSeek returns reasoning out-of-band. FlexFL parses
  exactly one function call per turn, so a reasoning block that mentions a call
  would make the loop dispatch a function the model was only considering.
- Transient failures are retried with backoff. A local server returns 503 while
  a 32B model loads; a hosted one rate-limits. Without retries, one dropped
  call aborts a ReAct loop and loses that instance from every arm — which
  biases the comparison toward whichever arm happened to avoid the blip.
- Context-limit errors are deliberately *not* retried, so they propagate to
  FlexFL's adaptive-MAX behaviour, which shrinks the loop and reruns. Retrying
  the same oversized prompt would burn budget and fail anyway.

Temperature defaults to 0. Arms are compared against each other, and sampling
noise between them would be indistinguishable from an element's real effect.

**One command for the whole study.** `scripts/run_pipeline.sh` runs fetch →
every arm → evaluation → figures, with the model and the benchmark as flags.
Each run writes to its own timestamped directory, so runs never overwrite each
other and two models stay comparable from the artifacts alone.

**Figures.** Eight of them, generated at the end of every run. The
load-bearing one is the trade-off plot: tokens on one axis, accuracy on the
other, with the Pareto frontier drawn. The whole project is a trade-off claim,
and a frontier is the only presentation where "cheaper but worse" and
"strictly better" are visually distinguishable at a glance. There is also a
per-instance heatmap, which exists to answer the first question a reviewer
will ask about any aggregate difference: do the arms fail on the *same*
instances, or different ones?

## Two real bugs fixed along the way

**The compression tax report described one instance.** In
`compression_tax_analyzer.analyze`, the loop over conditions was indented
outside the loop over instances. The `conds` variable leaked from the last
iteration, and the `if not raw_ok: continue` guard above it did nothing. Every
compression-tax number the analyzer produced came from whichever instance
happened to be last.

**Dead code in the localizer.** Three orphan lines of a clobbered
`_lexical_overlap_score` body sat unreachable after `graph_structural_briefing`'s
return statement. Harmless at runtime, but it meant a function definition had
been overwritten at some point and nobody noticed.

---

## What this does *not* fix

Worth stating plainly, because these are the things a reviewer will press on:

- **FlexFL's Stage 1 non-LLM half is still a stand-in.** The paper interleaves
  SBIR, Ochiai, and BoostN results with Agent4SR's, in a specific order. What
  runs here is a single stack-trace-plus-lexical-overlap heuristic. A real
  Ochiai-style suspiciousness score from pytest coverage is the natural first
  addition.
- **GraphLocator's causal issue graph is richer than the call-graph walk
  implemented here.** Sub-issues as first-class vertices and dynamic issue
  disentangling are not implemented.
- **lean-ctx runs in reference mode** until its daemon is installed. Those
  results are tagged provisional and excluded from headline numbers
  automatically.
- **Nothing here has been run end to end for real.** The component chain was
  verified with a scripted stand-in model, and the provider layer with unit
  checks, but a real run needs network access, a live model endpoint, and
  `graphify` installed. The first genuine run is where the dataset schema
  assumptions get tested.


---

# Second pass — what the first real server run exposed

The pilot on the GPU server produced zero outcomes on astropy and zero
candidates on django. Four separate causes, none of them the compression
pipeline, plus the missing piece of FlexFL itself.

## The gold patch installed, but the package did not

`pip install -e .` failed on astropy in every Python version the harness
probed, always with the same error: `No module named 'setuptools.dep_util'`.
setuptools 74 removed that module; astropy's setup.py reaches it through
extension-helpers. The version ladder could never have rescued this, because
the interpreter was never the problem — the build tooling was.

Pinning setuptools inside the venv is not enough on its own either. PEP 517
build isolation hands the build its own fresh environment with the newest
setuptools no matter what the venv holds, and the error surfaces at "Getting
requirements to build editable", which is that isolated build. So the fix is
a ladder of toolchains, each rung pairing a pin with `--no-build-isolation`:
modern first, then `setuptools<74`, then `setuptools<60` with
`SETUPTOOLS_USE_DISTUTILS=stdlib` for packages that reach into stdlib
distutils internals.

## Django was being run with the wrong test runner entirely

Django's FAIL_TO_PASS ids look like `test_verbose_name_inline
(admin_inlines.tests.TestVerboseNameInlineForms)`. That is unittest's own
repr, not a pytest node id, and Django does not use pytest — it has
`tests/runtests.py`. Passing that string to pytest as a positional argument
makes pytest ignore it and collect the whole rootdir, which is exactly what
the log showed.

The language adapter now detects the framework from the checkout and routes
accordingly: `tests/runtests.py` for Django, `bin/test` for sympy, pytest for
everything else, with the unittest id form normalized to the dotted form
every runner actually wants.

## The FlexFL candidate merge did not exist

This was the substantive one, and it is the thing that was asked about
directly.

Agent4SR's output was being used alone when the model produced a parseable
ranking, and thrown away and replaced by a heuristic when it did not. The
paper does neither. Section 4.5 concatenates top-5 from SBIR, top-5 from
Ochiai, top-5 from BoostN, then top-5 from Agent4SR, caps the result at 20,
and hands that to Agent4LR, whose re-ranked top-5 is the final answer.

Two details are easy to get backwards and both are now enforced:

- **Agent4SR goes last**, not first. The paper's own reasoning is that
  methods Agent4SR finds are likely to be found by Agent4LR anyway, so they
  do not need the scarce high-rank slots.
- **Agent4LR does not add to the list.** It re-ranks and replaces. The final
  answer is always drawn from the merged 20, never a superset of it.

The replication package ships SBIR/Ochiai/BoostN as CSVs, but only for
Defects4J. On SWE-bench they have to be computed, so:

- **Ochiai is real SBFL.** A second test run under coverage.py with
  `dynamic_context = test_function` produces a genuine per-test spectrum,
  which is then scored with the standard Ochiai formula. This required
  carrying PASS_TO_PASS through the pipeline — with only failing tests every
  covered method scores identically and the ranking degenerates to "what
  ran".
- **BoostN is BM25** over method-level documents, labelled a stand-in for
  BoostNSift because the sifting stage is not reproduced.
- **SBIR is reciprocal-rank fusion** of the two, also labelled a stand-in.
  RRF rather than averaging because Ochiai lives in [0,1] and BM25 is
  unbounded; combining them numerically would let BM25 win on scale alone.

Why this matters concretely: astropy-12907's bug is in `_cstack`, and the
test fails on a boolean-matrix assertion whose traceback never names that
function. Trace-based evidence cannot reach it by construction. Coverage can,
because `_cstack` runs in the failing test and not in the passing ones. On a
fixture reproducing exactly that shape, with a model scripted to guess the
wrong method every single time, `_cstack` now comes back at rank 2 on Ochiai
alone.

## Agent4LR's short answers were discarding the merge

Related, and only visible once the merge existed. A model that answers with
`Top_1` and nothing else — very common below about 7B — left four of the five
final slots empty, throwing away everything the traditional localizers had
earned. Agent4LR's ranking is now padded up to five from the merged
candidates, in merge order, with the model's own picks kept ahead of the
padding.

## Zero was unattributable

A Top-1 of zero could mean the model never produced a ranking, or that Ochiai
was unavailable, or that everything ran correctly and still missed. Nothing
in the output distinguished those. Each outcome now records the merge mode,
what every ranker returned, which ones were unavailable and why, Agent4SR's
pre-merge top-5 separately from the merged list, whether each stage produced
a parseable ranking at all, and the tail of each stage's raw model output.
The console prints the merge provenance per arm as it runs, and warns
explicitly when Agent4SR never emitted a parseable block — the signature of a
model too small for the ReAct protocol.

## On the model used in the pilot

`gemma4:e2b` is about 2B parameters. FlexFL requires ten turns of exactly
`FunctionName(Argument)` followed by a `Top_k :` block. The pilot log shows
37,448 tokens spent on one instance whose context was 1,455 — the loop ran
every turn without the model ever calling `exit()`, which is what a model
that is not following the protocol looks like. The merge makes such a run
degrade rather than collapse, since the traditional localizers still
contribute, but a model of roughly 7B or more (14B–32B coder models
preferably) is the real fix.

## Merging the two branches

`main` had been rebuilt around Defects4J and had the candidate merge; this
branch had the ablation matrix, token accounting, metrics, feedback loop and
figures. They were merged rather than chosen between. `graphify_structure.py`
keeps both views of the graph — this branch's structure map and call graph,
and main's `GraphifyIndex`, which has a much better `snippet()` that resolves
through the Python AST. `llm_backends.py` keeps the provider registry, with
main's `ChatBackend` surviving as a delegating shim so the Defects4J modules
gain retries and reasoning-tag stripping without an API change.

One thing the merge quietly did that needed undoing: it replaced this
branch's compressor wholesale, and with it the `compress()` entry point the
ablation runner calls. main's version is better in the way that matters — it
drives the real LeanCTX binary rather than a Python re-implementation — so it
was kept as the authoritative path, with the reference-mode fallback restored
underneath it. main deliberately had no fallback, which is right for a
single-arm run; across five arms an uninstalled binary would take out every
leanctx arm and leave nothing to compare against. The cost is paid in
labelling instead: reference mode still propagates to `provisional=True` and
is excluded from the headline table.
