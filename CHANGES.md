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
