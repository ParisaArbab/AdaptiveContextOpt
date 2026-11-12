# AITECH Research Iteration: Source-grounded retrieval, positional bias, and related API recovery

Date: 2026-11-11 (America/New_York)
Branch: `AITECH`
Status: **real CPU LLM experiments completed; related-API retrieval coverage measured.**

## Why the algorithm was changed

The original AITECH L0-to-L1 source upgrades on Lang-1, Time-25,
Math-1 and Chart-1 did not improve the Qwen 0.5B prediction. Merely
adding code is not the same as adding useful fault evidence.

We changed three parts, all constrained to the buggy source and test
information visible **before** looking at a patch:

1. **Model capacity:** Qwen2.5-Coder 1.5B instead of 0.5B, with
   frozen revision, CPU-only float32 inference and actual tokenizer
   costs. Do not attribute the difference between models to retrieval.
2. **Failure-grounded candidate rank:** weighted production stack-frame
   identity, BM25-like textual evidence, and weak production-class
   names inferred from failing test names. No learned embeddings.
3. **Related-API evidence recovery:** when a production class appears
   on the stack, add a source-present class with a reciprocal API name
   such as InputStream/OutputStream, Reader/Writer, Encoder/Decoder,
   or Parser/Printer. The link is explicitly **heuristic and
   unverified** as a causal/program dependence.
4. **Conservative STOP fallback:** if the current context already
   includes a class supported by a stack frame or test-name clue,
   avoid spending additional L1 tokens unless there is independent
   evidence for expected benefit. This is **untrained**, not the
   final calibrated ML action-value controller.
5. **Position-bias check:** reverse exactly the same eight candidates
   while holding model, failure text and total token ceiling fixed.
   Changed predictions under reversal are not evidence of causality.

## Qwen 1.5B development comparisons

Same model, eight candidate methods per arm, max 1800 input tokens,
greedy generation. Changed-class Hit@1 is a posthoc patch proxy.

| Bug | Previous L1 with stack/BM25 only | Revised L1 with test-name clues | Revised L0 tokens | Revised L1 tokens |
|---|---|---|---:|---:|
| Math-1 | correct class | correct class | 1077 | 1381 |
| Chart-1 | wrong class (PolarPlot) | correct class (AbstractCategoryItemRenderer) | 1069 | 1143 |

- Initial source:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43150737850
- Revised source:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43151088416

The changed test-name clue was informed by Chart-1, so **this 1/2
to 2/2 development result is not evidence of independent generalization**.
In revised Chart-1, reversing candidate order yielded a wrong class,
confirming significant output-order sensitivity.

## Independent-project diagnostic: Codec-1 and Compress-1

These bugs were not used to design the original guard. Qwen 1.5B
predicted a wrong changed class with STOP and with one forced L1
upgrade on **both bugs**. STOP saved model input tokens but did not
improve accuracy. On Codec-1, reversing the same eight candidates
changed the prediction to the correct changed class.

`Compress-1` exposes a stronger retrieval failure. The real
`EOFException` appears in `CpioArchiveInputStream`, but the official
changed class is `CpioArchiveOutputStream`. The correct writer class
was present in the frozen **25-method source pool** but absent from
both earlier **eight-method shortlists**. More L1 detail on the wrong
eight methods cannot recover that missing class.

- Real-project run:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43151606993
- Official pinned changed classes:
  https://raw.githubusercontent.com/rjust/defects4j/8c16da8230843cdc918eaf4ddb449637f02b83c6/framework/projects/Compress/modified_classes/1.src

## Measured retrieval-only gain at the same eight methods

Using exactly the same frozen 25-method source candidates and
posthoc official modified-class labels:

| Bug | Original eight: modified-class recall | Stack/BM25 eight | Related-API eight |
|---|---:|---:|---:|
| Codec-1 | 2/3 | 2/3 | 2/3 |
| Compress-1 | 0/1 | 0/1 | **1/1** |

The new method restores a `CpioArchiveOutputStream` method by linking
the trace-matched input stream class to its matching source-present
output stream class. It never invents a class not in the frozen pool.

**This is candidate recall, not LLM Hit@1, and the bug has now been
examined while developing the new policy.** Do not use Compress-1
as untouched future validation data.

- Real candidate-recall artifact:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43152818370
- Code:
  `aitech/related_api_retrieval.py`
  `aitech/related_api_retrieval_pilot.py`
- Tests:
  `tests/test_aitech_related_api_retrieval.py`
  `tests/test_aitech_related_api_retrieval_pilot.py`

## Model-level verification

A separate same-Qwen study compares stack-only, reciprocal-API,
general producer-versus-consumer reasoning prompt, and reverse-order
control on these **development** bugs:

https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43153151368

The model scores must not be inferred from the candidate-recall result.
Report outcomes only after verifying the saved raw prediction artifact.
All model scoring reads official labels **after** raw output persistence.

## Verified matched-model localization result after related-API retrieval

The same Qwen2.5-Coder-1.5B-Instruct CPU model was run on Codec-1
and Compress-1 using eight method IDs per completed arm:

| Bug | Stack-only L0 Hit@1 | Related-API L0 Hit@1 | General causal instruction Hit@1 | Related-API reverse order | Related-API peer L1 |
|---|---:|---:|---:|---:|---:|
| Codec-1 | 0 | 0 | 0 | 1 | N/A (no API sibling) |
| Compress-1 | 0 | 0 | 0 | 0 | 0 |

**This is a negative downstream result.** The related-API method
restored the correct changed *class* for Compress-1 but the 1.5B
model still chose an input-stream method, not the output-stream class.

We also tested upgrading the recovered paired API method to its
actual source body rather than keeping its constructor signature.
The algorithm now prefers a non-constructor method where available.
For Compress-1 the retrieved `CpioArchiveOutputStream.close()` source
body increased model input from **1024 to 1052 tokens**, with no
change in changed-class Hit@1.

Adding a general instruction to distinguish a failure-detection
site from the code that caused it also failed to change the class
prediction. The prompt had 1103 input tokens on Compress-1, with
no localization improvement.

On Codec-1, reversing the **same eight source methods** selected
`Caverphone`, a changed class; the forward order predicted
`DoubleMetaphone`, an unchanged class. The position-order result
must not be presented as a generally reliable improvement. It
demonstrates model sensitivity to presentation order.

Experiment runs:
- Initial same-model related-API test:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43153151368
- Peer source-body follow-up:
  https://github.com/ParisaArbab/AdaptiveContextOpt/actions/runs/43154014633

Reproduction code:
- `aitech/related_api_cpu_localization.py`
- `aitech/related_api_retrieval.py`
- `tests/test_aitech_related_api_cpu_localization.py`
- `tests/test_aitech_related_api_retrieval.py`

**Methodological conclusion:** Better candidate recall is necessary
but not sufficient. Our next localization model must reason at the
CLASS level first and test source-specific hypotheses, instead of
trusting one position-sensitive forced-choice number. To avoid
overfitting, new projects, not Codec/Compress/Chart, must be used
to evaluate this next model.

## Scientific decision and risks

- **Keep** the related-API method as an experimental retrieval
  candidate. It rescued a source class excluded by trace-only evidence.
- **Do not yet promote** it as a better fault localizer. Hit@1 may
  remain unchanged even when gold candidate recall improves.
- Reciprocal names are not data flow or causal support. Reliable
  compiler-bound and/or audited producer-consumer links are preferable.
- The fixed source pool and regex Java extraction remain incomplete.
- Qwen 1.5B ranking is sensitive to method presentation order.
- An untrained STOP heuristic is not an ML novelty contribution.
- Run future, truly new Defects4J projects without tuning on their
  patch labels; use more than two bugs and include bootstrap uncertainty.
- Collect independent positive, negative and neutral real action
  outcomes before attempting risk calibration and learning.

This work implements measurable algorithm changes and reports both
positive candidate-recall and negative LLM-localization findings.
