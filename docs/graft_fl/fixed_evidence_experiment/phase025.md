# Phase 2.5 · Local BGE-M3 installation and validation

12 November 2026. `GRAFT-FL`, baseline `e64d6dee9e642adaf08e4485512e7060f3335403`. **PASS.** The approved official BGE-M3 checkpoint is installed and verified. Real CPU embeddings passed repeatability and the paired offline selection gate. The frozen comparison protocol and implementation from Phase 2 retain their recorded hashes. Phase 3 is pending approval.

## Environment and dependencies

The project interpreter is pyenv Python 3.11.0 on Apple M3 Pro, arm64 macOS 26.1. The Mac had 18 GiB of memory and 49 GiB of available disk at preflight. PyTorch 2.8.0 successfully ran a CPU tensor operation and reported MPS available. Transformers 4.56.1, Tokenizers 0.22.0, Safetensors 0.6.2 and NumPy 2.3.3 were already installed.

The package resolver constrained every existing distribution, then the selected official wheels were hash-pinned and tested in a temporary overlay. The same six wheels were added to the project environment. All 413 previous package versions were preserved and the seven pre-existing `pip check` messages were identical before and after. The import `from FlagEmbedding import BGEM3FlagModel` succeeded. [Official package](https://pypi.org/project/FlagEmbedding/1.4.2/)

| Added distribution | Version |
|---|---|
| FlagEmbedding | 1.4.2 |
| Accelerate | 1.15.0 |
| IR Datasets | 0.6.3 |
| LZ4 | 4.4.5 |
| PEFT | 0.21.2 |
| SentencePiece | 0.2.3 |

## Checkpoint and frozen input

The official `BAAI/bge-m3` revision is `694b61558aea4ae2512ed8d0e189d5cf8adc2259`. Its eight selected files total 2,295,338,769 bytes. The selection includes the Safetensors base weights, both learned heads, and the tokenizer and configuration files. The model cache is under `~/.cache/graft-fl/models/BAAI/bge-m3/<revision>/`. [Official pinned snapshot](https://huggingface.co/BAAI/bge-m3/tree/694b61558aea4ae2512ed8d0e189d5cf8adc2259)

Every file matched the pinned upstream size and content identity. Large files matched their published SHA-256 hashes. Small configuration files matched their published Git blob identities, and their raw SHA-256 hashes were recorded for reuse. The base weight SHA-256 is `993b2248881724788dcab8c644a91dfd63584b6e5604ff2037cb5541e1e38e7e`. Download interruptions were recovered from saved nonoverlapping ranges. A local resolver failure was handled with per-request addresses obtained from DNS and the usual HTTPS certificate verification. The fully assembled weight hash was checked before loading the model. [Verified checkpoint file manifest](phase025_checkpoint.json)

The smoke uses the existing Phase 2 Java and Python fixture checkouts, with real BGE-M3 embeddings over synthetic source and failure text. The copied fixture retained the same source revision, graph hashes and candidate schedules. The actual local tokenizer produced 8 exact inputs per graph, 7–45 encoder tokens for Java and 6–36 for Python, including special tokens. This is the embedding tokenizer. The T0/T1 plan retains its original 2,048-unit `ByteCounter` allowance for the entire generation prompt.

The local validation driver runs the existing exporter, compares dense coordinates and lexical weights across fresh CPU loads, and calls the existing fixed planner. Its checks require the exact graph, source revision and checkpoint file digests; complete exact-text coverage; 1,024 finite dense components with unit norm within `1e-5`; nonempty positive sparse weights with numeric token IDs; and repeatability within `1e-6` absolute tolerance with identical sparse token IDs. The optional MPS CPU-parity tolerance is `1e-4` absolute with identical token IDs. The driver disables network connections, excludes labels, and rejects provider or journal creation during planning.

## Real encoder results

| Device and fixture | Unique inputs | Nonzero sparse weights per text | Repeat dense / sparse maximum absolute difference | Status |
|---|---:|---:|---:|---|
| CPU FP32 · Java | 8 | 5–30 | 0 / 0 | PASS |
| CPU FP32 · Python | 8 | 4–24 | 0 / 0 | PASS |
| MPS FP32 · Python | 8 | 4–24 | 0 / 0 | PASS, bounded smoke |

MPS ran after CPU, with CPU fallback disabled. For the eight Python inputs, MPS vs CPU maximum absolute differences were `2.95e-7` dense and `6.71e-7` sparse, with identical sparse token IDs. The paired experiment used the CPU exports. Repeat bundles contain different elapsed-time provenance, which changes full-file checksums even when the embedding values match exactly. Both selector arms read the same canonical CPU bundle object and file for each defect.

## Paired evidence with real embeddings

| Target | T0 / T1 selected nodes | Jaccard overlap | T0 / T1 complete-input proxy units | T1 iterations |
|---|---:|---:|---:|---:|
| Java `calculate` | 4 / 4 | 0.60 | 1,755 / 1,941 | 21 |
| Java `helper` | 4 / 4 | 0.60 | 1,752 / 1,787 | 21 |
| Python `calculate` | 4 / 5 | 0.80 | 1,688 / 1,969 | 19 |
| Python `helper` | 4 / 5 | 0.80 | 1,685 / 1,966 | 20 |

Both Java targets selected the line 3 guard in T1 (`code_f719ddfdd10759686c65`). For `calculate`, it replaced the helper signature and moved from node rank 6 to 2. For `helper`, it replaced the line 4 return and moved from rank 6 to 3. Both Python targets retained every T0 node and added the line 4 return (`code_f0c297744cf88a6e758c`). T1 converged in every target, with maximum residual `7.21e-10`. The full plan records the exact IDs, scores, source spans, evidence hashes and prompt payloads.

The fixed planner verified a shared checkpoint-backed embedding artifact per fixture, matching seed and adjacency hashes, and matched request settings. All ready requests had the same system prompt and differed only in the rendered evidence. Two separate dry runs under Python hash seeds 7 and 19 reproduced the plan byte-for-byte. Paths, repair, typed edge weighting and adaptive routing stayed disabled. The fixture measures how propagation changes evidence selection with a real encoder. Fault-localization accuracy and provider token usage remain unmeasured. **Zero LLM calls** were made.

## Portability

The same real embedding bundles and frozen graphs reproduced all eight macOS selections in a network-disabled Linux aarch64 container with Python 3.12.16. Selected IDs, exact evidence text hashes and proxy input sizes matched. A future Linux/A100 host can reuse the exact bundles for the same graphs or re-export with the pinned checkpoint, dependency versions and CPU FP32 profile using the same code. Linux re-encoding and A100 GPU execution remain unverified. The current exporter enables FP16 for `cuda` devices, so CUDA would need a separate precision validation before it could be used as a substitute for the CPU reference. Mac wheel hashes are specific to the installation platform.

## Artifacts and reproduction

The full local results are in `results/graft_fl_phase25_fixture/`. They include both prepared graphs, `embeddings/`, repeat exports, the package and model pins, `validation-cpu.json`, `validation-mps.json`, `linux-selector.json`, `paired-plan.json`, and raw CPU/MPS logs. The directory is ignored by Git. Retain or copy it with its fixture checkout when moving the comparison to another host. The verification driver rebuilds the local fixture paths from its own directory. To regenerate the exports, copy only `validate.py`, `checkpoint.json`, `config.json`, `fixture/` and `graphs/` to a fresh output directory before running it.

```bash
graft_results=results/graft_fl_phase25_fixture
graft_fresh_results=$(mktemp -d)
cp "$graft_results/validate.py" "$graft_results/checkpoint.json" \
  "$graft_results/config.json" "$graft_fresh_results/"
cp -R "$graft_results/fixture" "$graft_results/graphs" "$graft_fresh_results/"
graft_revision=694b61558aea4ae2512ed8d0e189d5cf8adc2259
graft_model="$HOME/.cache/graft-fl/models/BAAI/bge-m3/$graft_revision"
PYTHONPATH="$PWD" python -B "$graft_fresh_results/validate.py" --model-dir "$graft_model" --device cpu
# Optional on a host that reports MPS available, after the CPU run succeeds.
PYTHONPATH="$PWD" python -B "$graft_fresh_results/validate.py" --model-dir "$graft_model" --device mps
```

To repeat the paired selection with existing exports on the same host, run the fixed planner with the saved configuration and a new output filename.

```bash
graft_results=results/graft_fl_phase25_fixture
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=7 python -B \
  -m rave_fl.graft.fixed_evidence_experiment dry-run \
  --config "$graft_results/config.json" --output "$graft_results/plan-recheck.json"
cmp "$graft_results/paired-plan.json" "$graft_results/plan-recheck.json"
```

| Verified artifact | SHA-256 |
|---|---|
| Checkpoint file manifest | `9be187819d7229eb4286d793f96859fc7f4172e05cb0f30dd0d6604b55434609` |
| CPU validation | `849c6ea2ea22a67d3c946ed730267d6332121f8fa24dd4154ff1cd8c494ca270` |
| MPS validation | `7b3667e941864b19f4dc3b62fb2ea892e619675f37a8c6344e692d4058cdb9d8` |
| Linux selector parity | `608aa472cd7e2b724a1747a88b04c49293b6ee0a683197978b4fb1459d0b0078` |
| Paired plan and both seed replays | `7048c7968548ef559d01b36064961cc33bf17ea43e78e40004e0d7c46e7ed175` |
| Java CPU embedding bundle | `7b4152687b164041bb5e12d29a09e36b6960a58c8bb9398d671515a26cd31b97` |
| Python CPU embedding bundle | `1a763f792941fcbac737010175e4b7b06d181a294df8a5c728f62d0b7f300d8e` |

## Regression tests

After package installation, the Phase 2 focused command passed **39 tests in 2.90 s**. The guarded regression command in the [phase log](PHASE_LOG.md) passed **223 tests in 35.95 s**, including the focused checks. The regression guard allowed the tests' in-process HTTP fixture only. The frozen arm configs, preregistration and scientific implementation files retain their Phase 2 values.
