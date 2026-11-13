# Phase 2 · Embedding integrity and fixed evidence selection

12 November 2026. Branch `GRAFT-FL` at baseline `e64d6dee9e642adaf08e4485512e7060f3335403`. Phase 2 was explicitly approved. This phase measured offline selector behavior on synthetic fixtures and made zero real LLM calls.

## Gates

| Gate | Result |
|---|---|
| Fixed T0/T1 isolation | **PASS on synthetic fixtures.** A single prepared graph, schedule, query, embedding bundle, seed, adjacency, counter and allowance feed both arms. The frozen selector configs differ only by `propagate`. Both requests use the same system prompt and model options. The common implementation uses the same renderer, packer and RRF settings. |
| Exact text, provenance and vector validation | **PASS for the engineering checks.** Missing, extra, duplicate, malformed and truncated vector entries fail closed. Tests exercise the real-bundle metadata checks using fixture data. A real backend configuration rejects synthetic bundles before selection. |
| Real BGE-M3 encoding and paired selection | **BLOCKED.** The local inventory found neither an installed BGE-M3 checkpoint nor `FlagEmbedding`. The check covered the default Hugging Face cache, `.tools/`, `data/`, `configs/` and Downloads. No checkpoint environment override was set. No real embedding vectors were generated or evaluated. |
| Input counting | **PASS for the declared proxy.** `ByteCounter` counts UTF-8 system and user content bytes plus 256. A local target-model chat template was unavailable. The loose tokenizer metadata in Downloads has no chat template. Serving-template parity remains open. |
| Portability | **PASS for selector parity.** Network-disabled Linux aarch64 with Python 3.12.16 reproduced all eight macOS arm/target evidence IDs, exact rendered-text SHA-256 hashes and proxy input sizes from the same serialized graphs and bundles. A full Linux dry-plan attempt stopped at the missing Git executable in the cached container. Java extraction, end-to-end collection and a real encoder remain unverified on Linux. |

## Implementation

- [Embedding validation and selector](../../../rave_fl/graft/topology.py) check exact SHA-256 coverage of all supplied nodes and failure-plus-target queries. Real fixed plans additionally require the exact prepared graph hash and buggy source revision, a pinned BGE-M3 checkpoint identity, the 1,024-dimensional dense representation, checkpoint-file digests and explicit non-truncation metadata. The original generic embedding reader still accepts legacy artifacts that meet the structural checks. Older real artifacts lacking the new graph binding need a fresh local export for this experiment.
- [Local exporter](../../../rave_fl/graft/encode.py) checks for a complete local file layout, enables offline mode, checks token lengths before model loading and records checkpoint-file digests, source revision, graph digest and encoder settings. Weight files are hashed in chunks. Manifest consistency checks establish the binding declared by a local export. Actual checkpoint identity and encoder behavior remain part of the blocked real validation.
- [Fixed runner](../../../rave_fl/graft/fixed_evidence_experiment.py) asserts identical seed, transition graph and embedding hashes for each target. It checks that the ready requests differ only in rendered user evidence and writes an evidence comparison with exact hashes, node membership, source spans, overlap, rank changes, counter units and convergence diagnostics. Path preservation, repair, typed edge weighting, method Full and adaptive routing are disabled. The reported typed source relations describe provenance. `REQUIRES` is lexical, `MAY_CALL` is a static candidate, and stack and coverage edges require the corresponding observed input.
- [Focused tests](../../../tests/test_graft_phase2_integrity.py) exercise the integrity and no-fallback checks, a constructed bridge, convergence, infeasible anchors, exact evidence and source provenance. The [node-order regression](../../../tests/test_graft_topology.py) failed before the fix because changing node order changed floating-point accumulation. The selector now iterates in stable node-ID order and keeps the AST statement order recorded by the graph builder for equal-line spans. Both arms use the correction. The graph's serialized hash still binds its exact node order, even when reordered input yields identical evidence.

## Synthetic evidence results

The [paired evidence artifact](phase02_synthetic_evidence.json) lists every selected ID, the common and exclusive IDs, source spans and hashes, raw selector scores and ranks, evidence bytes, and convergence diagnostics. Ranks here order evidence nodes by score before redundancy-aware packing. Each target used the same 2,048 proxy-unit allowance in both arms. T0 used the shared seed scores directly. T1 converged on every target and conserved total score mass to floating-point precision.

| Fixture target | Nodes T0 / T1 | Common / union | T0 / T1 complete-input proxy units | T1 iterations | T1 residual |
|---|---:|---:|---:|---:|---:|
| Java `calculate` | 4 / 4 | 3 / 5 | 1,755 / 1,941 | 22 | 7.45 × 10⁻¹⁰ |
| Java `helper` | 4 / 4 | 3 / 5 | 1,752 / 1,787 | 22 | 5.44 × 10⁻¹⁰ |
| Python `calculate` | 5 / 5 | 4 / 6 | 1,936 / 1,965 | 21 | 8.02 × 10⁻¹⁰ |
| Python `helper` | 5 / 5 | 4 / 6 | 1,933 / 1,966 | 20 | 9.52 × 10⁻¹⁰ |

For Java `calculate`, T1 exchanged the helper signature (`method_3e097fc41decd87859c4`, line 6) for the target method's guard (`code_f719ddfdd10759686c65`, line 3). The guard moved from node rank 5 to 2 and its score rose from 0.114 to 0.495. For Python `calculate`, the guard at `example.py` lines 2–3 (`code_6ba829158922d3a29e06`) rose from zero to 0.417 and from rank 6 to 2. The constructed bridge test seeds one statement and uses orthogonal dense vectors to isolate sequential propagation. A neighboring statement with zero T0 score gains mass and enters T1's evidence. Its recorded `REQUIRES` relation describes lexical provenance while typed edge weighting stays disabled.

All four fixture targets swapped one node. T1's rendered evidence was 29–186 bytes larger even though node counts and allowances matched. These runs establish that propagation can change what the packer selects on controlled input. Evidence quality, actual provider token use, matched prompt sizes and fault-localization accuracy remain unmeasured.

## Reproduction

Run the focused checks from the repository root using the project's Python environment.

```bash
python -B -m pytest -q tests/test_graft_topology.py \
  tests/test_graft_phase2_integrity.py tests/test_graft_fixed_evidence_*.py
```

The [phase log](PHASE_LOG.md) records the guarded regression command, baseline validation and artifact digests. The standalone check reused the Phase 1 fixture configuration and wrote new outputs. `cmp` found byte-identical plans under hash seeds 7 and 19. Both synthetic pairs completed with eight simulated requests and 3,466 simulated input-plus-output tokens. Resume issued zero new requests. No real provider tokens were consumed.

```bash
graft_fixture_config=/tmp/graft-phase1-cli.jLs4c7/config.json
graft_output=$(mktemp -d)
PYTHONHASHSEED=7 python -B -m rave_fl.graft.fixed_evidence_experiment dry-run \
  --config "$graft_fixture_config" --output "$graft_output/plan-a.json"
PYTHONHASHSEED=19 python -B -m rave_fl.graft.fixed_evidence_experiment dry-run \
  --config "$graft_fixture_config" --output "$graft_output/plan-b.json"
cmp "$graft_output/plan-a.json" "$graft_output/plan-b.json"
python -B -m rave_fl.graft.fixed_evidence_experiment run \
  --config "$graft_fixture_config" --output "$graft_output/run"
python -B -m rave_fl.graft.fixed_evidence_experiment resume \
  --config "$graft_fixture_config" --output "$graft_output/run"
```

The temporary configuration can be recreated with `eval.fixtures.create`, a distinct labels manifest, and the test-only `synthetic_bundle` as shown in the [fixture setup](../../../tests/test_graft_fixed_evidence_schedule.py). The saved evidence artifact is a projection of the final dry plan with request payloads omitted. The dry plan contains the complete payloads.

## Real validation dependency

An explicit `--model-dir` must point at an existing local BGE-M3 snapshot, such as `$HOME/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<checkpoint-revision>/` or a user-selected equivalent. The expected snapshot contains `config.json`, the tokenizer files (`tokenizer.json` or `sentencepiece.bpe.model`), transformer weights, `sparse_linear.pt` and `colbert_linear.pt`. Install `FlagEmbedding` in the chosen project environment separately. The exporter uses that local directory and accepts an explicit `--revision`, `--graph` and `--output`. The graph must be the exact result of `build_graph` on the fixed runner's prepared online instance and candidate entities. The bundle belongs at `<embeddings_dir>/<safe_name(instance_id)>.json` for both arms. After those resources are available, repeat the bounded paired dry-run and verify exact evidence and template-counter identity before any provider execution.
