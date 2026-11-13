# Math-80-only manual readiness

The metadata-only Mac preflight sealed 42 requests for Math-80, with 21 candidates per arm. It verified the active Ollama 0.17.7 compatibility runner, pinned Llama3.1 8B manifest, model blob, generation defaults and no-tools Go template. The separate Math-only certificate uses the saved native token audit. The launcher rechecked exact token IDs for all 42 prompts and reproduced the archived payloads, evidence selections, graph and pinned BGE-M3 bundle. The only arm difference is propagation. No generation requests were made for this preparation.

| Scope | Calls | Settled upper bound | Serial peak | Applicable token cap |
|---|---:|---:|---:|---:|
| T0 | 21 | 24,669 | 31,846 | 36,000 |
| T1 | 21 | 24,680 | 31,860 | 36,000 |
| Math-80 pair | 42 | 49,349 | 56,529 | 72,000 |

The 216,000-token cohort cap also passes. The original 1,024-token input allowance, 160-token output cap and 8,352-token pending hold apply. The table shows planning quantities. The scope change permits a separately certified Math-only schedule in the existing launcher. It changes no scientific settings or candidate generation. Existing seals bind the source version. Preserve the original Lang artifacts and use the Math-specific seal below.

The focused suite passed 57 tests. The full fixture-guarded workspace suite reported 320 passes and the same three baseline failures in Agent4SR format-only finalization and the two tests for the absent `scripts/run_pipeline.sh`. The new certificate test rejects a mixed or incomplete Math-only request set.

The files are under `results/graft_fl_real_preparation/ollama0177-tokenizer-verified/`. The new certificate is `prompt-settlement-math80-certificate.json` (SHA-256 `53497e82ed8e28cd2f1555019edd1f1ee3e4e7de5b5b20f51a55bc779e9fe20e`). The new lock is `live-ready-math80/cohort.json` (canonical SHA-256 `155841b5216ce9d840fbe2fbf6143d6f714744e89f5ed19258d8af5de9f5d720`). The separate [configuration](../../../configs/graft_portable/real_mac_prompt_settlement_math80.json) names only Math-80. The [Lang diagnostic](../../../reports/graft_fl_lang54_real_smoke/README.md) was produced read-only.

After explicit approval, execute from the repository root. If Ollama restarts before the run, regenerate the Math-only preflight with `preflight --attest-live`, the same configuration and model paths, and a fresh output directory. Use that new lock in the manual command.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
"$HOME/.pyenv/versions/3.11.0/bin/python" -B -m rave_fl.graft.portable_experiment run \
  --profile local_mac --allow-inference \
  --config configs/graft_portable/real_mac_prompt_settlement_math80.json \
  --cohort results/graft_fl_real_preparation/ollama0177-tokenizer-verified/live-ready-math80/cohort.json \
  --serving-binary /Applications/Ollama.app/Contents/Resources/ollama \
  --generation-manifest "$HOME/.ollama/models/manifests/registry.ollama.ai/library/llama3.1/8b" \
  --ollama-models-dir "$HOME/.ollama/models" \
  --output results/graft_fl_real_preparation/manual-real-math80
```
