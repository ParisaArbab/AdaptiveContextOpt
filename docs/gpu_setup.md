# Running the small pilot on your own NVIDIA GPU

This is for when you clone the repo yourself onto a GPU machine, instead of
running inside this sandbox. Same benchmark (SWE-bench Lite), same small
pilot size — nothing about the pilot scope changes, only where it runs and
which model serves the localization agent.

## Why GPU matters here

Two places in this pipeline can use your GPU:

1. **Local open-source model inference** (the main one). The project
   compares closed models (Claude, GPT) against open models (DeepSeek,
   Qwen). Instead of paying for those via API, you can serve one locally
   with [vLLM](https://github.com/vllm-project/vllm) and point
   `agent_localizer.py` at it — `make_local_gpu_chat_fn()` is already wired
   for this (see below).
2. **lean-ctx's CUDA build.** The install script supports a `--cuda` flag
   (`lean-ctx`'s own embedding-based operations run faster on GPU). Not
   required, but worth using if you're already on a GPU box.

## 1. Clone and install

```bash
git clone https://github.com/ParisaArbab/AdaptiveContextOpt.git
cd AdaptiveContextOpt
pip install -r requirements.txt
```

## 2. Install lean-ctx (CUDA build)

```bash
curl -fsSL https://leanctx.com/install.sh | sh -s -- --cuda
lean-ctx proxy enable
lean-ctx doctor   # confirm the daemon answers — should say mode: daemon from here on
```

This was blocked in the sandbox by a GitHub API rate limit on its shared
IP — a normal machine with its own IP should pull the release fine.

## 3. Serve an open-source model locally with vLLM

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000
# or: vllm serve deepseek-ai/DeepSeek-Coder-V2-Instruct --port 8000
```

Pick a model size that fits your GPU's VRAM. Once it's serving:

```bash
export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
export LOCAL_LLM_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"   # must match what you served
```

`make_local_gpu_chat_fn()` in `agent_localizer.py` is a thin wrapper over
these two env vars — no API key needed, since local servers don't require
one. Ollama and TGI's OpenAI-compatible endpoints work the same way if you
prefer those over vLLM.

## 4. Docker, if available on this machine

If this GPU machine also has Docker (separate from the GPU — SWE-bench's
eval images don't need CUDA), you can run the authoritative harness instead
of `--local-fallback`:

```bash
# pull/build the SWE-bench eval images per princeton-nlp/SWE-bench's docs, then:
python wp1/run_wp1_benchmark.py --instances data/instances.json --backend local ...
# (omit --local-fallback)
```

Without Docker, keep `--local-fallback` as before — same caveat as always:
not equivalent to the official harness, pipeline-validation only.

## 5. Run the small pilot

Same instance set as the earlier pilot, pinned for reproducibility:

```bash
python wp1/fetch_instances.py \
    --instance-ids "astropy__astropy-12907,sympy__sympy-17630,sympy__sympy-16792" \
    --out data/instances.json

python wp1/run_wp1_benchmark.py \
    --instances data/instances.json \
    --local-fallback \
    --backend local \
    --out results/wp1_results.json

python wp1/compression_tax_analyzer.py \
    --results results/wp1_results.json \
    --out results/compression_tax_report.json
```

To go slightly wider than the 3 confirmed instances while staying a small
pilot, drop `--instance-ids` and use `--n 15 --seed 42` instead — same seed
as before, reproducible.

## Notes

- `--backend local` runs the *real* FlexFL Agent4SR/Agent4LR ReAct loops
  and GraphLocator's LLM-confirmed expansion — not the heuristic stand-ins.
  Expect more tool-calling round trips per instance than the heuristic
  backend; that's expected, not a bug.
- If you want to compare closed vs. open models in the same run, do two
  separate `run_wp1_benchmark.py` passes (`--backend claude` and
  `--backend local`) with the same `--instances` file and different
  `--out` paths, then diff the two result files.
