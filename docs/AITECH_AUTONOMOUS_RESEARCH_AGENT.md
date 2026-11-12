# AITECH autonomous research agent

The research agent has **two complementary execution paths**.

## 1. Daily Research Compass agent (already active)

The existing ChatGPT automation named **Autonomous AITECH and AdaptiveContextOpt
research** executes daily at 9 AM America/New_York on $0 paid cloud CPU.
It reads saved Research Compass state, researches primary literature, selects
one real-data research hypothesis, performs reproducible CPU experiments when
possible, analyzes failures, saves code/results/reports to Research Compass,
and reports actual findings. It does not require AIScalar.

Research Compass: https://parisa-research-compass.tuftedgem32.chatgpt.site/#agent

That automation has run real Defects4J CSV experiments. The independent
research agent is NOT an Ollama/GPU runner and must not claim LLM accuracy
unless a model actually ran.

## 2. GitHub Actions research runner

`.github/workflows/aitech-research-agent.yml` runs focused unit tests,
a deterministic CPU ablation on archived SymPy evidence, and saves
`results.json` plus `report.md` as a 30-day GitHub Actions artifact.

**The scheduled GitHub Actions event only runs from the repository's
default branch.** While the implementation remains on `AITECH`, it
runs on code pushes to AITECH, and can be enabled for scheduled execution
once the workflow is reviewed and merged into the default branch.
The separate Research Compass agent already runs daily without such a merge.

The CPU runner can optionally discover recent arXiv *metadata*; metadata
discovery is not equivalent to reading, validating or reproducing a paper.
A network failure is recorded instead of inventing sources.

## Local run (no keys, no AIScalar, no paid inference)

```bash
git switch AITECH
python -m unittest discover -s tests -p 'test_aitech_*.py' -v
python -m aitech.research_agent --no-search --output-dir results/aitech/agent_local
```

Optional paper metadata lookup:

```bash
python -m aitech.research_agent --output-dir results/aitech/agent_local
```

Output:
- `results/aitech/agent_local/results.json`
- `results/aitech/agent_local/report.md`

For actual LLM evaluation (not part of the unattended free-CPU agent),
use `aitech.ollama_localization` with a running local Ollama server
and explicitly record the model, GPU, tokens, and latency.

## Research safeguards

- $0 paid-API budget for the daily research automation.
- Maximum 5 candidate strategies and 600 seconds CPU, controlled by the
  Research Compass agent's saved mission constraints.
- Github job timeout of 10 minutes, uploads only data artifacts.
- No automated merge, paper submission, outbound messages, or GPU leasing.
- The GitHub runner makes **no automatic commits**. This avoids a
  self-triggered infinite loop and preserves review of scientific claims.
- Never use patch labels / gold files for selecting evidence.
- Do not treat low dangling count alone as localization success.
- Do not imply that the included seven-line SymPy ablation is a new
  independent benchmark or a trained ML policy.

## Most important next independent experiment

Use multiple Defects4J bugs and independent repositories, derive
support edges from AST/bytecode or ARISE rather than manually,
freeze ranked evidence before label scoring, and evaluate whether
dependency closure can preserve root-cause evidence under a fair budget.

Only after enough independent results should a learned risk predictor or
contextual bandit choose retrieval and restoration actions.
