# AIScalar setup for RAVE-FL (Phase 3b)

**No sudo, no change to existing AdaptiveContextOpt V2 installations.**
We use a dedicated repository clone and a Conda environment called `rave-fl`.
The bootstrap does not install or launch GPU services, and does not submit
resource-consuming jobs.

## 1. Connect to AIScalar from your Mac

```bash
ssh parbab@aiscalar.cdm.depaul.edu
```

Confirm free space and server modules before large downloads:

```bash
hostname
df -h "$HOME"
command -v conda || true
command -v defects4j || true
command -v ollama || true
command -v nvidia-smi || true
```

If `conda` is not found, check the server's environment modules
(`module avail`, if supported) or your user Miniconda installation.
Do not install a second Miniconda until you understand the existing setup.

## 2. Clone the research branch into a separate directory

```bash
mkdir -p "$HOME/RAVE_FL_Research"
cd "$HOME/RAVE_FL_Research"
git clone --branch feature/rave-fl-mvp \
  https://github.com/ParisaArbab/AdaptiveContextOpt.git
cd AdaptiveContextOpt
```

If the clone already exists, instead run
`git pull --ff-only origin feature/rave-fl-mvp` inside it.

## 3. Create and test Conda (Python 3.11 + Java 11)

```bash
bash scripts/aiscalar_rave_fl_bootstrap.sh 2>&1 | tee aiscalar_bootstrap.log
```

The script creates or reuses `rave-fl`, runs all RAVE-FL unit tests,
clones the FlexFL replication ranking dataset, checks Time-25 candidate
coverage, configures future **local** Git commits to
`pa.arbab@gmail.com`, then prints available server tools and GPU memory.

A new interactive shell must activate Conda again, for example:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rave-fl
```

## 4. Check Defects4J

Defects4J 3 requires Java 11, Git, Subversion, Perl and `cpanm`.
The official instructions are at https://github.com/rjust/defects4j .

If a working server installation is already present, supply the root:

```bash
export D4J_HOME="$HOME/defects4j"
export PATH="$D4J_HOME/framework/bin:$PATH"
defects4j info -p Time -b 25
```

**Do not automatically run an unreviewed, system-wide installer.**
If `defects4j` is not available, first inspect the bootstrap tool report.
For a user-local installation (only once the dependencies and quota are
available), typical upstream setup commands are:

```bash
cd "$HOME"
git clone https://github.com/rjust/defects4j.git
cd defects4j
cpanm --installdeps .
./init.sh
export PATH="$HOME/defects4j/framework/bin:$PATH"
defects4j info -p Time -b 25
```

The `cpanm` step may need a user-local Perl library rather than
system-wide write access. `init.sh` downloads considerable extra data.
If either step fails, save its log and fix the exact dependency issue.

## 5. Checkout Time-25b and validate all 24 candidates

Once `defects4j` works, go back to the RAVE-FL repository:

```bash
cd "$HOME/RAVE_FL_Research/AdaptiveContextOpt"
bash scripts/aiscalar_rave_fl_checkout.sh 2>&1 | tee aiscalar_checkout.log
```

This gets the **buggy**, not fixed, Time-25 checkout and checks every
method's source span in the union pool without calling an LLM.

If an exception reports a missing method or mismatched line span, do
**not** suppress it or substitute an unrelated source: benchmark source
revisions and method ranges must be reconciled first.

## 6. Model setup, after verifying server resource permissions

Only run model downloads and inference on an approved compute allocation,
not an overloaded/shared login node. First check:

```bash
nvidia-smi
command -v ollama
ollama list
```

Official Linux instructions: https://github.com/ollama/ollama/blob/main/docs/linux.mdx

If Ollama is available and server policies permit running it, use an
installed coder model with sufficient GPU RAM. A compatible example
model tag is `qwen2.5-coder:7b`; do not assume this tag is installed.

```bash
python -m rave_fl.compare_expanded \
  --flexfl-repo references/FlexFL_OriginalReplication \
  --checkout data/rave_fl_checkouts/Time-25 \
  --bug Time-25 --top-per-method 10 \
  --model qwen2.5-coder:7b --token-budget 16000 \
  --output results/rave_fl_phase3/expanded_ollama.json
```

Ollama must already be running and the named model installed for that
command. This experiment is a **rank-threshold selective baseline**, not
a trained RAVE-FL risk predictor. Ollama's actual reported tokens are
recorded, but the preflight budget is approximate and may overshoot.

## 7. Inspect and download logs

```bash
tail -n 60 aiscalar_bootstrap.log
tail -n 60 aiscalar_checkout.log
```

From your **Mac Terminal**, after the result files are created:

```bash
mkdir -p ~/Downloads/RAVE_FL_Results
scp parbab@aiscalar.cdm.depaul.edu:\
~/RAVE_FL_Research/AdaptiveContextOpt/results/rave_fl_phase3/expanded_dry_run.json \
  ~/Downloads/RAVE_FL_Results/
```

Research best practices: save model tag, code revision, environment
versions, selected candidates, per-call token usage, runtime, and the
failure outputs. Evaluate with gold labels only after inference.
