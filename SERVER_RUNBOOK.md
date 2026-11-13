# GRAFT-FL Linux/A100 batch deployment

This runbook covers the fixed T0/T1 graph-propagation comparison. The code uses the existing `remote_linux_a100` profile and fixed-evidence collector. The first manually authorized batch command checks one fully prepared core defect, then proceeds through certified groups automatically if the first pair is complete and all of its charges reconcile.

## 1. What is frozen

- [Selection manifest](configs/graft_server/selection-v1.json), schema `graft-linux-holdout-v1`, canonical SHA-256 `251767e96283e3855ac1d7389db1d76c015ce8551c213a5acb4aafa0efd0c2b6`. The pinned sampling frame has 687 eligible active Git defects with all three FlexFL ranking files. The two disjoint sets each cover 14 projects. Project and defect orders use SHA-256 and the documented seed. One defect is selected from each project per round until a project's supply is exhausted. This is a project-balanced design for the eligible frame. Chart uses SVN and is outside this frame. [Development and historical exclusions](configs/graft_server/exclusions-v1.json).
- Defects4J `8c16da8230843cdc918eaf4ddb449637f02b83c6`, FlexFL `744143aa3e9c222b111beca55f5e75eab6d16285`, BGE-M3 `694b61558aea4ae2512ed8d0e189d5cf8adc2259`. Each manifest row pins both source revisions and all three ranking blobs. The method-label reference is separately pinned for offline use.
- Ollama source `9b0c7cc7b90562b370ce6a30efdc667326799223` (0.17.7), `llama3.1:8b` manifest digest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`, model blob digest `667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`, installed Go template digest `948af2743fc78a328dcb3b0f5a31b3d75f415840fdb699e8b1235978392ecf85`. The in-process offline audit entrypoint uses the same Go template and compatibility tokenizer as the serving binary. The native audit loads only the vocabulary and compares rendered bytes and token IDs for every distinct packing trial. [Pinned upstream build](https://github.com/ollama/ollama/blob/v0.17.7/Dockerfile) · [Pinned compatibility runner](https://github.com/ollama/ollama/blob/v0.17.7/runner/llamarunner/runner.go)
- Top 10 from each of SBIR, Ochiai and BoostN, up to 24 candidates per arm, 1,024 complete serving-prompt tokens, 160 output tokens, 8,192 context, temperature 0, seed 0 and original RRF. Each pending request holds 8,352 tokens. A successful settlement uses the verified prompt count plus 160. Provider input/output counts are tracked separately. A paid but invalid verdict still consumes tokens.
- Caps are 36,000 tokens and 24 calls per arm, 72,000 tokens and 48 calls per defect, and 216,000 tokens and 144 calls per group. The aggregate ceilings are 7.2 million tokens and 4,800 calls for the core or 14.4 million and 9,600 for core plus extension. These are ceilings. Admission checks the sum of certified group peaks and planned calls. The actual prepared denominator, calls and spend come from the saved receipts.

## 2. Server prerequisites and resources pending verification

| Resource | Requirement | Check |
|---|---|---|
| Host and isolation | Linux, Python 3.11, Git, curl, Perl, Java/Javac 11, lsof, ldd, util-linux unshare with an unprivileged user/network namespace | `prepare_linux.sh` |
| A100 | NVIDIA A100, compute capability 8.0, an NVIDIA driver compatible with the installed CUDA 12 toolkit, visible GPU and compute PIDs, sufficient free GPU memory | `nvidia-smi` and `prepare_linux.sh`; first generation checks actual placement |
| Build tooling | Go 1.24.1 or compatible later Go, CMake 3.31 recommended, CUDA 12.8 toolkit with nvcc, modern Linux C/C++ compiler supported by nvcc (GCC 11 or 12), Git source at the pin | `build_ollama.sh`, build receipt and loader check |
| Python | FlagEmbedding 1.4.2, transformers 4.56.1, torch 2.8.0 CPU, plus requirements-rave-eval.txt | `prepare_linux.sh` |
| BGE-M3 | Exact eight-file snapshot, 2.30 GB decimal. CPU FP32, batch size 8, maximum length 8,192. Provision RAM for the chosen worker count. Start with one worker if memory is limited | File-map digest and per-case bundle verification |
| Llama3.1 and tokenizer | Existing audited 4.92 GB GGUF, template, parameters, config and license plus the two saved tokenizer JSON files | Build/manifest/blob hashes and native exact-ID verification |
| Defects4J | Clean pinned framework, Major 3.0.1 for Java 11, required Perl modules, local pinned Gradle wrapper distributions and dependency files where needed. Maven/Gradle cases need their exact project dependencies cached locally | Host checks and individual trigger logs |
| Dataset resources | At most 400 scoped Git revisions, 600 pinned ranking CSVs, and one separately verified offline label blob | `prepare_batch.sh` writes acquisition and trigger receipts |
| Working storage | Dedicated persistent filesystem supporting file locks, hard links and SQLite WAL. Allow space for scoped checkouts, local build caches, observed logs, embeddings, native ID captures and generation ledgers | `df -h`, per-case and per-group output directories |

The CPU build and vocabulary-only tokenizer were tested in a network-disabled Linux ARM64 container. All 72 saved Mac request payloads matched exact rendered bytes and token IDs. The server inventory, CUDA build, server-native parity, project dependencies, authentic holdout triggers, and the number of admissible defects remain to be measured. The Mac run artifacts and JxPath-14 reproduction belong to separate development work. The completed Mac calls are excluded from this holdout.

## 3. Mac synchronization

Run from the GRAFT-FL repository on the Mac. Fill in the SSH host once. Use a new dedicated destination. The transfer explicitly includes the gitignored model and tokenizer resources and excludes the old results, fixed sources, claim directories, local environment and secrets.

```bash
read -rp 'A100 SSH destination (user@host): ' GRAFT_HOST
test "$(git branch --show-current)" = GRAFT-FL
git log -1 --format='%H %s'
git diff --exit-code HEAD -- SERVER_RUNBOOK.md scripts/graft_server configs/graft_server rave_fl
GRAFT_BUNDLE="$(mktemp -d)/graft-fl.bundle"
git bundle create "$GRAFT_BUNDLE" GRAFT-FL
git bundle verify "$GRAFT_BUNDLE"

ssh "$GRAFT_HOST" 'mkdir -p ~/graft-a100/incoming ~/graft-a100/assets/{bge-m3,llama-tokenizer,ollama-models}'
rsync -av "$GRAFT_BUNDLE" "$GRAFT_HOST:graft-a100/incoming/graft-fl.bundle"
GRAFT_ENCODER="$HOME/.cache/graft-fl/models/BAAI/bge-m3/694b61558aea4ae2512ed8d0e189d5cf8adc2259"
GRAFT_AUDIT="$PWD/results/graft_fl_real_preparation/ollama0177-tokenizer-verified"
rsync -av "$GRAFT_ENCODER/" "$GRAFT_HOST:graft-a100/assets/bge-m3/"
rsync -av "$GRAFT_AUDIT/tokenizer/tokenizer.json" "$GRAFT_AUDIT/tokenizer/tokenizer_config.json" \
  "$GRAFT_HOST:graft-a100/assets/llama-tokenizer/"
GRAFT_MODEL_FILES="$(mktemp)"
python3 - > "$GRAFT_MODEL_FILES" <<'PY'
import hashlib, json
from pathlib import Path
base = Path.home() / ".ollama/models"
tag = Path("manifests/registry.ollama.ai/library/llama3.1/8b")
raw = (base / tag).read_bytes()
assert hashlib.sha256(raw).hexdigest() == "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"
obj = json.loads(raw)
print(tag)
for layer in [obj["config"], *obj["layers"]]:
    print("blobs/" + layer["digest"].replace(":", "-"))
PY
rsync -av --files-from="$GRAFT_MODEL_FILES" "$HOME/.ollama/models/" "$GRAFT_HOST:graft-a100/assets/ollama-models/"
rm "$GRAFT_MODEL_FILES"
```

Use Bash for the interactive snippets if the login shell is Zsh. Save the implementation commit printed above. To refresh code before any preparation, transfer a new bundle and fetch it into the existing clone. Keep the original commit and workspace for every prepared batch.

## 4. Set up Linux without a broad benchmark archive

```bash
export GRAFT_BASE="$HOME/graft-a100"
git clone -b GRAFT-FL "$GRAFT_BASE/incoming/graft-fl.bundle" "$GRAFT_BASE/code"
cd "$GRAFT_BASE/code"
git log -1 --format='%H %s'
python3.11 -m venv "$GRAFT_BASE/venv"
export GRAFT_PYTHON="$GRAFT_BASE/venv/bin/python"
"$GRAFT_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
"$GRAFT_PYTHON" -m pip install -r requirements-rave-eval.txt 'FlagEmbedding==1.4.2' 'transformers==4.56.1'
"$GRAFT_PYTHON" -m pip check
"$GRAFT_PYTHON" -m pip freeze > "$GRAFT_BASE/incoming/python-receipt.txt"
nvidia-smi
nvcc --version
go version
cmake --version
df -h "$GRAFT_BASE"

git init -q "$GRAFT_BASE/assets/ollama-src"
git -C "$GRAFT_BASE/assets/ollama-src" fetch --depth=1 --no-tags --no-recurse-submodules \
  https://github.com/ollama/ollama.git 9b0c7cc7b90562b370ce6a30efdc667326799223
git -C "$GRAFT_BASE/assets/ollama-src" checkout --detach FETCH_HEAD
scripts/graft_server/build_ollama.sh "$GRAFT_BASE/assets/ollama-src" "$GRAFT_BASE/assets/ollama-build" \
  > "$GRAFT_BASE/incoming/ollama-build.log" 2>&1

git init -q "$GRAFT_BASE/assets/defects4j"
git -C "$GRAFT_BASE/assets/defects4j" fetch --depth=1 --no-tags --no-recurse-submodules \
  https://github.com/rjust/defects4j.git 8c16da8230843cdc918eaf4ddb449637f02b83c6
git -C "$GRAFT_BASE/assets/defects4j" checkout --detach FETCH_HEAD
export GRAFT_D4J="$GRAFT_BASE/assets/defects4j"
cpanm --local-lib-contained "$GRAFT_BASE/assets/perl" --installdeps "$GRAFT_D4J"
```

Major is a scoped tool dependency. Install only the official Major 3.0.1 Java 11 archive in the pinned framework and apply the exact wrapper setup from its pinned `init.sh`. Keep the archive and its checksum receipt. The first snippet uses Bash and expects a fresh framework checkout.

```bash
major_zip="$GRAFT_BASE/incoming/major-3.0.1_jre11.zip"
curl --fail --location --proto '=https' --output "$major_zip" \
  https://mutation-testing.org/downloads/major-3.0.1_jre11.zip
sha256sum "$major_zip" | tee "$major_zip.sha256"
unzip -q "$major_zip" -d "$GRAFT_D4J"
test -x "$GRAFT_D4J/major/bin/ant"
perl -pi -e '$_ .= qq(    -Djava.awt.headless=true \\\n) if /CodeCacheSize/' "$GRAFT_D4J/major/bin/ant"
perl -pi -e '$_ .= qq(\nif [ -z "\$MML" ]; then javac \$*; exit \$?; fi\n) if /^REFACTOR=/' "$GRAFT_D4J/major/bin/major"
perl -pi -e '$_ = qq(REFACTOR=\${REFACTOR:-"enable.decl.refactor enable.method.refactor"}\n) if /^REFACTOR=/' \
  "$GRAFT_D4J/major/bin/major"
mkdir -p "$GRAFT_D4J/framework/lib/test_generation/generation" \
  "$GRAFT_D4J/framework/lib/build_systems/gradle/"{deps,dists} "$GRAFT_BASE/assets/build-cache"
```

The directory checks above satisfy Defects4J's startup checks for developer triggers. Builds still require actual dependencies. If you have an existing **verified** Defects4J runtime on Linux, copy its `framework/lib/build_systems/gradle/dists/` and `deps/` into those paths. Put staged `.m2/` and `.gradle/` under `$GRAFT_BASE/assets/build-cache/` and retain a checksum list of what you staged. The batch workflow will identify project-specific misses from the pinned wrapper and isolated build logs. Use the exact dependency/version named by that wrapper and capture source and checksum before supplying it. If an official adapter changes production Java bytes or paths beyond the documented Joda-Time module mapping, that case stays blocked. The bulk Defects4J project archive is unnecessary. See the [pinned Defects4J initialization script](https://github.com/rjust/defects4j/blob/8c16da8230843cdc918eaf4ddb449637f02b83c6/init.sh).

Set `GRAFT_JAVA11` to an installed JDK 11 root, then create the host config.

```bash
read -rp 'Absolute JDK 11 directory: ' GRAFT_JAVA11
export GRAFT_JAVA11
"$GRAFT_JAVA11/bin/java" -version
"$GRAFT_PYTHON" - "$GRAFT_BASE/host.json" <<'PY'
import json, os, sys
from pathlib import Path
base = Path(os.environ["GRAFT_BASE"]) / "assets"
settings = {
    "defects4j": str(base / "defects4j"),
    "encoder": str(base / "bge-m3"),
    "tokenizer": str(base / "llama-tokenizer"),
    "binary": str(base / "ollama-build/dist/bin/ollama"),
    "models": str(base / "ollama-models"),
    "java_home": os.environ["GRAFT_JAVA11"],
    "perl_lib": str(base / "perl/lib/perl5"),
    "build_cache": str(base / "build-cache"),
    "endpoint": "http://127.0.0.1:11439",
}
Path(sys.argv[1]).write_text(json.dumps(settings, indent=2) + "\n")
PY
```

## 5. Dedicated listener and read-only preflight

Choose one full A100 UUID from `nvidia-smi -L`. Start a private listener under the same Linux user. Keep the listener dedicated to this experiment and keep its log.

```bash
nvidia-smi -L
read -rp 'Full A100 GPU UUID (GPU-...): ' GRAFT_GPU
env -u LD_PRELOAD -u LD_LIBRARY_PATH -u OLLAMA_LIBRARY_PATH \
  OLLAMA_MODELS="$GRAFT_BASE/assets/ollama-models" OLLAMA_HOST=127.0.0.1:11439 \
  OLLAMA_NEW_ENGINE=0 OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1 \
  OLLAMA_FLASH_ATTENTION=0 OLLAMA_KV_CACHE_TYPE=f16 CUDA_VISIBLE_DEVICES="$GRAFT_GPU" \
  nohup "$GRAFT_BASE/assets/ollama-build/dist/bin/ollama" serve \
  > "$GRAFT_BASE/incoming/ollama-server.log" 2>&1 &
echo "$!"
scripts/graft_server/prepare_linux.sh --host-config "$GRAFT_BASE/host.json" \
  --output "$GRAFT_BASE/incoming/host-$(date -u +%Y%m%dT%H%M%SZ).json"
```

Read-only preflight uses local files, device inspection, vocabulary-only tokenization and Ollama metadata endpoints. Physical generation placement is checked during the first authorized smoke. The launcher requires Ollama to report complete GPU placement and context 8,192 and binds its actual compatibility runner PID and mapped CUDA library to the A100. Missing or mismatched placement stops dispatch, preserves actual charges if known, and retains the full request hold.

## 6. Prepare and run

Keep one persistent workspace, including for the extension. `--count 100` selects the frozen core. `--count 200` prepares the same core plus the frozen extension. During preparation each defect uses the pinned buggy source and ranking CSVs, strict AST coordinate checks, official developer triggers in a Linux network namespace, and the pinned CPU encoder. Fixed revisions and labels stay outside online inputs. A failed defect stays in its original selection slot and receives a reason. Group certificates are specific to their exact member schedule and cannot be reused on a different schedule.

```bash
export GRAFT_WORK="$GRAFT_BASE/holdout-v1"
export GRAFT_COUNT=100
set -o pipefail
scripts/graft_server/prepare_batch.sh --host-config "$GRAFT_BASE/host.json" \
  --work "$GRAFT_WORK" --count "$GRAFT_COUNT" --workers 2 \
  2>&1 | tee -a "$GRAFT_BASE/incoming/prepare.log"
cat "$GRAFT_WORK/core/disposition.csv"
```

Read the disposition and sealed group bounds before generation. This is the single manual inference command. For the 200-defect evaluation, change `GRAFT_COUNT` to 200 and rerun preparation in the same workspace before invoking it.

```bash
scripts/graft_server/run_batch.sh --host-config "$GRAFT_BASE/host.json" \
  --work "$GRAFT_WORK" --count "$GRAFT_COUNT" --allow-inference \
  2>&1 | tee -a "$GRAFT_BASE/incoming/collection.log"
```

Monitor from a second session.

```bash
tail -F "$HOME/graft-a100/incoming/collection.log" "$HOME/graft-a100/incoming/ollama-server.log"
```

After a clean pause, use the same inference command and original absolute paths. A completed group is audited and skipped. An interrupted group resumes from the original claim and journal when a known paid prefix can be reconciled. A listener restart is allowed only after reattesting the identical pinned build, model, GPU inventory and runtime settings. The existing claim still applies. If a request is pending, unknown, or in violation, the hold stays in place and execution stops. Preserve the original journal and claim and report the state. A new workspace or certificate is never a retry mechanism for uncertain usage.

## 7. Report and archive

```bash
GRAFT_REPORT="$GRAFT_BASE/report-$GRAFT_COUNT-$(date -u +%Y%m%dT%H%M%SZ)"
scripts/graft_server/report_batch.sh --work "$GRAFT_WORK" --count "$GRAFT_COUNT" --output "$GRAFT_REPORT"
cat "$GRAFT_REPORT/REPORT.md"
```

The report contains the full selected denominator and separate preparation, execution, matched-complete and labeled counts. The paired tables include exact gold rank, MRR, Top-1, Top-5, candidate recall, provider input/output tokens, signed T1−T0 differences, evidence exposure, suspicion-score ties and invalid replies. A complete labeled candidate miss contributes zero accuracy. Incomplete pairs and missing labels stay visible outside the matched accuracy comparison. Unresolved reservations and any known token counts are retained in `usage.json`. The matched reporter emits PNG and SVG and excludes incomplete pairs from paired accuracy-versus-cost plots.

Copy reports and journals after collection is quiescent. Preserve the original server tree for resume. For read-only reproduction, also retain the sealed group plans, certificates, evidence inputs, model/tokenizer receipts, and offline labels in separate archive areas. Absolute local bindings are preserved in the archived manifests.

```bash
# From the Mac, after setting GRAFT_HOST as above
mkdir -p "$HOME/graft-a100-archive"
rsync -av "$GRAFT_HOST:graft-a100/report-*/" "$HOME/graft-a100-archive/reports/"
rsync -av "$GRAFT_HOST:graft-a100/holdout-v1/" "$HOME/graft-a100-archive/holdout-v1/"
```

## 8. Local verification

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python -m pytest -q tests/test_graft_server_batch.py tests/test_graft_ollama_live.py tests/test_graft_prompt_settlement.py
python -m pytest -q tests
python -m scripts.graft_server.selection \
  --defects4j data/graft_real_preparation/references/defects4j \
  --flexfl data/graft_real_preparation/references/flexfl > /tmp/graft-selection-v1.json
cmp /tmp/graft-selection-v1.json configs/graft_server/selection-v1.json
```
