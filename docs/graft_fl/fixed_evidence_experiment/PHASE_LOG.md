# Fixed-evidence experiment phase log

| Phase | Date | Baseline branch and revision | Result | Fresh model calls | New inference tokens |
|---|---|---|---|---:|---:|
| 0 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Design gate passed. User confirmed proceeding with the pre-existing untracked files. Live feasibility blocked by external resources. | 0 | 0 |
| 1 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Fixed-schedule gate passed. 203 tests, two byte-identical plans and a completed synthetic run/resume. | 0 | 0 |
| 2 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Engineering checks passed. 39 focused checks, 223 total regressions and matching Linux selector outputs. Real BGE-M3 validation blocked by missing local resources. | 0 | 0 |
| 2.5 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Real BGE-M3 gate passed on Mac CPU. Bounded MPS and Linux real-bundle selection checks passed. 39 focused checks and 223 total regression tests passed. | 0 | 0 |
| 3 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Offline prompt and synthetic accounting gates passed. Serial fixture admission verified at the boundary. 53 focused and 237 total tests passed. | 0 | 0 |
| 4 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Synthetic rehearsal and report regeneration passed. Five complete pairs, one paid-invalid pair, one separate zero-call admission rejection. 44 focused and 246 total tests passed. | 0 | 0 |
| 5 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Portable launcher passed two-profile shard and merge validation, a 100-entry dry-run, and a Linux audit. 46 focused and 255 total tests passed. | 0 | 0 |
| 6 | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Offline engineering checks passed. Three real Defects4J inputs and serving-token parity remain blocked. 29 focused and 261 total tests passed. | 0 | 0 |
| 6 follow-up | 2026-11-12 | GRAFT-FL · e64d6dee9e642adaf08e4485512e7060f3335403 | Pinned resource inventory and independent Java source-audit checks passed on fixtures. Target data and serving token parity remain blocked. 34 focused and 266 total tests passed. | 0 | 0 |

[Audit and Phase 1 proposal](phase00.md) · [Preregistration and baseline SHA-256 fingerprints](preregistration.json) · [Draft 2020-12 schema](preregistration.schema.json)

Phase 0 produced the audit, preregistration and schema in this directory. [Phase 1](phase01.md), [Phase 2](phase02.md), [Phase 2.5](phase025.md), [Phase 3](phase03.md), [Phase 4](phase04.md), [Phase 5](phase05.md), and [Phase 6](phase06.md) record validation based on the same HEAD. Further work is pending approval.

## Validation record

Executed with `$HOME/.pyenv/versions/3.11.0/bin/python -B -` from the repository root. The command below records the exact schema, configuration and hash validation performed. Independent import and environment checks are listed in the audit.

```python
import copy, hashlib, json, subprocess
from dataclasses import asdict
from pathlib import Path
from jsonschema import Draft202012Validator
from rave_fl.graft.topology import TopologyConfig
from rave_fl.graft.selection import SYSTEM

d = Path('docs/graft_fl/fixed_evidence_experiment')
spec = json.loads((d / 'preregistration.json').read_text())
schema = json.loads((d / 'preregistration.schema.json').read_text())
Draft202012Validator.check_schema(schema)
validator = Draft202012Validator(schema)
validator.validate(spec)
t0 = spec['arms']['dense_sparse_no_propagation']
t1 = spec['arms']['dense_sparse_with_propagation']
assert t0 == asdict(TopologyConfig(propagate=False))
assert t1 == asdict(TopologyConfig(propagate=True))
assert {k for k in t0 | t1 if t0.get(k) != t1.get(k)} == {'propagate'}
c, caps = spec['common'], spec['caps']
assert c['input_allowance'] + c['output_cap'] <= c['context_window']
reserve = (c['candidate_cap'] - 1) * (c['input_allowance'] + c['output_cap']) + c['context_window'] + c['output_cap']
assert reserve <= min(caps['per_arm_per_bug_logical'], caps['per_arm_per_bug_physical'])
assert 2 * caps['per_arm_per_bug_physical'] == caps['per_bug_physical']
assert caps['max_smoke_bugs'] * caps['per_bug_physical'] == caps['per_run_physical']
assert c['candidate_cap'] == caps['per_arm_per_bug_fresh_calls']
assert 2 * caps['per_arm_per_bug_fresh_calls'] == caps['per_bug_fresh_calls']
assert caps['max_smoke_bugs'] * caps['per_bug_fresh_calls'] == caps['per_run_fresh_calls']
assert c['system_sha256'] == hashlib.sha256(SYSTEM.encode()).hexdigest()
assert spec['revision'] == subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
assert spec['branch'] == subprocess.check_output(['git','branch','--show-current'], text=True).strip()

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert spec['master_prompt_sha256'] == digest(Path.home()/'Downloads/GRAFT_FL_Codex_Master_Prompt.md')
for row in spec['frozen_inputs']:
    path = Path(row['path'])
    if 'glob' in row:
        members = [{'path': str(p), 'sha256': digest(p)} for p in sorted(path.glob(row['glob']))]
        assert len(members) == row['files']
        actual = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    else:
        actual = digest(path)
    assert actual == row['sha256'], row['path']
for key in ('paths', 'repair'):
    bad = copy.deepcopy(spec)
    bad['common'][key] = True
    assert not validator.is_valid(bad)
bad = copy.deepcopy(spec)
bad['arms']['dense_sparse_with_propagation']['propagate'] = False
assert not validator.is_valid(bad)
bad = copy.deepcopy(spec)
bad['caps']['phase0_fresh_calls'] = 1
assert not validator.is_valid(bad)
print(f"PASS schema + one-flag diff + cap arithmetic ({reserve:,} conditional last-request headroom) + {len(spec['frozen_inputs'])} frozen hash entries + 4 rejected mutations")
```

**PASS.** Schema accepted the draft, T0/T1 differ by `propagate` only, conditional admission arithmetic gives 35,584 tokens, all 27 fingerprint entries matched, and all four negative cases were rejected. Two fingerprint entries represent the sorted path-plus-content digest lists for 97 manifests and 97 results files. The fingerprint checks verify the named records, and the archive itself remains an externally supplied historical input.

The intermediate `diff -u /tmp/graft-phase0-status-before.txt /tmp/graft-phase0-status-after.txt` exited 1 because the three draft files then existed. The final status assertion compared the status lines while excluding the four exact Phase 0 paths. It passed, and `git diff --exit-code HEAD` exited 0. No software test suite, real encoder smoke or model inference was run during this phase.

## Phase 1 regression record

The first focused command exited 0 with **14 passed**.

```bash
"$HOME/.pyenv/versions/3.11.0/bin/python" -B -m pytest -q tests/test_graft_fixed_evidence_schedule.py tests/test_graft_fixed_evidence_accounting.py
```

The first combined run exited 1 with **201 passed, 2 failed**. The legacy shell entry point used the default Python without NumPy. The initial socket guard also blocked the legacy test's own fake HTTP server. The rerun selected the project interpreter for the shell child and permitted socket connections only to the HTTP fixture server created by the tests. Legacy code and test assertions stayed intact.

```bash
"$HOME/.pyenv/versions/3.11.0/bin/python" -B - tests/test_graft_*.py tests/test_rave_eval.py tests/test_rave_fl*.py <<'PY'
from http.server import ThreadingHTTPServer
import os
import socket
import sys
import pytest

os.environ["RAVE_PYTHON"] = sys.executable
fixture_addresses = set()
bind = ThreadingHTTPServer.server_bind
connect = socket.socket.connect
connect_ex = socket.socket.connect_ex

def remember_fixture(server):
    bind(server)
    if server.server_address[0] == "127.0.0.1":
        fixture_addresses.add(server.server_address)

def only_fixture(original):
    def guarded(sock, address):
        if address not in fixture_addresses:
            raise AssertionError(f"Non-fixture connection in Phase 1: {address!r}")
        return original(sock, address)
    return guarded

ThreadingHTTPServer.server_bind = remember_fixture
socket.socket.connect = only_fixture(connect)
socket.socket.connect_ex = only_fixture(connect_ex)
raise SystemExit(pytest.main(["-q", *sys.argv[1:]]))
PY
```

**PASS.** Exit 0, **203 passed in 34.68s**. The tests include 14 fixed-evidence cases and 189 existing checks. Standalone CLI validation used eight simulated requests and 3,466 simulated tokens. The resume repeated zero requests. Real LLM calls and real verification tokens were both zero.

| Phase 1 file | SHA-256 |
|---|---|
| `rave_fl/graft/fixed_evidence_experiment.py` | `619e92fd159ef2d02f55f2e063c90e490fe9463ff8d5a040557a4a67623ecf39` |
| `tests/test_graft_fixed_evidence_schedule.py` | `a473efb8faa6d37d170628e82cc576e157619a6c0a8dc678d1ee948e6504545b` |
| `tests/test_graft_fixed_evidence_accounting.py` | `d7ccdd2dca24029bdab174f999c35e30a662ff4920eb76cb35c50c87b38c8754` |
| `configs/graft_fixed_evidence_dry_run.json` | `2105f0e17c71bd6550dc96d637085475589592f0f5811096355f33748ba4d7bd` |

**PASS.** Phase 1 worktree and baseline check found only the expected five new Phase 1 files, plus the update to this existing phase log. All 27 Phase 0 baseline fingerprint entries still match. No Linux, real provider, real encoder or scientific quality results were produced in this phase.

## Phase 2 validation record

The focused command in [Phase 2](phase02.md) returned exit 0 with **39 passed**. The final regression run used the fixture-only socket guard above with the same arguments (`tests/test_graft_*.py tests/test_rave_eval.py tests/test_rave_fl*.py`) and `RAVE_PYTHON` set to the active Python. It returned exit 0 with **223 passed in 35.82s**. The focused count is included in the regression count. The first node-order test run failed on T1 before the deterministic traversal fix. An early integrity test used a placeholder source revision that the existing schema rejected. The test now supplies a full 40-character fixture SHA.

The preregistration schema and both frozen arm configs matched. Of the 27 registered fingerprints, 25 matched the working tree, including all historical archive entries. The two deliberate changes matched the original baseline hashes when read from Git `HEAD`. Those two entries remain unchanged in the preregistration. The independent Linux run used the cached `gymbro-web:latest` image with `--pull never --network none` and a read-only checkout mount. It passed for all eight selections from frozen serialized graphs and bundles. The full dry-plan check in that container was blocked at Git discovery.

| Phase 2 code or test | SHA-256 |
|---|---|
| `rave_fl/graft/topology.py` | `1f661dd8242ca6af5edfb25bc1418765555f9cfd68d572e64327c48658468b38` |
| `rave_fl/graft/encode.py` | `2cdfc2bfd23e9a3b416d6be1f65bc2ee613775fb6e6317dd54ff0a6790e22a85` |
| `rave_fl/graft/fixed_evidence_experiment.py` | `f6dc889e514dc32eb1c9a8cdbb9b862949d69076ce0e7f4654d71103ad7fe915` |
| `tests/test_graft_topology.py` | `596427eed3fffc79408d2b2be65cdc0c89a016acdcfa104e37eedc9cb44e3b91` |
| `tests/test_graft_phase2_integrity.py` | `c4a06613ab2ea69594ec51b263986ea1a87d3da6eeba86278cce0dd01b48cce7` |
| `tests/test_graft_fixed_evidence_accounting.py` | `8657c16e33e949238148477cf6f5823b326c01584ec108401360e19e5c794792` |

The final standalone run used the same synthetic source and bundles as Phase 1. Both dry-runs exited 0 and `cmp` found identical bytes. Collection and resume both exited 0. The Phase 2 working files remain uncommitted on `GRAFT-FL` with the pending Phase 0 and Phase 1 files.

| Input or output | SHA-256 |
|---|---|
| Synthetic online manifest | `6c2feca42de64e924888f41060fe6399a83170a25ea7b87f388c53e87acebbc2` |
| Synthetic Java bundle file | `cb65bc13b8f721704e234a1bcee54f0a8a0b6425c7b3ec64c81061855c720898` |
| Synthetic Python bundle file | `d97fdb6db9efa07e9d26f9cf4527fc32b76997548a46d32f1ffec6ffc461a1ef` |
| `/tmp/graft-phase1-cli.jLs4c7/config.json` | `7921dd5583084f330f312fd8821861e567af99f22fb09ef1307fbcdf30896d28` |
| `/tmp/graft-phase2.x5Cqna/plan-final-a.json` and `plan-final-b.json` | `e280fdf53bff61a1655c80b91aa17f9de3e3a32d08da3f46d8f66f5dc4e4f3a1` |
| `/tmp/graft-phase2.x5Cqna/run-final/manifest.json` | `75319d33fab658e03f4a477b5a860b9adeddd75249e50649958777baf73781ca` |
| `/tmp/graft-phase2.x5Cqna/run-final/results.json` | `6b356c8d35d0ddd03599e44ba490828ea4b887ca8535576a9c940150ca5d471d` |
| [Synthetic evidence comparison](phase02_synthetic_evidence.json) | `a6bdc5d3e78c896d4141941731549ed2b43f443de350d955420643a60b5245a1` |

**Real encoder gate BLOCKED.** Provide an existing complete local BGE-M3 snapshot and the `FlagEmbedding` dependency to validate a real bundle and its paired evidence. The provider chat-template token gate also remains open. No real LLM calls, model downloads, large experiment runs or Phase 3 work occurred.

## Phase 2.5 validation record

**PASS.** The user authorized the 2.30 GB pinned checkpoint transfer. All eight official files verified. The Mac CPU exporter generated 1,024-dimensional dense and sparse lexical embeddings for the same 16 exact fixture inputs as the Phase 2 plans. Fresh CPU loads gave identical vectors and weights. MPS passed the bounded FP32 Python-fixture check with CPU fallback disabled. The two arms consumed the same real bundle for each defect and reproduced all four paired selections under separate hash seeds. Linux replayed all eight arm/target selections with exact evidence IDs, rendered hashes, and proxy sizes. A100 GPU encoding remains unverified.

No existing dependency versions or dependency-check messages changed. The focused suite passed 39 tests in 2.90 s. The fixture-guarded regression suite passed 223 tests in 35.95 s. The frozen preregistration and Phase 2 scientific implementation files retained their hashes. There were zero LLM calls and zero provider tokens. The [Phase 2.5 report](phase025.md) lists the observed changes, reproducible commands and artifact hashes. The [checkpoint manifest](phase025_checkpoint.json) pins the verified files. Working files remain uncommitted on `GRAFT-FL`.

## Phase 3 validation record

**PASS.** The [Phase 3 report](phase03.md) lists the prompt audit, serial bound, signed paired deltas, and open deployment gates. A saved real-bundle fixture plan reproduced all eight Phase 2.5 request payloads and selected evidence. Independent hash seeds 7 and 19 reproduced the new plan bytes. The focused command returned exit 0 with **53 passed in 9.00 s**. The regression command above ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and the same local fixture socket guard. It returned exit 0 with **237 passed in 41.25 s**. A network-disabled Linux aarch64 container independently reproduced all 16 arm/target prompt audits from the two saved plans.

The scientific preregistration, topology, encoder, renderer and packer were verified unchanged. The source diff in `selection.py` is confined to the optional local tokenizer counter. The shared backend and response parser changes handle malformed paid replies. Real-provider reservation and the strict context check retain their original bounds. Earlier results and manifests remain intact. Working files remain uncommitted on `GRAFT-FL`.

| Phase 3 code, test or artifact | SHA-256 |
|---|---|
| `rave_fl/graft/fixed_evidence_experiment.py` | `e2aa291621bcb34ad423d3eed38f57e78af458abca5fa6e467d3357d2bdc3c38` |
| `rave_fl/graft/selection.py` | `8082cf090b43faed8f644bb9b08559d4d02234375927a9161c7b83103ad15517` |
| `rave_fl/eval/backend.py` | `4bf80ca3d830ba1248cdd72265b9dd7e030462ef4881def3b09317c5fde87032` |
| `rave_fl/pilot_backend.py` | `2854019e7cfc34ea43c72899032c84f36b94a8e2312e3fc1b054787e36bd29ff` |
| `tests/test_graft_phase3_admission.py` | `c01028708fe28c06d6a634cb3efa9366e5f94bb1d61902fa58eb36263c0167e1` |
| `results/graft_fl_phase3_fixture/real-embedding-plan.json` | `947213ecab8faabe91b10c718af731ba97b0abc6c523d34221997a95a1203ba4` |
| `results/graft_fl_phase3_fixture/real-backend-dry-plan.json` | `7b3cde796937177d9110b043a5808e35a9e772f4bf8d28336a16f619a1172663` |
| `results/graft_fl_phase3_fixture/tight-synthetic-run/results.json` | `dfb1a204f4781004272cd4a945194f035bfd56265112d6e580bd8069b5b17197` |
| `results/graft_fl_phase3_fixture/phase3_summary.json` | `63fba43136235436adb52826be455e2fd6eb5b5152fc1cacd5eb49810e73af3e` |
| `results/graft_fl_phase3_fixture/linux-admission.json` | `4f79b001d47609731f57998498cf465fec2b476d1e053373c392aac2d8d6341c` |

## Phase 4 validation record

**PASS.** [Phase 4](phase04.md) records the cohort definitions, rehearsal outcomes, open deployment gates and exact reproduction commands. The focused command exited 0 with **44 passed in 21.66s**. The regression suite used the preceding fixture-only socket guard, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, and the same `tests/test_graft_*.py tests/test_rave_eval.py tests/test_rave_fl*.py` selection. It exited 0 with **246 passed in 56.90s**. Independent offline regeneration matched all 15 report files byte-for-byte, including the five PNG/SVG pairs. Linux aarch64 independently reconciled all 24 saved calls and matched the Mac aggregates. Phase 4 issued zero real LLM requests and zero real provider tokens.

The fixed runner and scientific preregistration retained their Phase 3 digests. The new report entry point reads saved manifests independently of the current executable fingerprint, and earlier archives remain intact. The existing runner continues to reject resume when its executable fingerprint differs. Working files remain uncommitted on `GRAFT-FL`.

| Phase 4 file or retained artifact | SHA-256 |
|---|---|
| `rave_fl/graft/fixed_evidence_reporting.py` | `c3a3db0097d4facf1bdb3aec6873f7753c42995c801ca0e153042eab7659fe27` |
| `rave_fl/graft/fixed_evidence_figures.py` | `652f9f4137e090a1a2afdf2f9987940bf9f660f3f8e40cc40c9137585aa8d436` |
| `scripts/rehearse_graft_fixed_evidence.py` | `367fcc3f8022bdb2c8951a1876e702092a1d919f2399e0f829b34a8e18f08739` |
| `tests/test_graft_phase4_reporting.py` | `c384b916956320f971813f89aa05e861ae880ccf639933c0fd90aec8757dd82a` |
| `results/graft_fl_phase4_final/rehearsal.json` | `ccc85056354a4a6b707425acfd7806d302255c25a7b64afc53ff8b04b58804ec` |
| `results/graft_fl_phase4_final/run/manifest.json` | `58ea12de0bafd4613aa4c1a2cb4be19553a8e854cb5676ab1939830e8237d25e` |
| `results/graft_fl_phase4_final/run/results.json` | `22c57e1ccd5612ec4ed5df03da3f6efc965ebe8d067622fe54b744ed6942f04e` |
| `results/graft_fl_phase4_final/report/summary.json` | `a61b6e2a911e9f86e6a61f44027e9e42357d484bc1e6511df5b980881727dd37` |

## Phase 5 validation record

**PASS for offline portability.** The [Phase 5 note](phase05.md) and [portable runbook](PORTABLE_RUNBOOK.md) describe the accepted commands and remaining real-execution gates. The focused suite exited 0 with **46 passed in 25.75s**. The full regression selection used the preceding fixture-only socket guard and offline environment flags. It exited 0 with **255 passed in 60.19s**. Both synthetic profiles generated an identical cohort lock. The two shards used eight simulated calls and 3,343 simulated tokens, and both resumes made zero new calls. The merged files were byte-identical with reversed shard arguments. The launcher also regenerated the sealed Phase 4 report byte-for-byte.

The 100-entry clone manifest exercised ten disjoint shards and executed zero calls. Linux aarch64 independently audited the two saved synthetic shards and both lock layouts. Real LLM calls, model loads, additional downloads and server jobs were all zero. The preregistration SHA-256 remains `52bd384d9cfbb7b3a75e7b94e5f8eef60b26c48dd24961e34d0a3fd7b33d2389`. The fixed execution diff adds only manifest binding and input verification. Working files remain uncommitted on `GRAFT-FL`.

| Phase 5 file or retained artifact | SHA-256 |
|---|---|
| `rave_fl/graft/portable_experiment.py` | `71e2c7a3dc84f797a992e332e01bf116fc61fbc5b79460006b4fc188a02ce2d0` |
| `rave_fl/graft/fixed_evidence_experiment.py` | `b06e28d869da238c65d653ecbc4aa3a490207dd92c19713a3c495d585ffbf078` |
| `rave_fl/graft/fixed_evidence_reporting.py` | `5c64841c889182441da44282c6492410f1f37b5d64d5ca115b2894a294e656a3` |
| `tests/test_graft_phase5_portable.py` | `24bf3264f64eef82bc469f05295ce8c3c16ecb206ab91869b1c43712ae848d36` |
| `configs/graft_portable/local_mac.json` | `cf86520a5292dc768839799e8d9ff3ba957f5bfa716578572c2e80d1e839732d` |
| `configs/graft_portable/remote_linux_a100.json` | `fcacfd047d732224ed56799092a815e645ce517c3af6a2f79986687400357f34` |
| `results/graft_fl_phase5_fixture/local_mac/cohort.json` | `a468f4ddf415cc2b7875e30ee4cf2ba7734d3db33abb1f3ef19b10d5066de8bc` |
| `results/graft_fl_phase5_fixture/large-dry/cohort.json` | `a15cf5de6ddab192a0c2350332141cf8c3f641fe7109ab9da00b2324043110ab` |
| `results/graft_fl_phase5_fixture/merged/summary.json` | `59f484212f111d84ce11b71c917e1ccda8457b3e2b7ed23b56dc98758cf8f30b` |
| `results/graft_fl_phase5_fixture/linux-audit.json` | `48fe54fe1c7edff337ce5db78fd2385d4937b87d66545b7d81a676d9e8342bfb` |
| `results/graft_fl_phase5_local_model_dry/preflight.json` | `171c55fa22e53acd04d359c5e20f20e77a1fae4490f3c15b3fe192aa0673b555` |

## Phase 6 validation record

**PASS for offline engineering.** The [Phase 6 report](phase06.md) documents pinned archive evidence, installed-template checks, relocation safeguards and exact reproduction commands. The focused suite exited 0 with **29 passed in 15.04s**. The same fixture-only regression guard and offline environment flags as Phase 5 returned exit 0 with **261 passed in 67.21s**. Linux aarch64 reproduced both rendered fixture targets after path relocation. The installed Llama and Qwen model digests formed separate offline conditions with identical frozen arm settings.

**BLOCKED for real evaluation.** The Lang-54 and Math-80 revision mappings, archival trigger traces and production patches matched their local pinned Defects4J objects. Closure-61’s corresponding blobs were unavailable offline. No target had a verified ready checkout, fresh failure replay, complete candidate pool or independently confirmed method gold. The installed serving template bytes were verified, but exact token-ID parity with the served build remains open. No real provider requests or benchmark runs were made. The preregistration digest remained unchanged. Changes remain uncommitted on `GRAFT-FL`.

| Phase 6 file or retained artifact | SHA-256 |
|---|---|
| `rave_fl/graft/fixed_inputs.py` | `6471b916f948cc882afdf837498987e496b9b55f38996ab6a1a87cbeb260039c` |
| `rave_fl/graft/fixed_evidence_experiment.py` | `d2a39f6030f6bc1cf76e6f7187412bf0dd26101c3f11ad0f2235b115d3812336` |
| `rave_fl/graft/portable_experiment.py` | `7c10ca3cffa3a483d5ddb00bedfd0cba29f131df90120cee9d72ba041bc27fbb` |
| `rave_fl/graft/selection.py` | `a8187a88e83899fa4e003f93b0587b7db396fb5b98af8ca1f607b783115abc29` |
| `scripts/audit_graft_phase6.py` | `8b067b01fbd22e2fd4fe73ac8adc546d90175f887418683bb3f3307944b4d7ac` |
| `tests/test_graft_phase6_readiness.py` | `8cd2ee06ec8cad0673d1d7dccd34f9e7bf3e25e7b4ca974ac28c6b3b63b0021e` |
| `results/graft_fl_phase6/defects4j-final/defects4j-readiness.json` | `7539be14b29748cab301a5b877ab6c9293e91bc0983e8f2929c8c85a5c4e42bf` |
| `results/graft_fl_phase6/serving-final/serving-readiness.json` | `fccedad5dfbe419a90d5f048ff6d2c6a15313ad7e50a81047cd6d9244516d5a2` |
| `results/graft_fl_phase6/linux-relocation.json` | `304de8b35fd1e618c2b6f320dcf01db2b010382cc67f0f3d5160e06d7feca5d3` |

## Real-data preparation follow-up

The [preparation report](REAL_DATA_PREPARATION.md) records the exact first-batch resources pending download approval, individual readiness gates and manual commands. The guarded regression command from Phase 6 passed **266 tests in 63.24s**. The focused suite passed **34 tests in 12.75s**. The source verifier matched synthetic candidate spans and independently derived gold against pinned Git inputs and preserved a distinct offline label file. The Mac fixture dry-run verified the installed model artifacts and BGE-M3 layout. The proposed real-data dry-run exited 3 with a missing-manifest diagnostic. The installed serving-template audit reproduced the Phase 6 bytes. No downloads, server jobs or provider calls occurred. The preregistration hash and existing scientific implementation are unchanged.

| Follow-up file or artifact | SHA-256 |
|---|---|
| `scripts/verify_graft_real_sources.py` | `5968bfa339b22f6d4f3078c4a1b7214aa0c13d0f44222800db4a13bb230be3e9` |
| `tests/test_graft_real_preparation.py` | `6ef1d948031ae211d67144d43c41c4fc906ea8b1f445cf5ce317d36729d41eee` |
| `configs/graft_portable/real_mac_preflight.json` | `e617ffec3c370532501e2d6a8bb866108ed43778cf2cd2cfbc7f71e45720d86f` |
| `results/graft_fl_real_preparation/download-inventory.json` | `44c4a6495d6ba77dc73f65705a565693f66756a6f3ed1ac85bc3182debcd323b` |
| `results/graft_fl_real_preparation/real-data-dry-run/preflight.json` | `15ea4cbac2fef2f75550c019bc9c0d5d099f46403473a71491a6290804ab99dd` |
| `results/graft_fl_real_preparation/fixture-dry-run/preflight.json` | `cbfa1867088f895cc6fe296c9f373fdff7c34c1b12d96dba4ebe162f923ff292` |
| `results/graft_fl_real_preparation/serving/serving-readiness.json` | `fccedad5dfbe419a90d5f048ff6d2c6a15313ad7e50a81047cd6d9244516d5a2` |
| `results/graft_fl_real_preparation/fixture-proxy-1024/plans/shard-000.json` | `7a34c65e546b6a114cae4f1fe2dfc88e67ad902caa15b448436bf41dac731d72` |
