"""Capture native vocabulary-only token IDs for an immutable fixed-evidence plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import struct
import subprocess
import tempfile
from unittest.mock import patch

from rave_fl.eval.schema import file_hash
from rave_fl.graft import fixed_evidence_experiment as fixed
from rave_fl.graft.prompt_settlement import MAC_BINARY_SHA256, MODEL_DIGEST, TEMPLATE, VERSION, require
from scripts.build_graft_settlement_certificate import rendered


# This ABI was validated for the audited arm64 Mac build. Porting the
# vocabulary-only capture to another serving binary needs an independent audit.


class AuditCounter:
    def __init__(self, base, captures):
        self.base = base
        self.captures = captures
        self.name = base.name
        self.exact = base.exact

    def count(self, messages):
        prompt = rendered(messages)
        if prompt not in self.captures:
            self.captures[prompt] = self.base.token_ids(messages)
        return len(self.captures[prompt])


def verify_local_inputs(binary, manifest, model_dir):
    require((platform.system(), platform.machine()) == ("Darwin", "arm64")
            and file_hash(binary) == MAC_BINARY_SHA256
            and plistlib.loads((binary.parents[1] / "Info.plist").read_bytes())["CFBundleVersion"] == VERSION,
            "Native capture is supported on the audited Ollama 0.17.7 arm64 Mac build")
    require(file_hash(manifest) == MODEL_DIGEST, "Installed model manifest differs from the frozen digest")
    value = json.loads(manifest.read_text())

    def verified(layer):
        sha = layer["digest"].removeprefix("sha256:")
        blob = model_dir / "blobs" / f"sha256-{sha}"
        require(blob.stat().st_size == layer["size"] and file_hash(blob) == sha, "Installed model layer drift")
        return blob, sha

    layers = {r["mediaType"].rsplit(".", 1)[-1]: r for r in value["layers"]}
    _, template = verified(layers["template"])
    weights, model = verified(layers["model"])
    cfg, _ = verified(value["config"])
    verified(layers["params"])
    config = json.loads(cfg.read_text())
    require(template == TEMPLATE and config["model_family"] == "llama"
            and not any(config.get(k) for k in ("renderer", "remote_host", "remote_model")),
            "Model does not use the audited no-tools compatibility path")
    return weights, model


def native_token_ids(prompts, binary, weights, output):
    """Call the audited llama tokenizer on a disposable copy without starting Ollama serve."""
    with tempfile.TemporaryDirectory(prefix="graft-vocab-") as directory:
        tmp = Path(directory)
        copy = tmp / "ollama"
        shutil.copy2(binary, copy)

        def executable_text(path):
            return subprocess.check_output(["otool", "-arch", "arm64", "-s", "__TEXT", "__text", path]).splitlines()[2:]

        ent = tmp / "debug.plist"
        ent.write_bytes(plistlib.dumps({"com.apple.security.get-task-allow": True}))
        subprocess.run(["codesign", "--force", "--sign", "-", "--entitlements", ent, copy],
                       check=True, capture_output=True)
        require(executable_text(copy) == executable_text(binary) and file_hash(binary) == MAC_BINARY_SHA256,
                "Disposable executable differs from the audited binary")
        incoming, outgoing = tmp / "in", tmp / "out"
        with incoming.open("wb") as stream:
            for prompt in prompts:
                require(0 < len(prompt) < 16384, "Rendered capture exceeds the native input buffer")
                stream.write(struct.pack("<i", len(prompt)) + prompt)
        # Validated v0.17.7 arm64 layout. The loader skips tensors and the
        # debugger only calls vocabulary/tokenizer functions before exit.
        commands = [
            "breakpoint set -n write", "run help", "breakpoint disable 1",
            "expression -l c++ -- struct Params { void *devices; void *tensor_overrides; int layers; int split; "
            "int gpu; const float *tensor_split; void *progress; void *progress_data; const void *overrides; "
            "bool vocab_only; bool mmap; bool mlock; bool check_tensors; bool extra_buffers; bool no_host; bool no_alloc; }; "
            "static_assert(sizeof(Params)==72 && __builtin_offsetof(Params,vocab_only)==64); "
            "((void (*)(void))llama_backend_init)(); auto p=((Params (*)(void))llama_model_default_params)(); "
            "void *none[1]={nullptr}; p.devices=none; p.vocab_only=true; p.layers=0; p.mmap=false; p.extra_buffers=false; "
            f"void *$model=((void *(*)(const char *, Params))llama_model_load_from_file)({json.dumps(str(weights))},p)",
            "expression -l c++ -- void *$vocab=((void *(*)(void *))llama_model_get_vocab)($model)",
            "expression -t 90000000 -l c++ -- "
            f"void *fi=((void *(*)(const char *, const char *))fopen)({json.dumps(str(incoming))},\"rb\"); "
            f"void *fo=((void *(*)(const char *, const char *))fopen)({json.dumps(str(outgoing))},\"wb\"); "
            "auto rd=(unsigned long (*)(void *,unsigned long,unsigned long,void *))fread; "
            "auto wr=(unsigned long (*)(const void *,unsigned long,unsigned long,void *))fwrite; "
            "char input[16384]; int ids[16386]; int n=0; int size=0; int done=0; "
            "while(fi && fo && rd(&size,4,1,fi)==1 && size>0 && size<16384 && rd(input,1,size,fi)==size) { "
            "n=((int (*)(void *,const char *,int,int *,int,bool,bool))llama_tokenize)($vocab,input,size,ids,16386,true,true); "
            "if(n<=0 || wr(&n,4,1,fo)!=1 || wr(ids,4,n,fo)!=n) break; ++done; } "
            "if(fi) ((int (*)(void *))fclose)(fi); if(fo) ((int (*)(void *))fclose)(fo); done",
            "expression -l c++ -- ((void (*)(void *))llama_model_free)($model)", "process kill",
        ]
        script = tmp / "tokenize.lldb"
        script.write_text("\n".join(commands) + "\n")
        run = subprocess.run(["lldb", "--batch", "-s", script, copy], capture_output=True, timeout=160)
        log = run.stdout + run.stderr
        (output / "native-tokenizer.log").write_bytes(log)
        run.check_returncode()
        require(b"vocab only - skipping tensors" in log, "The native process did not attest vocabulary-only loading")
        with outgoing.open("rb") as stream:
            for _ in prompts:
                raw = stream.read(4)
                require(len(raw) == 4, "Incomplete native token capture")
                size, = struct.unpack("<i", raw)
                require(0 < size <= 16386, "Invalid native token sequence size")
                yield list(struct.unpack(f"<{size}i", stream.read(4 * size)))
            require(not stream.read(1), "Unexpected extra native token capture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "serving-binary", "generation-manifest", "ollama-models-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1",
                      GIT_OPTIONAL_LOCKS="0", TOKENIZERS_PARALLELISM="false")
    settings = json.loads(args.config.read_text())
    config = fixed.configuration(settings)
    opts = config["benchmark"]
    require(not config.get("prompt_settlement") and config["tokenizer_dir"] and opts["backend"] == "ollama"
            and opts["sample"] == "first" and opts["limit"] is None
            and opts["bugs"] == sorted(set(opts["bugs"])) and 0 < len(opts["bugs"]) <= 3,
            "Capture requires an explicit sorted, unsealed Ollama schedule of at most three defects")
    weights, model_sha = verify_local_inputs(args.serving_binary, args.generation_manifest, args.ollama_models_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    captures = {}
    resources = fixed.resources

    def traced_resources(cfg, defect):
        counter, bundle = resources(cfg, defect)
        return AuditCounter(counter, captures), bundle

    with patch.object(fixed, "resources", traced_resources):
        plan, _ = fixed.prepare(config)
    require(plan["new_model_calls"] == 0, "Capture plan must be provider-free")
    plan_path = args.output / "paired-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    require(all(all(row.get("request") for rows in p["arms"].values() for row in rows)
                and p["schedule"] for p in plan["pairs"]), "At least one pair has missing or overflowing requests")
    rows = []
    for (prompt, local), native in zip(captures.items(),
                                     native_token_ids(captures, args.serving_binary, weights, args.output), strict=True):
        rows.append({"rendered_sha256": hashlib.sha256(prompt).hexdigest(), "tokens": len(native),
                     "token_ids": native, "matches": local == native})
    mismatch = [row["rendered_sha256"] for row in rows if not row["matches"]]
    receipt = {"schema": "graft-native-schedule-audit-v1", "capture_plan_sha256": file_hash(plan_path),
               "model_sha256": MODEL_DIGEST, "template_sha256": TEMPLATE, "model_blob_sha256": model_sha,
               "serving_binary_sha256": MAC_BINARY_SHA256, "system": platform.system(), "architecture": platform.machine(),
               "requests": len(rows), "mismatches": mismatch, "rows": rows}
    (args.output / "native-comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")
    require(not mismatch, "Native token IDs differ from the local packing counter")
    print(f"{len(rows)} native token sequences matched; zero generation requests")


if __name__ == "__main__":
    main()
