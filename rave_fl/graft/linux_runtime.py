"""Validation for the pinned Linux serving and vocabulary-only build."""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint

OLLAMA_SOURCE = "9b0c7cc7b90562b370ce6a30efdc667326799223"


def need(ok, message):
    if not ok:
        raise CompatibilityError(message)


def stream_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def linux_build(binary, *, receipt_sha=None):
    binary = Path(binary).resolve()
    root = binary.parents[1]
    path = root / "graft-build.json"
    value = json.loads(path.read_bytes())
    scripts = Path(__file__).resolve().parents[2] / "scripts/graft_server"
    libs = {p.relative_to(root).as_posix(): stream_hash(p)
            for p in sorted((root / "lib/ollama").rglob("*")) if p.is_file()}
    need(platform.system() == "Linux" and value["schema"] == "graft-ollama-linux-build-v1"
         and value["source_commit"] == OLLAMA_SOURCE
         and value["audit_source_sha256"] == file_hash(scripts / "native_tokenizer.go")
         and value["build_script_sha256"] == file_hash(scripts / "build_ollama.sh")
         and value["binary_sha256"] == stream_hash(binary)
         and value["libraries"] == libs and any("cuda" in p for p in libs)
         and (receipt_sha is None or file_hash(path) == receipt_sha),
         "Ollama Linux binary, audited source or CUDA library inventory differs from its build receipt")
    return {"linux_build_sha256": file_hash(path), "ollama_source_commit": OLLAMA_SOURCE,
            "library_inventory_sha256": fingerprint(libs)}


def nvidia_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        text=True, timeout=10)
    rows = [[item.strip() for item in line.split(",")] for line in output.splitlines()]
    return sorted(({"uuid": u, "name": name, "driver": driver, "compute_cap": cc}
                   for u, name, driver, cc in rows if "A100" in name and cc == "8.0"), key=lambda gpu: gpu["uuid"])


def runner_gpu(listener_pid, binary, gpus, model_path):
    """Match an actual CUDA compute process to the compatibility runner of our listener."""
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"], text=True, timeout=10)
    match = []
    allowed = {gpu["uuid"] for gpu in gpus}
    for line in rows.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or fields[0] not in allowed or not fields[1].isdigit() or not fields[2].isdigit():
            continue
        uuid, pid, mib = fields
        proc = Path("/proc") / pid
        try:
            parent = re.search(r"^PPid:\s+(\d+)", (proc / "status").read_text(), re.M)
            args = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            same = (proc / "exe").samefile(binary)
            # The runner is expected to map the audited GGML CUDA library,
            # even on hosts with other globally installed Ollama versions.
            maps = (proc / "maps").read_text().splitlines()
            cuda_paths = {Path(line.split(maxsplit=5)[5]).resolve() for line in maps if "libggml-cuda.so" in line}
        except (OSError, ValueError):
            continue
        if (parent and int(parent[1]) == listener_pid and same and len(args) > 1 and args[1] == "runner"
            and "--ollama-engine" not in args and "--model" in args
            and args.index("--model") + 1 < len(args)
            and Path(args[args.index("--model") + 1]).resolve() == model_path and int(mib) > 0
            and cuda_paths and all(path.is_file() and path.is_relative_to(binary.parents[1] / "lib/ollama")
                                   for path in cuda_paths)):
            match.append({"pid": int(pid), "gpu_uuid": uuid, "allocated_mib": int(mib),
                          "cuda_libraries": sorted(str(p.relative_to(binary.parents[1])) for p in cuda_paths)})
    need(bool(match), "No attested Llama compatibility runner on an A100 GPU")
    return match
