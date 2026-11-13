"""Local, attested Ollama transport for the fixed-evidence collector."""
import ctypes
from dataclasses import dataclass
from contextlib import closing
import hashlib
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
from urllib.parse import urlsplit

from rave_fl.graft.controller import GraftProvider
from rave_fl.graft.linux_runtime import nvidia_inventory, runner_gpu
from rave_fl.graft.prompt_settlement import MODEL, RUNNER, SettlementCertificate, require


def lsof_records(*args):
    output = subprocess.run(["lsof", "-nP", *args], check=True, capture_output=True,
                            text=True, timeout=5).stdout
    pid, records = None, []
    for line in output.splitlines():
        code, value = line[0], line[1:]
        if code == "p":
            pid = int(value)
        elif code == "f":
            records.append({"pid": pid, "fd": value})
        elif records:
            records[-1][code] = value
    return records


def process_arguments(pid, system):
    if system == "Linux":
        root = Path("/proc") / str(pid)
        return ((root / "cmdline").read_bytes().rstrip(b"\0").split(b"\0"),
                (root / "environ").read_bytes().rstrip(b"\0").split(b"\0"))
    require(system == "Darwin", "Unsupported live listener inspection host")
    # CTL_KERN / KERN_PROCARGS2 supplies the executable arguments and initial
    # environment for this same-user process. Never emit the environment.
    sysctl = ctypes.CDLL(None, use_errno=True).sysctl
    sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    mib = (ctypes.c_int * 3)(1, 49, pid)
    size = ctypes.c_size_t()
    require(sysctl(mib, 3, None, ctypes.byref(size), None, 0) == 0 and 4 < size.value <= 1024 * 1024,
            "Cannot read listener argument size")
    data = ctypes.create_string_buffer(size.value)
    require(sysctl(mib, 3, data, ctypes.byref(size), None, 0) == 0, "Cannot read listener arguments")
    raw = data.raw[:size.value]
    count, = struct.unpack_from("=i", raw)
    parts = raw[raw.index(b"\0", 4) + 1:].lstrip(b"\0").split(b"\0")
    require(0 < count < len(parts) and all(parts[:count]), "Incomplete listener arguments")
    return parts[:count], parts[count:]


def listener(port, binary, models_dir):
    address = f"127.0.0.1:{port}"
    rows = lsof_records(f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpfint")
    require(len(rows) == 1 and rows[0].get("n") == address, "Endpoint needs one direct IPv4 loopback listener")
    pid = rows[0]["pid"]
    ps = subprocess.check_output(["ps", "-p", str(pid), "-o", "uid=", "-o", "lstart="],
                                 text=True, timeout=5).strip().split(maxsplit=1)
    require(len(ps) == 2 and int(ps[0]) == os.getuid(), "Listener must run under the current user")
    system = platform.system()
    stat = binary.stat()
    if system == "Linux":
        running = (Path("/proc") / str(pid) / "exe").stat()
        require((running.st_dev, running.st_ino) == (stat.st_dev, stat.st_ino),
                "Running executable differs from the audited binary")
    else:
        text = lsof_records("-a", "-p", str(pid), "-d", "txt", "-FfinD")
        require(any(row.get("n") == str(binary) and int(row["i"]) == stat.st_ino
                    and int(row["D"], 16) == stat.st_dev for row in text if "i" in row and "D" in row),
                "Running executable differs from the audited binary")
    args, entries = process_arguments(pid, system)
    require(len(args) == 2 and args[1] == b"serve"
            and Path(os.fsdecode(args[0])).resolve() == binary, "Listener is not the audited Ollama serve process")
    env = {}
    for item in entries:
        key, sep, value = item.partition(b"=")
        if sep and key in {b"HOME", b"OLLAMA_MODELS", b"OLLAMA_NEW_ENGINE", b"LD_LIBRARY_PATH", b"LD_PRELOAD",
                           b"OLLAMA_LIBRARY_PATH", b"CUDA_VISIBLE_DEVICES", b"OLLAMA_KV_CACHE_TYPE",
                           b"OLLAMA_FLASH_ATTENTION", b"OLLAMA_NUM_PARALLEL", b"OLLAMA_KEEP_ALIVE"}:
            require(key not in env, "Ambiguous Ollama process environment")
            env[key] = os.fsdecode(value).strip().strip("\"'").strip()
    require(env.get(b"OLLAMA_NEW_ENGINE", "").lower() in {"", "0", "false", "f"},
            "Compatibility runner mode cannot be established")
    require(env.get(b"OLLAMA_MODELS") or env.get(b"HOME"), "Listener model store is unavailable")
    model_root = Path(env[b"OLLAMA_MODELS"]) if env.get(b"OLLAMA_MODELS") else Path(env[b"HOME"]) / ".ollama/models"
    require(model_root.resolve() == models_dir, "Listener uses a different model store")
    linux_settings = {}
    if system == "Linux":
        require(not any(env.get(k) for k in (b"LD_LIBRARY_PATH", b"LD_PRELOAD", b"OLLAMA_LIBRARY_PATH"))
                and env.get(b"OLLAMA_NUM_PARALLEL") == "1"
                and env.get(b"OLLAMA_KEEP_ALIVE") == "-1"
                and re.fullmatch(r"GPU-[0-9a-fA-F-]+", env.get(b"CUDA_VISIBLE_DEVICES", "")) is not None
                and env.get(b"OLLAMA_KV_CACHE_TYPE", "f16") == "f16"
                and env.get(b"OLLAMA_FLASH_ATTENTION", "0") in {"0", "false"},
                "Use the runbook's dedicated Ollama environment without library overrides")
        linux_settings = {"cuda_visible_devices": env.get(b"CUDA_VISIBLE_DEVICES"),
                          "parallel": 1, "keep_alive": "-1", "kv_cache_type": "f16", "flash_attention": False}
    return {"pid": pid, "started": ps[1], "runner": RUNNER,
            "executable_device": stat.st_dev, "executable_inode": stat.st_ino,
            "executable_mtime_ns": stat.st_mtime_ns, "executable_ctime_ns": stat.st_ctime_ns,
            **({"linux_settings": linux_settings} if linux_settings else {})}


@dataclass(frozen=True)
class LiveOllamaPaths:
    binary: Path
    manifest: Path
    models_dir: Path

    def provider(self, config):
        certificate = SettlementCertificate(config)
        certificate.verify_live_caps(config)
        return AttestedOllama(config["benchmark"], certificate, self.binary, self.manifest, self.models_dir)


class AttestedOllama(GraftProvider):
    def __init__(self, opts, certificate, binary, manifest, models_dir):
        super().__init__(opts)
        url = urlsplit(self.endpoint)
        require(url.scheme == "http" and url.netloc == f"127.0.0.1:{url.port}" and url.port
                and not (url.path or url.query or url.fragment),
                "Attested Ollama requires a direct http://127.0.0.1:PORT endpoint")
        require(binary and manifest and models_dir, "Live Ollama requires serving binary, manifest and model store paths")
        self.port = url.port
        self.certificate = certificate
        self.binary, self.manifest, self.models_dir = (Path(path).resolve() for path in (binary, manifest, models_dir))
        require(self.manifest == self.models_dir / "manifests/registry.ollama.ai/library/llama3.1/8b",
                "Manifest must be the active Llama tag in the listener's model store")

    def check_listener(self):
        return listener(self.port, self.binary, self.models_dir)

    def http_json(self, path, payload=None, *, before_send=None):
        timeout = self.opts["timeout"] if path == "/api/chat" else 10
        with closing(HTTPConnection("127.0.0.1", self.port, timeout=timeout)) as connection:
            connection.connect()
            (before_send or self.assert_stable)()
            data = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
            connection.request("POST" if data is not None else "GET", path, data,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            require(response.status == 200, f"Ollama {path} returned HTTP {response.status}")
            raw = response.read(16 * 1024 * 1024 + 1)
            require(len(raw) <= 16 * 1024 * 1024, "Ollama response exceeds read limit")
            result = json.loads(raw)
            require(isinstance(result, dict), "Ollama response must be a JSON object")
            return result

    def get(self, path):
        return self.http_json(path)

    def local_model_state(self):
        paths = (self.manifest, *self.model_paths)
        return tuple((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                     for s in (path.stat() for path in paths))

    def preflight(self):
        self.process = self.check_listener()
        self.model_paths = self.certificate.local_model_paths(self.manifest, self.models_dir)
        self.model_state = self.local_model_state()
        self.certificate.verify_local_host(self.binary, self.manifest, self.models_dir, platform.system(), platform.machine())
        self.a100_gpus = self.visible_a100()
        require(self.opts.get("device") != "a100" or self.a100_gpus, "No NVIDIA A100 (compute capability 8.0) visible")
        self.metadata = self.verify_endpoint()
        if self.a100_gpus:
            self.metadata["a100_inventory"] = self.a100_gpus
        return self.metadata

    def visible_a100(self):
        if self.opts.get("device") != "a100":
            return []
        selected = self.process["linux_settings"]["cuda_visible_devices"]
        return [gpu for gpu in nvidia_inventory() if gpu["uuid"] == selected]

    def assert_stable(self):
        require(self.check_listener() == self.process, "Ollama listener changed after attestation")
        require(self.local_model_state() == self.model_state, "Ollama model files changed after attestation")

    def matches_cohort(self, original):
        if not self.a100_gpus:
            return self.metadata == original

        def stable(attestation):
            return {**attestation, "listener": {k: v for k, v in attestation["listener"].items()
                                                if k not in {"pid", "started"}}}

        # A fresh Linux process must reattest the same executable, files,
        # settings and GPU inventory. The original cohort and claim stay sealed.
        return stable(self.metadata) == stable(original)

    def verify_endpoint(self):
        self.assert_stable()
        identity = self.certificate.identity
        version = self.get("/api/version")["version"]
        matches = [row for row in self.get("/api/tags")["models"] if row.get("name") == MODEL]
        require(version == identity["serving_version"] and len(matches) == 1
                and matches[0].get("digest") == identity["model_manifest_sha256"],
                "Active Ollama version or model digest differs from audit")
        show = self.http_json("/api/show", {"model": MODEL})
        template = show.get("template")
        from_lines = re.findall(r"^FROM (.+)$", show.get("modelfile", ""), re.MULTILINE)
        require(isinstance(template, str) and hashlib.sha256(template.encode()).hexdigest() == identity["template_sha256"]
                and from_lines == [str(self.model_paths[0])] and not show.get("system") and not show.get("messages")
                and show.get("model_info", {}).get("general.architecture") == "llama"
                and set(show.get("capabilities", [])) == {"completion", "tools"},
                "Active Ollama template, local model path or generation route differs from audit")
        self.assert_stable()
        return {"name": MODEL, "digest": identity["model_manifest_sha256"], "provider_version": version,
                "synthetic": False, "endpoint": self.endpoint, "prompt_settlement_identity": identity,
                "listener": self.process}

    def before_fresh_request(self):
        current = self.verify_endpoint()
        if self.a100_gpus:
            current["a100_inventory"] = self.visible_a100()
        require(current == self.metadata, "Serving identity changed after preflight")

    def execute(self, payload, ids, action):
        return self.http_json("/api/chat", payload, before_send=self.assert_stable)

    def verify_generation_device(self):
        if not self.a100_gpus:
            return None
        models = self.get("/api/ps")["models"]
        matches = [row for row in models if row.get("digest") == self.metadata["digest"]]
        require(len(matches) == 1 and type(matches[0].get("size")) is int
                and matches[0]["size"] > 0 and matches[0].get("size_vram") == matches[0]["size"]
                and matches[0].get("context_length") == self.opts["context_window"],
                "Expected complete GPU placement and the frozen context from Ollama /api/ps")
        return {"model_size": matches[0]["size"], "size_vram": matches[0]["size_vram"],
                "context_length": matches[0]["context_length"], "listener": self.process,
                "compute_processes": runner_gpu(self.process["pid"], self.binary, self.a100_gpus, self.model_paths[0])}

    def response_identity_matches(self, response):
        return response.get("model") == MODEL

    def claim_collection(self, cohort_sha256, directory, *, resume):
        # Store the claim on the serving host so a copied lock cannot silently
        # start another paid run.
        root = self.models_dir.parent / "graft-fl-live-claims"
        root.mkdir(mode=0o700, exist_ok=True)
        claim = root / f"{self.certificate.identity['certificate_sha256']}.json"
        identity = {"cohort_sha256": cohort_sha256, "output": str(directory.resolve())}
        if resume:
            require(claim.is_file() and json.loads(claim.read_text()) == identity,
                    "Live cohort claim is missing or belongs to a different run directory")
            return
        try:
            with claim.open("x") as stream:
                json.dump(identity, stream)
                stream.flush()
                os.fsync(stream.fileno())
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except FileExistsError as exc:
            raise ValueError("Live cohort already claimed. Resume its original journal.") from exc
