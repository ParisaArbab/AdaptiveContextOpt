"""Read-only platform/resource preflight. Never calls a model or tests a checkout."""
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess

from rave_fl.eval.adapters import load_instances, capabilities
from rave_fl.eval.backend import Provider


def inspect(opts):
    target = Path(opts.get("output_dir") or "results/rave_fl_eval").resolve()
    parent = target
    while not parent.exists():
        parent = parent.parent
    disk = shutil.disk_usage(parent)
    checks = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "available_disk_bytes": disk.free,
        "output_parent_writable": os.access(parent, os.W_OK),
        "ollama_cli": shutil.which("ollama"),
        "git": shutil.which("git"),
        "defects4j": shutil.which("defects4j"),
        "device_requested": opts["device"],
        "inference_performed": False,
        "external_tests_performed": False,
    }
    try:
        checks["physical_ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError):
        checks["physical_ram_bytes"] = None
    checks["dependencies"] = {
        name: bool(importlib.util.find_spec(name))
        for name in ["numpy", "matplotlib", "tree_sitter", "tree_sitter_java", "yaml"]
    }
    checks["java"] = None
    if shutil.which("java"):
        p = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
        checks["java"] = p.stderr.strip()
    if shutil.which("nvidia-smi"):
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        checks["nvidia_gpus"] = (
            p.stdout.strip() if p.returncode == 0 else {"error": p.stderr.strip()}
        )
    else:
        checks["nvidia_gpus"] = None
    checks["hardware_status"] = "device managed by provider; no performance measurement"
    if opts["device"] in {"a100", "cuda"} and not checks["nvidia_gpus"]:
        checks[
            "hardware_status"
        ] = "NVIDIA hardware unavailable on this host; remote server must be checked separately"
    try:
        instances, source = load_instances(opts)
        checks["selection"] = [i.instance_id for i in instances]
        count = len(instances)
        checks["missing_checkouts"] = [
            i.instance_id for i in instances if not Path(i.checkout).is_dir()
        ]
        checks["evidence_kinds"] = sorted({i.evidence.kind for i in instances})
        checks["dataset_sha256"] = source["dataset_sha256"]
    except Exception as exc:
        checks["dataset_error"] = str(exc)
        count = len(opts.get("bugs", [])) or opts.get("limit")
    if count is not None:
        pool = opts["candidate_count"]
        batches = (pool + opts["batch_size"] - 1) // opts["batch_size"]
        checks["resource_estimate"] = {
            "assumed_instances": count,
            "candidate_pool_limit": pool,
            "shared_Full_plus_L1_reference_calls_upper": count * (pool + batches),
            "all_policy_schedule_calls_upper": count
            * len(opts["policies"])
            * len(opts["fusions"])
            * opts["max_steps"],
            "configured_new_call_cap": opts["max_new_calls"],
            "configured_physical_token_cap": opts["physical_budget"],
            "per_bug_physical_cap": opts["per_bug_physical_budget"],
            "maximum_request_reservation": opts["context_window"]
            + opts["output_tokens"] * opts["batch_size"],
            "checkout_storage": "Unavailable until repository sizes are known; no cloning/dataset preparation in preflight",
            "interpretation": "Planning upper bounds, not measured/expected provider cost; incomplete collection is possible under caps",
        }
    if opts.get("check_provider"):
        try:
            checks["provider"] = Provider(opts).preflight()
        except Exception as exc:
            checks["provider_error"] = str(exc)
    else:
        checks[
            "provider"
        ] = "Not contacted; use --check-provider for GET-only installation/version checks"
    checks["capabilities"] = capabilities()[opts["dataset"]]
    return checks
