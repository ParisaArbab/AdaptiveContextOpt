"""Offline serving certificate for the fixed text-only Ollama smoke condition."""
import hashlib
import json
from pathlib import Path

from rave_fl.eval.backend import PromptSettlement
from rave_fl.eval.schema import CompatibilityError, file_hash, fingerprint


MODEL = "llama3.1:8b"
MODEL_DIGEST = "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"
TEMPLATE = "948af2743fc78a328dcb3b0f5a31b3d75f415840fdb699e8b1235978392ecf85"
VERSION = "0.17.7"
RUNNER = "llama_compatibility"
BUGS = {"Lang-54": 15, "Math-80": 21}
COHORTS = (("Lang-54",), tuple(BUGS))
SCHEMA = "graft-prompt-settlement-v1"
CAPS = {"budget": 36000, "per_bug_physical_budget": 72000, "physical_budget": 216000,
        "max_new_calls": 144}
ARM_CAPS = {"per_arm_physical_budget": 36000, "per_arm_max_new_calls": 24, "per_bug_max_new_calls": 48}


def require(condition, message):
    if not condition:
        raise CompatibilityError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class SettlementCertificate:
    def __init__(self, config):
        spec = config["prompt_settlement"]
        path = Path(spec["file"])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "Settlement certificate checksum mismatch")
        value = json.loads(raw)
        serving = value["serving"]
        opts = config["benchmark"]
        require(value["schema"] == SCHEMA and opts["backend"] == "ollama"
                and opts["model"] == MODEL and opts["model_digest"] == MODEL_DIGEST
                and serving["model_manifest_sha256"] == MODEL_DIGEST
                and serving["template_sha256"] == TEMPLATE
                and serving["serving_version"] == VERSION and serving["runner"] == RUNNER
                and digest(serving["binary_sha256"]) and digest(serving["model_blob_sha256"])
                and digest(serving["native_audit_sha256"])
                and digest(serving["native_tokenizer_log_sha256"])
                and serving["system"] in {"Darwin", "Linux"} and serving["architecture"] in {"arm64", "aarch64", "x86_64"}
                and config["graft"]["input_budget"] == 1024
                and opts["context_window"] == 8192 and opts["output_tokens"] == 160
                and opts["temperature"] == 0 and opts["seed"] == 0 and opts["device"] != "cpu"
                and opts["dataset"] == "defects4j" and opts["candidate_provider"] == "flexfl"
                and opts["candidate_count"] == 24 and opts["top_per_method"] == 10,
                "Settlement certificate differs from the frozen Ollama condition")
        tokenizer = Path(config["tokenizer_dir"] or "")
        files = value["tokenizer_files"]
        require(config["tokenizer_dir"] and set(files) == {"tokenizer_config.json", "tokenizer.json"}
                and all(digest(h) and file_hash(tokenizer / n) == h for n, h in files.items()),
                "Settlement tokenizer audit checksum mismatch")
        require(file_hash(opts["dataset_file"]) == value["dataset_sha256"], "Settlement dataset bytes differ from audit")
        self.pairs = value["pairs"]
        self.cohort = tuple(self.pairs)
        require(self.cohort in COHORTS, "Settlement certificate cohort differs from audit")
        self.identity = {**serving, "certificate_sha256": spec["sha256"], "tokenizer_files": files}
        rows = value["requests"]
        self.calls = 2 * sum(BUGS[bug] for bug in self.cohort)
        require(len(rows) == self.calls, "Settlement certificate must cover the entire audited schedule")
        self.by_action = {}
        for row in rows:
            key = (row["instance_id"], row["arm"], row["target"])
            require(key[0] in self.pairs and key[1] in {"dense_sparse_no_propagation", "dense_sparse_with_propagation"}
                    and isinstance(key[2], str) and key[2] and key not in self.by_action
                    and digest(row["request_sha256"]) and digest(row["token_ids_sha256"])
                    and type(row["prompt_tokens"]) is int and 0 < row["prompt_tokens"] <= 1024,
                    "Invalid or duplicate audited action")
            self.by_action[key] = row

    def verify_live_caps(self, config):
        require(all(config["benchmark"][k] == v for k, v in CAPS.items())
                and all(config[k] == v for k, v in ARM_CAPS.items()), "Live smoke requires the original approved caps")

    def local_model_paths(self, manifest, models_dir):
        value = json.loads(Path(manifest).read_text())

        def path(layer, expected=None):
            sha = layer["digest"].removeprefix("sha256:")
            require(digest(sha) and (expected is None or sha == expected), "Model layer differs from audit")
            blob = Path(models_dir) / "blobs" / ("sha256-" + sha)
            require(blob.is_file() and blob.stat().st_size == layer["size"], "Missing or incomplete model layer")
            return blob

        model, = [item for item in value["layers"] if item["mediaType"].endswith(".model")]
        template, = [item for item in value["layers"] if item["mediaType"].endswith(".template")]
        params, = [item for item in value["layers"] if item["mediaType"].endswith(".params")]
        return (path(model, self.identity["model_blob_sha256"]), path(template, TEMPLATE), path(value["config"]), path(params))

    def verify_local_host(self, binary, manifest, models_dir, system, architecture):
        require(binary and manifest and models_dir, "Settlement preflight requires local serving binary and model paths")
        require((system, architecture) == (self.identity["system"], self.identity["architecture"])
                and file_hash(binary) == self.identity["binary_sha256"]
                and file_hash(manifest) == self.identity["model_manifest_sha256"],
                "Serving build or host differs from the settlement audit")
        model, template, config, params = self.local_model_paths(manifest, models_dir)
        require(file_hash(template) == self.identity["template_sha256"], "Installed no-tools template differs from audit")
        require(file_hash(params) == params.name.removeprefix("sha256-"), "Installed generation defaults differ from audit")
        value = json.loads(config.read_text())
        require(file_hash(config) == config.name.removeprefix("sha256-")
                and value.get("model_family") == "llama" and not value.get("renderer")
                and not value.get("remote_host") and not value.get("remote_model"),
                "Local model configuration differs from audited compatibility path")
        with model.open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == self.identity["model_blob_sha256"],
                    "Local model blob checksum mismatch")

    def verify_pair(self, bug, prepared, graph, embeddings, schedule):
        require(self.pairs.get(bug) == {"pool_sha256": prepared["pool_hash"], "graph_sha256": graph.hash,
                "embeddings_sha256": embeddings.hash, "schedule_sha256": fingerprint(schedule)},
                "Candidate pool, graph, embeddings or schedule differs from settlement audit")

    def for_request(self, bug, arm, target, payload, counter):
        row = self.by_action.get((bug, arm, target))
        request_sha = fingerprint(payload)
        require(row and row["request_sha256"] == request_sha, "Exact planned request missing from settlement audit")
        ids = counter.token_ids(payload["messages"])
        require(len(ids) == row["prompt_tokens"] and fingerprint(ids) == row["token_ids_sha256"],
                "Exact planned prompt token IDs drifted from settlement audit")
        return PromptSettlement(request_sha, self.identity, len(ids), payload["options"]["num_predict"])
