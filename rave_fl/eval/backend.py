"""Exact selected-action cache, durable request journal and bounded providers."""
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import tempfile
import urllib.request

from rave_fl.core import Action, approx_tokens
from rave_fl.pilot_backend import build_prompt, parse_response
from rave_fl.pilot_types import CallResult, Observation
from rave_fl.eval.schema import CompatibilityError, fingerprint


class ExecutionLimit(RuntimeError):
    def __init__(self, message, paid_response=None):
        super().__init__(message)
        self.paid_response = paid_response


class UnknownUsage(RuntimeError):
    pass


def prompt(candidates, context, action, level, language, granularity):
    system, user = build_prompt(candidates, context, action, level)
    if language != "java":
        system = system.replace("Java", language.capitalize())
    if granularity == "file":
        system += " Inspect the supplied file as the candidate entity; do not substitute a function-level verdict."
    return system, user


class Provider:
    def __init__(self, opts):
        self.opts = opts
        self.endpoint = opts["endpoint"].rstrip("/")
        self.model = opts["model"]
        self.kind = opts["backend"]
        self.metadata = {}
        self.function = None
        if self.kind == "plugin":
            if not opts.get("backend_factory"):
                raise CompatibilityError(
                    "Plugin backend requires --backend-factory module:function"
                )
            module, name = opts["backend_factory"].split(":", 1)
            self.function = getattr(importlib.import_module(module), name)

    def get(self, path):
        with urllib.request.urlopen(self.endpoint + path, timeout=10) as response:
            return json.load(response)

    def preflight(self):
        if self.kind == "fake":
            self.metadata = {
                "name": "synthetic-fixture",
                "digest": "synthetic-v1",
                "provider_version": "fake-v1",
                "synthetic": True,
            }
        elif self.kind == "plugin":
            if not self.opts.get("model_digest"):
                raise CompatibilityError("Plugin backend requires an immutable --model-digest")
            self.metadata = {
                "name": self.model,
                "digest": self.opts["model_digest"],
                "provider_version": "plugin:" + self.opts["backend_factory"],
                "synthetic": False,
            }
        else:
            tags = self.get("/api/tags")
            found = next((r for r in tags["models"] if r["name"] == self.model), None)
            if found is None:
                raise CompatibilityError(
                    "Model not installed: "
                    + self.model
                    + "; no weights are downloaded automatically"
                )
            version = self.get("/api/version")
            self.metadata = {
                **found,
                "provider_version": version.get("version", "unavailable"),
                "synthetic": False,
            }
            if self.opts.get("model_digest") and found["digest"] != self.opts["model_digest"]:
                raise CompatibilityError("Configured model digest does not match provider")
        return self.metadata

    def execute(self, payload, ids, action):
        if self.kind == "fake":
            rows = []
            for n, ident in enumerate(ids):
                score = (
                    int(
                        hashlib.sha256((ident + str(payload["messages"])).encode()).hexdigest()[:8],
                        16,
                    )
                    % 9
                    + 1
                ) / 10
                rows.append(
                    {
                        "id": f"c{n}",
                        "suspicion": score,
                        "uncertainty": 0.3,
                        "explanation": "Synthetic fixture response",
                        "missing_evidence": "",
                    }
                )
            content = json.dumps(
                {"suspicion": rows[0]["suspicion"], "explanation": "Synthetic fixture response"}
                if action == Action.FULL_VERIFY
                else {"results": rows}
            )
            return {
                "done": True,
                "message": {"content": content},
                "prompt_eval_count": max(
                    1, approx_tokens("".join(m["content"] for m in payload["messages"]))
                ),
                "eval_count": min(
                    payload["options"]["num_predict"], max(1, approx_tokens(content))
                ),
                "synthetic": True,
            }
        if self.kind == "plugin":
            return self.function(payload)
        request = urllib.request.Request(
            self.endpoint + "/api/chat",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.opts["timeout"]) as response:
            return json.load(response)


class Store:
    """A run lock excludes other processes; this lock coordinates worker threads."""

    def __init__(self, run_dir, opts):
        self.root = Path(run_dir)
        self.opts = opts
        self.lock = threading.Condition(threading.RLock())
        self.db = sqlite3.connect(self.root / "journal.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, instance TEXT, status TEXT, request TEXT, response TEXT, reservation INTEGER, input INTEGER, output INTEGER, error TEXT)"
        )
        self.db.commit()
        self.cache_dir = (
            Path(opts["cache_dir"]).resolve() if opts.get("cache_dir") else self.root / "cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.root / "raw_ledgers/requests.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)

    def event(self, value):
        with self.ledger.open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def totals(self, instance=None):
        query = "SELECT status,reservation,input,output FROM calls"
        params = ()
        if instance is not None:
            query += " WHERE instance=?"
            params = (instance,)
        rows = self.db.execute(query, params).fetchall()
        known = sum((r[2] or 0) + (r[3] or 0) for r in rows)
        reserved = sum(r[1] for r in rows if r[0] in {"pending", "unknown"})
        return {
            "known_provider_tokens": known,
            "reserved_or_unknown_tokens": reserved,
            "new_calls": sum(r[0] != "reused" for r in rows),
            "usage_unknown": any(r[0] in {"unknown", "violation"} for r in rows),
            "pending": any(r[0] == "pending" for r in rows),
        }

    def snapshot(self):
        with self.lock:
            totals = self.totals()
            synthetic = self.opts["backend"] == "fake"
            return {
                **totals,
                "physical_inference_tokens": 0 if synthetic else totals["known_provider_tokens"],
                "simulated_provider_tokens": totals["known_provider_tokens"] if synthetic else 0,
                "physical_token_definition": "Known provider input+output counters; unknown reservations separately retained",
                "synthetic": synthetic,
            }

    def request(self, key, instance, provider, payload, ids, action, readonly=False):
        record = {
            "fingerprint": key,
            "request": payload,
            "model_digest": provider.metadata["digest"],
        }
        cache = self.cache_dir / (key + ".json")
        with self.lock:
            while True:
                row = self.db.execute(
                    "SELECT status,response,error FROM calls WHERE key=?", (key,)
                ).fetchone()
                if row and row[0] == "pending":
                    # A pending record predating this process is uncertain, never
                    # silently retried. Live pending keys are tracked in memory.
                    if key not in getattr(self, "active", set()):
                        raise UnknownUsage("Interrupted request has no usage response: " + key)
                    self.lock.wait()
                    continue
                break
            if row:
                if row[0] not in {"complete", "reused"}:
                    raise UnknownUsage(row[2] or "Previous provider usage unresolved")
                self.event(
                    {"event": "selected_journal_reuse", "fingerprint": key, "instance": instance}
                )
                return json.loads(row[1]), "journal"
            if self.opts["cache_mode"] != "off" and cache.exists():
                data = json.loads(cache.read_text())
                if (
                    data["identity"] != record
                    or fingerprint(data["response"]) != data["response_sha256"]
                ):
                    raise CompatibilityError("Cache identity or response checksum mismatch")
                self.db.execute(
                    "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        key,
                        instance,
                        "reused",
                        json.dumps(payload),
                        json.dumps(data["response"]),
                        0,
                        0,
                        0,
                        None,
                    ),
                )
                self.db.commit()
                self.event(
                    {"event": "selected_cache_reuse", "fingerprint": key, "instance": instance}
                )
                return data["response"], "shared_cache"
            if readonly or self.opts["cache_mode"] == "readonly":
                raise ExecutionLimit("Exact cached action unavailable; no inference permitted")
            total = self.totals()
            bug = self.totals(instance)
            if total["usage_unknown"] or total["pending"] and not getattr(self, "active", set()):
                raise UnknownUsage("Unresolved run usage blocks additional requests")
            reservation = payload["options"]["num_ctx"] + payload["options"]["num_predict"]
            if (
                total["new_calls"] >= self.opts["max_new_calls"]
                or total["known_provider_tokens"]
                + total["reserved_or_unknown_tokens"]
                + reservation
                > self.opts["physical_budget"]
            ):
                raise ExecutionLimit(
                    "Whole-run physical token/call cap cannot reserve next request"
                )
            if (
                bug["known_provider_tokens"] + bug["reserved_or_unknown_tokens"] + reservation
                > self.opts["per_bug_physical_budget"]
            ):
                raise ExecutionLimit("Per-bug physical token cap cannot reserve next request")
            self.db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    instance,
                    "pending",
                    json.dumps(payload),
                    None,
                    reservation,
                    None,
                    None,
                    None,
                ),
            )
            self.db.commit()
            self.active = getattr(self, "active", set())
            self.active.add(key)
            self.event(
                {
                    "event": "request",
                    "fingerprint": key,
                    "instance": instance,
                    "reservation": reservation,
                    **record,
                }
            )
        start = time.monotonic()
        try:
            response = provider.execute(payload, ids, action)
            if response.get("done") is not True:
                raise ValueError("Provider response is incomplete")
            inp, out = response.get("prompt_eval_count"), response.get("eval_count")
            if type(inp) is not int or type(out) is not int or inp < 0 or out < 0:
                raise ValueError("Provider omitted valid token counters")
            with self.lock:
                self.db.execute(
                    "UPDATE calls SET status=?,response=?,input=?,output=? WHERE key=?",
                    ("complete", json.dumps(response), inp, out, key),
                )
                self.db.commit()
                self.event(
                    {
                        "event": "response",
                        "fingerprint": key,
                        "instance": instance,
                        "response": response,
                        "duration_seconds": time.monotonic() - start,
                    }
                )
                if inp > payload["options"]["num_ctx"] or out > payload["options"]["num_predict"]:
                    self.db.execute(
                        "UPDATE calls SET status=?,error=? WHERE key=?",
                        ("violation", "Provider violated context/output reservation", key),
                    )
                    self.db.commit()
                    raise ExecutionLimit(
                        "Provider counter bound violated; paid usage retained; stop run", response
                    )
                with tempfile.NamedTemporaryFile(
                    "w", dir=self.cache_dir, prefix=key + ".", suffix=".tmp", delete=False
                ) as handle:
                    json.dump(
                        {
                            "identity": record,
                            "response": response,
                            "response_sha256": fingerprint(response),
                        },
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                    temp = Path(handle.name)
                temp.replace(cache)
            return response, "fresh"
        except Exception as exc:
            with self.lock:
                row = self.db.execute("SELECT status FROM calls WHERE key=?", (key,)).fetchone()
                if row and row[0] == "pending":
                    self.db.execute(
                        "UPDATE calls SET status=?,error=? WHERE key=?",
                        ("unknown", f"{type(exc).__name__}: {exc}", key),
                    )
                    self.db.commit()
                    self.event(
                        {
                            "event": "error",
                            "fingerprint": key,
                            "instance": instance,
                            "usage_unknown": True,
                            "error": str(exc),
                        }
                    )
            raise
        finally:
            with self.lock:
                self.active.discard(key)
                self.lock.notify_all()


class Backend:
    def __init__(self, instance, entities, provider, store, opts, budget):
        self.instance, self.entities, self.provider, self.store, self.opts = (
            instance,
            entities,
            provider,
            store,
            opts,
        )
        self.context_window = opts["context_window"] + (
            opts["output_tokens"] * opts["batch_size"] if opts["admission"] == "strict" else 0
        )
        self.budget = budget
        self.spent = 0
        self.events = []
        self.pool_hash = fingerprint([asdict(e) for e in entities])
        self.spec = None
        if opts["admission"] == "strict" and opts.get("reservation_spec"):
            self.spec = json.loads(Path(opts["reservation_spec"]).read_text())
            if (
                self.spec["model_digest"] != provider.metadata["digest"]
                or self.spec["model"] != opts["model"]
            ):
                raise CompatibilityError(
                    "Strict model-specific reservation model/digest mismatch; omit spec for conservative context reservation"
                )
            if (
                hashlib.sha256(self.spec["template"].encode()).hexdigest()
                != self.spec["template_sha256"]
            ):
                raise CompatibilityError("Reservation template checksum mismatch")
            if self.spec["template_overhead_byte_allowance"] < 256:
                raise CompatibilityError("Insufficient model-specific template byte allowance")

    def payload(self, cs, action, level):
        system, user = prompt(
            cs,
            self.instance.evidence.text,
            action,
            level,
            self.instance.language,
            self.opts["granularity"],
        )
        options = {
            "temperature": self.opts["temperature"],
            "seed": self.opts["seed"],
            "num_ctx": self.opts["context_window"],
            "num_predict": self.opts["output_tokens"] * len(cs),
        }
        if self.opts["device"] == "cpu":
            options["num_gpu"] = 0
        return {
            "model": self.opts["model"],
            "stream": False,
            "think": False,
            "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": options,
        }

    def planning(self, cs, action, level):
        payload = self.payload(cs, action, level)
        content = "".join(m["content"] for m in payload["messages"])
        cap = payload["options"]["num_predict"]
        predicted = 2 * approx_tokens(content) + cap + 96
        if self.opts["admission"] == "historical":
            reserved = predicted
        elif self.provider.kind == "fake":
            reserved = len(content.encode()) + cap + 256
        elif self.spec:
            # This bound applies to the pinned plain two-message template. File
            # and Python text still pass as content, never tools/history.
            reserved = max(
                predicted,
                min(
                    self.opts["context_window"],
                    len(content.encode()) + self.spec["template_overhead_byte_allowance"],
                )
                + cap,
            )
        else:
            reserved = self.opts["context_window"] + cap
        return predicted, reserved

    def estimate(self, cs, action, level):
        predicted, reserved = self.planning(cs, action, level)
        # Legacy engines compare estimate to context_window. A complete-window
        # reservation includes output; expose a larger admission window only for
        # budget reservation while enforcing prompt fit separately at execute.
        return reserved

    def execute(self, cs, action, level):
        predicted, reserved = self.planning(cs, action, level)
        if reserved > self.budget - self.spent:
            raise ExecutionLimit("Per-policy reservation cannot admit selected action")
        if predicted > self.opts["context_window"]:
            raise CompatibilityError(
                "Conservative prompt/context estimate too large; raise context or reduce pool/batch"
            )
        payload = self.payload(cs, action, level)
        identity = {
            "payload": payload,
            "provider": self.provider.kind,
            "model_digest": self.provider.metadata["digest"],
            "provider_version": self.provider.metadata["provider_version"],
            "pool_hash": self.pool_hash,
            "evidence": self.instance.evidence.sha256,
            "revision": self.instance.revision,
            "candidate_ids": [c.function_id for c in cs],
            "action": action.value,
            "level": level,
            "language": self.instance.language,
            "granularity": self.opts["granularity"],
            "representation": self.opts["representation"],
            "prompt_version": "rave-universal-v1 (Java method content preserves legacy prompt)",
        }
        key = fingerprint(identity)
        start = time.monotonic()
        try:
            response, reuse = self.store.request(
                key,
                self.instance.instance_id,
                self.provider,
                payload,
                identity["candidate_ids"],
                action,
                readonly=self.opts["cache_mode"] == "readonly",
            )
        except ExecutionLimit as exc:
            if exc.paid_response:
                inp = exc.paid_response["prompt_eval_count"]
                out = exc.paid_response["eval_count"]
                self.spent += inp + out
                self.events.append(
                    {
                        "fingerprint": key,
                        "predicted_tokens": predicted,
                        "reserved_tokens": reserved,
                        "actual_tokens": inp + out,
                        "input_tokens": inp,
                        "output_tokens": out,
                        "reuse": "fresh_bound_violation",
                        "remaining": self.budget - self.spent,
                        "error": str(exc),
                    }
                )
            raise
        raw = response.get("message", {}).get("content", "")
        obs, error = parse_response(raw, cs, action)
        inp, out = response["prompt_eval_count"], response["eval_count"]
        self.spent += inp + out
        self.events.append(
            {
                "fingerprint": key,
                "predicted_tokens": predicted,
                "reserved_tokens": reserved,
                "actual_tokens": inp + out,
                "input_tokens": inp,
                "output_tokens": out,
                "reuse": reuse,
                "remaining": self.budget - self.spent,
                "error": error,
            }
        )
        if self.opts["admission"] == "strict" and inp + out > reserved:
            with self.store.lock:
                self.store.db.execute(
                    "UPDATE calls SET status=?,error=? WHERE key=?",
                    ("violation", "Strict reservation assumption violated", key),
                )
                self.store.db.commit()
                self.store.event(
                    {
                        "event": "reservation_violation",
                        "fingerprint": key,
                        "actual_tokens": inp + out,
                        "reserved_tokens": reserved,
                    }
                )
            raise ExecutionLimit(
                "Provider usage exceeded strict reservation; retained in raw ledger"
            )
        return CallResult(
            obs, inp, out, time.monotonic() - start if reuse == "fresh" else 0.0, raw, error
        )
