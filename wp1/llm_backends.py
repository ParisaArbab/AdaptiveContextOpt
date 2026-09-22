"""Small provider layer for Agent4SR and Agent4LR.

Default use is Ollama because the target WP1 server runs local Llama, Qwen and
Mistral models. OpenAI and Anthropic are optional alternatives.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import time


class ChatBackend:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 180,
        ollama_num_ctx: int | None = None,
        ollama_num_predict: int | None = None,
        ollama_max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.ollama_num_ctx = (
            int(os.getenv("OLLAMA_NUM_CTX", "16384"))
            if ollama_num_ctx is None
            else int(ollama_num_ctx)
        )
        self.ollama_num_predict = (
            int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
            if ollama_num_predict is None
            else int(ollama_num_predict)
        )
        self.ollama_max_retries = max(1, int(ollama_max_retries))

    def complete(self, system: str, user: str) -> str:
        if self.provider == "ollama":
            return self._ollama(system, user)
        if self.provider in {"openai", "openai-compatible", "vllm"}:
            return self._openai(system, user)
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _ollama(self, system: str, user: str) -> str:
        base = (self.base_url or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": self.ollama_num_predict,
                    "num_ctx": self.ollama_num_ctx,
                },
            }
        ).encode()

        last_error: Exception | None = None
        for attempt in range(1, self.ollama_max_retries + 1):
            req = urllib.request.Request(
                f"{base}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                return _strip_reasoning(str(data.get("message", {}).get("content", "")))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in {500, 502, 503, 504} or attempt == self.ollama_max_retries:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == self.ollama_max_retries:
                    raise

            wait_seconds = min(15, 2 ** (attempt - 1))
            print(
                f"[Ollama] transient failure on attempt {attempt}/"
                f"{self.ollama_max_retries}; retrying in {wait_seconds}s...",
                flush=True,
            )
            time.sleep(wait_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama request failed without an explicit error")

    def _openai(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package for this backend") from exc
        key = self.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _strip_reasoning(response.choices[0].message.content or "")

    def _anthropic(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the anthropic package for this backend") from exc
        client = anthropic.Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=self.model,
            max_tokens=1800,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "\n".join(getattr(block, "text", "") for block in response.content)
        return _strip_reasoning(text)


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()
