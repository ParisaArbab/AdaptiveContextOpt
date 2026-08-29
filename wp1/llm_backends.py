"""Small provider layer for Agent4SR and Agent4LR.

Default use is Ollama because the target WP1 server runs local Llama, Qwen and
Mistral models. OpenAI and Anthropic are optional alternatives.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request


class ChatBackend:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 180,
    ):
        self.provider = provider.lower()
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

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
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        return _strip_reasoning(str(data.get("message", {}).get("content", "")))

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
