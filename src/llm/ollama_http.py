"""Minimal Ollama /api/chat client with timing, for benchmarking local and cloud models.

Kept dependency-free (urllib) so the same code runs on hopelab and a laptop, and
so latency numbers include nothing but the HTTP round trip.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict]
    seconds: float
    prompt_tokens: int
    output_tokens: int
    eval_seconds: float  # pure generation time reported by Ollama
    error: str = ""

    @property
    def tok_per_s(self) -> float:
        return self.output_tokens / self.eval_seconds if self.eval_seconds else 0.0


def parse_json_object(text: str) -> dict | None:
    """First JSON object in a reply. Cloud models on Ollama sometimes ignore `format`
    and wrap the object in a ```json fence, so plain json.loads is not enough."""
    import re

    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def chat(model: str, messages: list[dict], *, fmt: dict | str | None = None,
         tools: list[dict] | None = None, think: bool = False, temperature: float = 0.0,
         num_ctx: int = 8192, timeout: int = 600, retries: int = 4,
         options: dict | None = None) -> ChatResult:
    """options: extra Ollama runtime options, e.g. {"num_gpu": 0, "num_thread": 4} for CPU-only."""
    body: dict = {"model": model, "messages": messages, "stream": False, "think": think,
                  "keep_alive": "30m",
                  "options": {"temperature": temperature, "num_ctx": num_ctx, **(options or {})}}
    if fmt is not None:
        body["format"] = fmt
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{HOST}/api/chat", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            msg = d.get("message", {})
            return ChatResult(msg.get("content", ""), msg.get("tool_calls") or [], time.time() - t0,
                              d.get("prompt_eval_count", 0), d.get("eval_count", 0),
                              d.get("eval_duration", 0) / 1e9)
        except urllib.error.HTTPError as e:
            # cloud models rate-limit (429); back off and retry
            if e.code == 429 and attempt < retries:
                time.sleep(5 * 2 ** attempt)
                continue
            return ChatResult("", [], time.time() - t0, 0, 0, 0, error=f"HTTP {e.code}")
        except Exception as e:  # noqa: BLE001 - benchmark must record, not crash
            return ChatResult("", [], time.time() - t0, 0, 0, 0, error=repr(e)[:200])
    return ChatResult("", [], 0, 0, 0, 0, error="retries exhausted")
