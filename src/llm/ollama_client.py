"""Ollama client wrapper.

By default talks to home_mac via SSH tunnel on localhost:11434.
Fail loud if tunnel is down — do not silent-fallback.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import ollama

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


def get_client(host: str = DEFAULT_HOST) -> ollama.Client:
    return ollama.Client(host=host, timeout=120)


def health_check(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
    """Raise if Ollama unreachable or model missing."""
    client = get_client(host)
    tags = client.list()
    names = {m.model for m in tags.models}
    if model not in names:
        raise RuntimeError(f"Model {model!r} not in Ollama at {host}. Available: {sorted(names)}")


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.3,
    json_mode: bool = False,
    num_ctx: int = 8192,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> LLMResponse:
    options: dict = {"temperature": temperature, "num_ctx": num_ctx}
    fmt = "json" if json_mode else None
    last_err: Exception | None = None
    for attempt in range(max_retries):
        client = get_client(host)
        try:
            resp = client.chat(model=model, messages=messages, options=options, format=fmt)
            return LLMResponse(
                text=resp.message.content or "",
                model=resp.model,
                prompt_tokens=resp.prompt_eval_count or 0,
                completion_tokens=resp.eval_count or 0,
            )
        except Exception as e:  # noqa: BLE001 — connection / timeout / 5xx
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
    assert last_err is not None
    raise last_err


def chat_json(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    temperature: float = 0.3,
    num_ctx: int = 8192,
    max_retries: int = 2,
) -> dict:
    """Return parsed JSON dict; retry once if model emits malformed JSON."""
    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        resp = chat(messages, model, host, temperature, json_mode=True, num_ctx=num_ctx)
        try:
            return json.loads(resp.text)
        except json.JSONDecodeError as e:
            last_err = e
    assert last_err is not None
    raise RuntimeError(f"LLM returned invalid JSON after {max_retries+1} attempts: {last_err}")
