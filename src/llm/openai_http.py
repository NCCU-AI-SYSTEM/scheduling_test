"""Minimal OpenAI-compatible /chat/completions client, same result type as ollama_http.chat.

The backend (CourseLangChain) reaches every model through an OpenAI-compatible endpoint
(LiteLLM, Gemini, Ollama's /v1), so the planner must work over one too. Settings describe
this machine, not the data, so they come from the environment:

  V4_OPENAI_BASE_URL  e.g. http://localhost:11434/v1 (no default: see src/llm/router.py)
  V4_OPENAI_API_KEY   sent as a bearer token when set
  V4_OPENAI_EXTRA     JSON merged into the request body, for endpoint-specific fields such as
                      {"reasoning_effort": "none"} on Ollama

Ollama runtime options (num_gpu, num_thread, num_ctx) have no equivalent here: on these
endpoints they are the server's configuration, so they are dropped.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from src.llm.ollama_http import ChatResult


def chat(model: str, messages: list[dict], *, fmt: dict | str | None = None, temperature: float = 0.0,
         timeout: int = 600, retries: int = 4, options: dict | None = None, **_ignored) -> ChatResult:
    base = os.environ.get("V4_OPENAI_BASE_URL", "").rstrip("/")
    if not base:
        raise ValueError("V4_OPENAI_BASE_URL is not set")
    body: dict = {"model": model, "messages": messages, "stream": False, "temperature": temperature}
    if isinstance(fmt, dict):
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": fmt}}
    elif fmt == "json":
        body["response_format"] = {"type": "json_object"}
    body |= json.loads(os.environ.get("V4_OPENAI_EXTRA") or "{}")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("V4_OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(f"{base}/chat/completions", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            usage = d.get("usage") or {}
            msg = (d.get("choices") or [{}])[0].get("message") or {}
            seconds = time.time() - t0
            return ChatResult(msg.get("content") or "", msg.get("tool_calls") or [], seconds,
                              usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), seconds)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(5 * 2 ** attempt)
                continue
            return ChatResult("", [], time.time() - t0, 0, 0, 0, error=f"HTTP {e.code}")
        except Exception as e:  # noqa: BLE001 - the planner falls back on any failure; record why
            return ChatResult("", [], time.time() - t0, 0, 0, 0, error=repr(e)[:200])
    return ChatResult("", [], 0, 0, 0, 0, error="retries exhausted")
