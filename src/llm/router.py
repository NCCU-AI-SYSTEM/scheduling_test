"""Which client the planner calls: an OpenAI-compatible endpoint when V4_OPENAI_BASE_URL is
set (the backend's setup), otherwise native Ollama (experiments, and CPU thread options)."""

from __future__ import annotations

import os

from src.llm import ollama_http, openai_http
from src.llm.ollama_http import ChatResult


def chat(model: str, messages: list[dict], **kw) -> ChatResult:
    client = openai_http if os.environ.get("V4_OPENAI_BASE_URL") else ollama_http
    return client.chat(model, messages, **kw)
