"""LLM-based query rewriters.

Primary flow: OpenAI gpt-4o-mini batch (build_rewrite_batch.py → run_rewrite_batch.py --merge)
writes disk cache. Falls back to Ollama gemma4:e4b for cache misses.

Methods:
  - HyDE      : hypothetical course description matching the query
  - Q2D       : Query2Doc — expanded passage containing the query
  - Multi     : N paraphrases for RAG-Fusion
  - StepBack  : abstract the query to a higher-level concept

All return strings (or list[str] for Multi). Caller embeds/searches them.
Cached on disk under data/processed/query_cache/<sha256>.json so re-runs are free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.llm.ollama_client import DEFAULT_MODEL, chat, chat_json

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "query_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache(name: str, query: str, model: str) -> Path:
    h = hashlib.sha256(f"{name}|{model}|{query}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{name}_{h}.json"


# ---------- HyDE -------------------------------------------------------------

HYDE_SYSTEM = (
    "你是政治大學課程資訊助教。"
    "依使用者的查詢，撰寫一段 80 至 150 字的繁體中文課程簡介，"
    "假裝這就是符合查詢的課程，"
    "內容要含主題、學科術語、可能的週次主題與學習目標。"
    "只輸出該段文字，不加標題、不加引號。"
)


def hyde(query: str, model: str = DEFAULT_MODEL, use_cache: bool = True) -> str:
    cp = _cache("hyde", query, model)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["text"]
    resp = chat(
        [
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user", "content": f"查詢：{query}"},
        ],
        model=model,
        temperature=0.4,
        num_ctx=2048,
    )
    out = resp.text.strip()
    if use_cache:
        cp.write_text(json.dumps({"text": out}, ensure_ascii=False))
    return out


# ---------- Query2Doc --------------------------------------------------------

Q2D_SYSTEM = (
    "你是政治大學課程資訊助教。"
    "依使用者的查詢，產生一段 100 字左右的擴展段落，"
    "段落必須包含原查詢字串，加入 3-5 個相關學科術語、口語別稱、可能的課程內容。"
    "只輸出段落本文。"
)


def q2d(query: str, model: str = DEFAULT_MODEL, use_cache: bool = True) -> str:
    cp = _cache("q2d", query, model)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["text"]
    resp = chat(
        [
            {"role": "system", "content": Q2D_SYSTEM},
            {"role": "user", "content": f"查詢：{query}"},
        ],
        model=model,
        temperature=0.4,
        num_ctx=2048,
    )
    out = resp.text.strip()
    if not out.startswith(query) and query not in out:
        # ensure query token presence — Q2D paper does this
        out = f"{query}。{out}"
    if use_cache:
        cp.write_text(json.dumps({"text": out}, ensure_ascii=False))
    return out


# ---------- Multi-query ------------------------------------------------------

MULTI_SYSTEM = (
    "你是政治大學課程搜尋助手。"
    "依使用者的查詢，產出 4 個語意相同但措辭不同的繁體中文改寫。"
    "改寫要涵蓋：學術術語版、口語版、相關別稱版、加入學科背景版。"
    "輸出嚴格 JSON：{\"queries\": [\"...\", \"...\", \"...\", \"...\"]}"
)


def multi_query(
    query: str, model: str = DEFAULT_MODEL, use_cache: bool = True
) -> list[str]:
    cp = _cache("multi", query, model)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["queries"]
    obj = chat_json(
        [
            {"role": "system", "content": MULTI_SYSTEM},
            {"role": "user", "content": f"查詢：{query}"},
        ],
        model=model,
        temperature=0.6,
        num_ctx=2048,
    )
    qs = [q.strip() for q in (obj.get("queries") or []) if q and q.strip()]
    qs = [q for q in qs if q != query][:4]
    if use_cache:
        cp.write_text(json.dumps({"queries": qs}, ensure_ascii=False))
    return qs


# ---------- Step-back --------------------------------------------------------

STEPBACK_SYSTEM = (
    "你是政治大學課程搜尋助手。"
    "把使用者的具體查詢抽象成更上層的學科或領域概念，"
    "回傳一句不超過 20 字的繁體中文。"
    "例：『想學 LSTM 神經網路』 -> 『深度學習』。"
    "只輸出抽象概念字串。"
)


def step_back(
    query: str, model: str = DEFAULT_MODEL, use_cache: bool = True
) -> str:
    cp = _cache("stepback", query, model)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())["text"]
    resp = chat(
        [
            {"role": "system", "content": STEPBACK_SYSTEM},
            {"role": "user", "content": f"查詢：{query}"},
        ],
        model=model,
        temperature=0.2,
        num_ctx=1024,
    )
    out = resp.text.strip().strip("「」\"'")
    if use_cache:
        cp.write_text(json.dumps({"text": out}, ensure_ascii=False))
    return out


__all__ = ["hyde", "q2d", "multi_query", "step_back"]
