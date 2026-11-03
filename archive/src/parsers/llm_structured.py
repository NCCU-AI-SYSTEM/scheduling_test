"""
P4 Parser：LLM structured output（Pydantic schema 強制輸出）。

用 Pydantic 定義 schema，搭配 openai 的 structured output 模式，
確保 LLM 輸出格式完全符合 ConditionResult 的結構。
使用 Trend Micro endpoint（優先）或 OPENAI_API_KEY。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import BaseModel

from src.parsers.base import ConditionResult
from src.parsers.llm_client import make_client, extract_json
from src.parsers.llm_zeroshot import SYSTEM_PROMPT
from src.parsers.llm_fewshot import FEW_SHOT

logger = logging.getLogger(__name__)


class MustConditions(BaseModel):
    course_lang: list[str] = []
    weekday:     list[int] = []
    hour_range:  Optional[list[int]] = None
    point:       Optional[float]     = None
    kind:        list[str] = []
    lmt_kind:    list[str] = []
    unit:        list[str] = []


class ConditionSchema(BaseModel):
    must:     MustConditions
    must_not: MustConditions


def parse(query: str, model: str | None = None) -> ConditionResult:
    """
    P4：LLM structured output 解析（Pydantic schema）。

    先嘗試 client.beta.chat.completions.parse()（支援 structured output 的模型），
    若失敗則 fallback 到 few-shot + JSON 抽取。

    參數：
        query: 使用者的自然語言查詢
        model: 指定 LLM model（None 則用預設）
    回傳：
        ConditionResult
    """
    client, default_model = make_client()
    use_model = model or os.environ.get("PARSER_LLM_MODEL", default_model)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT)  # type: ignore[arg-type]
    messages.append({"role": "user", "content": query})

    # 嘗試 structured output
    try:
        r = client.beta.chat.completions.parse(
            model=use_model,
            messages=messages,  # type: ignore[arg-type]
            response_format=ConditionSchema,
            max_tokens=400,
        )
        parsed = r.choices[0].message.parsed
        if parsed:
            return ConditionResult(
                must=parsed.must.model_dump(),
                must_not=parsed.must_not.model_dump(),
            )
    except Exception as e:
        logger.debug(f"structured output 失敗，fallback JSON 抽取：{e}")

    # Fallback：直接打 API 再 regex 抽 JSON
    try:
        r2 = client.chat.completions.create(
            model=use_model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=400,
        )
        raw = r2.choices[0].message.content or ""
        obj = extract_json(raw)
        if obj:
            return ConditionResult(
                must=obj.get("must", {}),
                must_not=obj.get("must_not", {}),
            )
    except Exception as e2:
        logger.error(f"llm_structured fallback 也失敗：{e2}")

    return ConditionResult()
