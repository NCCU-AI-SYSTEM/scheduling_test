"""
P3 Parser：LLM few-shot。

給 5 個範例後讓 LLM 解析，幫助理解否定語義和多維度條件。
使用 Trend Micro endpoint（優先）或 OPENAI_API_KEY。
"""

from __future__ import annotations

import logging
import os

from src.parsers.base import ConditionResult
from src.parsers.llm_client import make_client, extract_json
from src.parsers.llm_zeroshot import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# few-shot 範例（user/assistant 對話）
FEW_SHOT: list[dict] = [
    {
        "role": "user",
        "content": "星期三下午英文授課的選修課",
    },
    {
        "role": "assistant",
        "content": '{"must":{"course_lang":["英文"],"weekday":[3],"hour_range":[13,18],"point":null,"kind":["選修"],"lmt_kind":[],"unit":[]},"must_not":{"course_lang":[],"weekday":[],"hour_range":null,"point":null,"kind":[],"lmt_kind":[],"unit":[]}}',
    },
    {
        "role": "user",
        "content": "不要早上的課，找社會通識，3學分",
    },
    {
        "role": "assistant",
        "content": '{"must":{"course_lang":[],"weekday":[],"hour_range":null,"point":3.0,"kind":[],"lmt_kind":["社會通識"],"unit":[]},"must_not":{"course_lang":[],"weekday":[],"hour_range":[8,12],"point":null,"kind":[],"lmt_kind":[],"unit":[]}}',
    },
    {
        "role": "user",
        "content": "企管系中文授課三學分課",
    },
    {
        "role": "assistant",
        "content": '{"must":{"course_lang":["中文"],"weekday":[],"hour_range":null,"point":3.0,"kind":[],"lmt_kind":[],"unit":["企管系"]},"must_not":{"course_lang":[],"weekday":[],"hour_range":null,"point":null,"kind":[],"lmt_kind":[],"unit":[]}}',
    },
    {
        "role": "user",
        "content": "週一到週四都有課，不要必修",
    },
    {
        "role": "assistant",
        "content": '{"must":{"course_lang":[],"weekday":[1,2,3,4],"hour_range":null,"point":null,"kind":[],"lmt_kind":[],"unit":[]},"must_not":{"course_lang":[],"weekday":[],"hour_range":null,"point":null,"kind":["必修"],"lmt_kind":[],"unit":[]}}',
    },
    {
        "role": "user",
        "content": "我想選中文授課、星期二上課的課，不要下午時段",
    },
    {
        "role": "assistant",
        "content": '{"must":{"course_lang":["中文"],"weekday":[2],"hour_range":null,"point":null,"kind":[],"lmt_kind":[],"unit":[]},"must_not":{"course_lang":[],"weekday":[],"hour_range":[13,18],"point":null,"kind":[],"lmt_kind":[],"unit":[]}}',
    },
]


def parse(query: str, model: str | None = None) -> ConditionResult:
    """
    P3：LLM few-shot 解析。

    參數：
        query: 使用者的自然語言查詢
        model: 指定 LLM model（None 則用預設）
    回傳：
        ConditionResult
    """
    client, default_model = make_client()
    use_model = model or os.environ.get("PARSER_LLM_MODEL", default_model)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT)
    messages.append({"role": "user", "content": query})

    try:
        r = client.chat.completions.create(
            model=use_model,
            messages=messages,
            max_tokens=400,
        )
        raw = r.choices[0].message.content or ""
        obj = extract_json(raw)
        if not obj:
            logger.warning(f"llm_fewshot 無法解析 JSON: {raw[:100]}")
            return ConditionResult()
        return ConditionResult(
            must=obj.get("must", {}),
            must_not=obj.get("must_not", {}),
        )
    except Exception as e:
        logger.error(f"llm_fewshot 呼叫失敗：{e}")
        return ConditionResult()
