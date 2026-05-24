"""
P4 Parser：LLM structured output + JSON schema。

使用 OpenAI API 的 structured output 功能（beta.chat.completions.parse()），
搭配 Pydantic model 強制確保輸出格式正確，避免 LLM 輸出格式錯誤。
同樣包含 few-shot 範例提升準確度。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.parsers.base import ConditionResult

logger = logging.getLogger(__name__)

# OpenAI 設定（從環境變數讀取）
_BASE_URL = os.environ.get("OPENAI_BASE_URL")
_API_KEY = os.environ.get("OPENAI_API_KEY")
_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")


# ── Pydantic Models（定義 JSON schema）──────────────────────────────────────

class MustConditions(BaseModel):
    """必須符合的課程搜尋條件。"""
    course_lang: list[str] = Field(default_factory=list, description="授課語言，e.g. ['英文','日文']")
    weekday: list[int] = Field(default_factory=list, description="星期幾（1=週一...7=週日），e.g. [1,2,3]")
    hour_range: Optional[list[int]] = Field(default=None, description="時間範圍 [開始小時, 結束小時]（24小時制），e.g. [13,18]")
    point: Optional[float] = Field(default=None, description="學分數，e.g. 3.0")
    kind: list[str] = Field(default_factory=list, description="課程類別（必修/選修/通識/體育）")
    lmt_kind: list[str] = Field(default_factory=list, description="通識細分，e.g. ['社會通識']")
    unit: list[str] = Field(default_factory=list, description="開課系所，e.g. ['企管系']")


class MustNotConditions(BaseModel):
    """必須排除的課程搜尋條件。"""
    course_lang: list[str] = Field(default_factory=list, description="排除的授課語言")
    weekday: list[int] = Field(default_factory=list, description="排除的星期幾")
    hour_range: Optional[list[int]] = Field(default=None, description="排除的時間範圍")
    point: Optional[float] = Field(default=None, description="排除的學分數")
    kind: list[str] = Field(default_factory=list, description="排除的課程類別")
    lmt_kind: list[str] = Field(default_factory=list, description="排除的通識細分")
    unit: list[str] = Field(default_factory=list, description="排除的開課系所")


class ShouldConditions(BaseModel):
    """選擇性偏好條件（BOOST 邏輯）。"""
    point: Optional[float] = Field(default=None, description="偏好的學分數")


class ConditionSchema(BaseModel):
    """完整的課程搜尋條件 schema。"""
    must: MustConditions = Field(default_factory=MustConditions)
    must_not: MustNotConditions = Field(default_factory=MustNotConditions)
    should: ShouldConditions = Field(default_factory=ShouldConditions)


# ── System Prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一個課程搜尋條件解析器。分析使用者的自然語言查詢，提取結構化的搜尋條件。

欄位說明：
- course_lang：授課語言，e.g. ["英文","日文"]
- weekday：星期幾，e.g. [1,2,3]（1=週一...7=週日）
- hour_range：時間範圍 [開始小時, 結束小時]（24小時制），e.g. [13,18]
- point：學分數，e.g. 3.0
- kind：課程類別，e.g. ["選修"]（必修/選修/通識/體育）
- lmt_kind：通識細分，e.g. ["社會通識"]
- unit：開課系所，e.g. ["企管系"]

注意：
- 只填有明確提到的條件，沒提到的保持空 [] 或 null
- 否定詞（不要/不含/非/排除等）開頭的條件放 must_not
- 時間範圍用 24 小時制的 [開始, 結束]
- 星期幾用數字 1-7（1=週一...7=週日）"""

# Few-shot 範例（user/assistant 對話形式）
_FEW_SHOT_EXAMPLES: list[Any] = [
    {
        "role": "user",
        "content": "星期三下午英文授課的選修課"
    },
    {
        "role": "assistant",
        "content": ConditionSchema(
            must=MustConditions(course_lang=["英文"], weekday=[3], hour_range=[13, 18], kind=["選修"]),
            must_not=MustNotConditions(),
            should=ShouldConditions(),
        ).model_dump_json(),
    },
    {
        "role": "user",
        "content": "不要早上的課，找社會通識"
    },
    {
        "role": "assistant",
        "content": ConditionSchema(
            must=MustConditions(lmt_kind=["社會通識"]),
            must_not=MustNotConditions(hour_range=[8, 12]),
            should=ShouldConditions(),
        ).model_dump_json(),
    },
    {
        "role": "user",
        "content": "企管系開的三學分中文課"
    },
    {
        "role": "assistant",
        "content": ConditionSchema(
            must=MustConditions(course_lang=["中文"], point=3.0, unit=["企管系"]),
            must_not=MustNotConditions(),
            should=ShouldConditions(),
        ).model_dump_json(),
    },
    {
        "role": "user",
        "content": "週一到週四都有的課，不要必修"
    },
    {
        "role": "assistant",
        "content": ConditionSchema(
            must=MustConditions(weekday=[1, 2, 3, 4]),
            must_not=MustNotConditions(kind=["必修"]),
            should=ShouldConditions(),
        ).model_dump_json(),
    },
    {
        "role": "user",
        "content": "英文授課但不要英文課"
    },
    {
        "role": "assistant",
        "content": ConditionSchema(
            must=MustConditions(course_lang=["英文"]),
            must_not=MustNotConditions(),
            should=ShouldConditions(),
        ).model_dump_json(),
    },
]


def _build_messages(query: str) -> list[Any]:
    """
    建立包含 system prompt 和 few-shot 範例的 messages 列表。

    參數：
        query: 使用者的查詢字串

    回傳：
        list[Any]：OpenAI messages 格式的列表
    """
    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages.extend(_FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": query})
    return messages


def _get_client():
    """
    建立 OpenAI client。

    回傳：
        OpenAI client 實例
    """
    from openai import OpenAI
    return OpenAI(base_url=_BASE_URL, api_key=_API_KEY)


def _schema_to_condition_result(schema: ConditionSchema) -> ConditionResult:
    """
    將 Pydantic ConditionSchema 轉換為 ConditionResult。

    參數：
        schema: Pydantic 解析後的結構化條件

    回傳：
        ConditionResult：課程搜尋條件
    """
    must = schema.must.model_dump()
    must_not = schema.must_not.model_dump()
    should = schema.should.model_dump()
    return ConditionResult(must=must, must_not=must_not, should=should)


def parse(query: str) -> ConditionResult:
    """
    使用 LLM structured output（Pydantic schema）解析查詢字串。
    搭配 client.beta.chat.completions.parse() 強制輸出符合 schema 的 JSON，
    避免格式錯誤。同時包含 few-shot 範例提升準確度。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        ConditionResult：結構化的搜尋條件；出錯時回傳空的 ConditionResult
    """
    try:
        client = _get_client()
        messages = _build_messages(query)
        response = client.beta.chat.completions.parse(
            model=_MODEL,
            messages=messages,
            response_format=ConditionSchema,
            temperature=0,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.error("llm_structured 回傳 None，使用空 ConditionResult")
            return ConditionResult()

        result = _schema_to_condition_result(parsed)
        logger.debug(f"llm_structured 解析結果：{result.to_dict()}")
        return result

    except Exception as e:
        logger.error(f"llm_structured 呼叫失敗：{e}")
        return ConditionResult()
