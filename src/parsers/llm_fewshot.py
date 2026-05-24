"""
P3 Parser：LLM few-shot。

使用 OpenAI API，在 system prompt 後加入 5 個 few-shot 範例（以 user/assistant 對話形式），
讓 LLM 更準確地理解輸出格式。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.parsers.base import ConditionResult

logger = logging.getLogger(__name__)

# OpenAI 設定（從環境變數讀取）
_BASE_URL = os.environ.get("OPENAI_BASE_URL")
_API_KEY = os.environ.get("OPENAI_API_KEY")
_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

SYSTEM_PROMPT = """你是一個課程搜尋條件解析器。分析使用者的自然語言查詢，提取結構化的搜尋條件。

輸出 JSON 格式：
{
  "must": {
    "course_lang": [],
    "weekday": [],
    "hour_range": null,
    "point": null,
    "kind": [],
    "lmt_kind": [],
    "unit": []
  },
  "must_not": {
    "course_lang": [],
    "weekday": [],
    "hour_range": null,
    "point": null,
    "kind": [],
    "lmt_kind": [],
    "unit": []
  },
  "should": {
    "point": null
  }
}

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
- 星期幾用數字 1-7（1=週一...7=週日）
- 輸出必須是合法的 JSON，不要有額外的說明文字"""

# Few-shot 範例（user/assistant 對話形式）
FEW_SHOT_EXAMPLES = [
    # 範例 1：基本組合
    {
        "user": "星期三下午英文授課的選修課",
        "assistant": json.dumps({
            "must": {
                "course_lang": ["英文"],
                "weekday": [3],
                "hour_range": [13, 18],
                "point": None,
                "kind": ["選修"],
                "lmt_kind": [],
                "unit": []
            },
            "must_not": {
                "course_lang": [],
                "weekday": [],
                "hour_range": None,
                "point": None,
                "kind": [],
                "lmt_kind": [],
                "unit": []
            },
            "should": {"point": None}
        }, ensure_ascii=False),
    },
    # 範例 2：否定詞 + 通識細分
    {
        "user": "不要早上的課，找社會通識",
        "assistant": json.dumps({
            "must": {
                "course_lang": [],
                "weekday": [],
                "hour_range": None,
                "point": None,
                "kind": [],
                "lmt_kind": ["社會通識"],
                "unit": []
            },
            "must_not": {
                "course_lang": [],
                "weekday": [],
                "hour_range": [8, 12],
                "point": None,
                "kind": [],
                "lmt_kind": [],
                "unit": []
            },
            "should": {"point": None}
        }, ensure_ascii=False),
    },
    # 範例 3：系所 + 學分 + 語言
    {
        "user": "企管系開的三學分中文課",
        "assistant": json.dumps({
            "must": {
                "course_lang": ["中文"],
                "weekday": [],
                "hour_range": None,
                "point": 3.0,
                "kind": [],
                "lmt_kind": [],
                "unit": ["企管系"]
            },
            "must_not": {
                "course_lang": [],
                "weekday": [],
                "hour_range": None,
                "point": None,
                "kind": [],
                "lmt_kind": [],
                "unit": []
            },
            "should": {"point": None}
        }, ensure_ascii=False),
    },
    # 範例 4：星期範圍 + 否定課程類型
    {
        "user": "週一到週四都有的課，不要必修",
        "assistant": json.dumps({
            "must": {
                "course_lang": [],
                "weekday": [1, 2, 3, 4],
                "hour_range": None,
                "point": None,
                "kind": [],
                "lmt_kind": [],
                "unit": []
            },
            "must_not": {
                "course_lang": [],
                "weekday": [],
                "hour_range": None,
                "point": None,
                "kind": ["必修"],
                "lmt_kind": [],
                "unit": []
            },
            "should": {"point": None}
        }, ensure_ascii=False),
    },
    # 範例 5：矛盾情況處理（英文授課≠英文課）
    {
        "user": "英文授課但不要英文課",
        "assistant": json.dumps({
            "must": {
                "course_lang": ["英文"],
                "weekday": [],
                "hour_range": None,
                "point": None,
                "kind": [],
                "lmt_kind": [],
                "unit": []
            },
            "must_not": {
                "course_lang": [],
                "weekday": [],
                "hour_range": None,
                "point": None,
                "kind": [],
                "lmt_kind": [],
                "unit": []
            },
            "should": {"point": None}
        }, ensure_ascii=False),
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
    # 加入 few-shot 範例
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    # 加入實際查詢
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


def _parse_response(content: str) -> ConditionResult:
    """
    解析 LLM 回傳的 JSON 字串，轉換為 ConditionResult。

    參數：
        content: LLM 回傳的 JSON 字串

    回傳：
        ConditionResult：解析成功則回傳結構化條件，失敗則回傳空的 ConditionResult
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失敗：{e}，原始內容：{content[:200]}")
        return ConditionResult()

    must = data.get("must", {}) or {}
    must_not = data.get("must_not", {}) or {}
    should = data.get("should", {}) or {}

    return ConditionResult(must=must, must_not=must_not, should=should)


def parse(query: str) -> ConditionResult:
    """
    使用 LLM few-shot 解析查詢字串。
    在 system prompt 後插入 5 個範例對話，提升解析準確度。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        ConditionResult：結構化的搜尋條件；出錯時回傳空的 ConditionResult
    """
    try:
        client = _get_client()
        messages = _build_messages(query)
        response = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        result = _parse_response(content)
        logger.debug(f"llm_fewshot 解析結果：{result.to_dict()}")
        return result

    except Exception as e:
        logger.error(f"llm_fewshot 呼叫失敗：{e}")
        return ConditionResult()
