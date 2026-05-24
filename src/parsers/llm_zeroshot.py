"""
P2 Parser：LLM zero-shot。

使用 OpenAI API，不給範例（zero-shot），
讓 LLM 直接從查詢字串解析出結構化的課程搜尋條件。
"""

from __future__ import annotations

import json
import logging
import os

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
    使用 LLM zero-shot 解析查詢字串。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        ConditionResult：結構化的搜尋條件；出錯時回傳空的 ConditionResult
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        result = _parse_response(content)
        logger.debug(f"llm_zeroshot 解析結果：{result.to_dict()}")
        return result

    except Exception as e:
        logger.error(f"llm_zeroshot 呼叫失敗：{e}")
        return ConditionResult()
