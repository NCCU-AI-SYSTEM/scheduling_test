"""
P2 Parser：LLM zero-shot。

不給範例，讓 LLM 直接從查詢字串解析出結構化的課程搜尋條件。
使用 Trend Micro endpoint（優先）或 OPENAI_API_KEY。
"""

from __future__ import annotations

import logging
import os

from src.parsers.base import ConditionResult
from src.parsers.llm_client import make_client, extract_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一個課程搜尋條件解析器。分析使用者的自然語言查詢，提取結構化的搜尋條件。

輸出 JSON 格式：
{
  "must": {
    "course_lang": [],      // 必須是這些語言授課，e.g. ["英文","日文"]
    "weekday": [],          // 必須在這些星期，e.g. [1,2,3]（1=週一...7=週日）
    "hour_range": null,     // 必須在此時間範圍 [開始,結束]，e.g. [13,18]
    "point": null,          // 必須是此學分數，e.g. 3.0
    "kind": [],             // 必須是此課程類型，e.g. ["選修","通識"]
    "lmt_kind": [],         // 通識細分，e.g. ["社會通識"]
    "unit": []              // 開課系所，e.g. ["企管系"]
  },
  "must_not": {
    "course_lang": [],
    "weekday": [],
    "hour_range": null,
    "point": null,
    "kind": [],
    "lmt_kind": [],
    "unit": []
  }
}

規則：
- 否定詞（不要、排除、非、除了、避開、不含）開頭的條件放 must_not
- 只填有明確提到的條件，沒提到的保持 [] 或 null
- weekday 用數字 1-7（1=週一, 7=週日）
- hour_range 用 24 小時制 [開始,結束]（早上=[8,12], 下午=[13,18], 晚上=[18,24]）
- 直接輸出 JSON，不要加說明"""


def parse(query: str, model: str | None = None) -> ConditionResult:
    """
    P2：LLM zero-shot 解析。

    參數：
        query: 使用者的自然語言查詢
        model: 指定 LLM model（None 則用預設）
    回傳：
        ConditionResult
    """
    client, default_model = make_client()
    use_model = model or os.environ.get("PARSER_LLM_MODEL", default_model)

    try:
        r = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            max_tokens=400,
        )
        raw = r.choices[0].message.content or ""
        obj = extract_json(raw)
        if not obj:
            logger.warning(f"llm_zeroshot 無法解析 JSON: {raw[:100]}")
            return ConditionResult()
        return ConditionResult(
            must=obj.get("must", {}),
            must_not=obj.get("must_not", {}),
        )
    except Exception as e:
        logger.error(f"llm_zeroshot 呼叫失敗：{e}")
        return ConditionResult()
