"""
P0 Parser：包裝現有 rule-based 的 parse_constraints。

將 src/query_rewriters/structured.py 的 Constraints dataclass
轉換為新的 ConditionResult 格式。
"""

from __future__ import annotations

import logging
from typing import Any

from src.parsers.base import ConditionResult

logger = logging.getLogger(__name__)


def parse(query: str) -> ConditionResult:
    """
    使用現有的 rule-based parse_constraints 解析查詢字串，
    並將結果轉換為 ConditionResult。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        ConditionResult：結構化的搜尋條件
    """
    try:
        from src.query_rewriters.structured import parse_constraints
    except ImportError:
        logger.error("無法匯入 parse_constraints，請確認 src/query_rewriters/structured.py 存在")
        return ConditionResult()

    try:
        constraints = parse_constraints(query)
    except Exception as e:
        logger.error(f"parse_constraints 執行失敗：{e}")
        return ConditionResult()

    return _constraints_to_condition_result(constraints)


def _constraints_to_condition_result(constraints: Any) -> ConditionResult:
    """
    將舊的 Constraints dataclass 轉換成 ConditionResult。

    對應關係：
    - lang_include   → must["course_lang"]
    - lang_exclude   → must_not["course_lang"]
    - weekday_include → must["weekday"]
    - weekday_exclude → must_not["weekday"]
    - hour_min/hour_max → must["hour_range"] = [hour_min, hour_max]
    - hour_exclude_ranges → must_not["hour_range"]（取第一個）
    - unit_include   → must["unit"]
    - point_min/point_max → must["point"]（若 min==max 則取單值）
    - kind_include   → must["kind"]

    參數：
        constraints: 舊版 Constraints dataclass 實例

    回傳：
        ConditionResult：新版結構化條件
    """
    must: dict = {}
    must_not: dict = {}

    # course_lang
    lang_include = getattr(constraints, "lang_include", None) or []
    lang_exclude = getattr(constraints, "lang_exclude", None) or []
    if lang_include:
        must["course_lang"] = list(lang_include)
    if lang_exclude:
        must_not["course_lang"] = list(lang_exclude)

    # weekday
    weekday_include = getattr(constraints, "weekday_include", None) or []
    weekday_exclude = getattr(constraints, "weekday_exclude", None) or []
    if weekday_include:
        must["weekday"] = list(weekday_include)
    if weekday_exclude:
        must_not["weekday"] = list(weekday_exclude)

    # hour_range（must）
    hour_min = getattr(constraints, "hour_min", None)
    hour_max = getattr(constraints, "hour_max", None)
    if hour_min is not None and hour_max is not None:
        must["hour_range"] = [int(hour_min), int(hour_max)]
    elif hour_min is not None:
        must["hour_range"] = [int(hour_min), 24]
    elif hour_max is not None:
        must["hour_range"] = [0, int(hour_max)]

    # hour_range（must_not）：取 hour_exclude_ranges 的第一個
    hour_exclude_ranges = getattr(constraints, "hour_exclude_ranges", None) or []
    if hour_exclude_ranges:
        first_range = hour_exclude_ranges[0]
        if isinstance(first_range, (list, tuple)) and len(first_range) >= 2:
            must_not["hour_range"] = [int(first_range[0]), int(first_range[1])]

    # unit
    unit_include = getattr(constraints, "unit_include", None) or []
    if unit_include:
        must["unit"] = list(unit_include)

    # point：若 min==max 取單值（float）
    point_min = getattr(constraints, "point_min", None)
    point_max = getattr(constraints, "point_max", None)
    if point_min is not None and point_max is not None:
        if point_min == point_max:
            must["point"] = float(point_min)
        else:
            # 範圍情況：取中間值或保留 min（依業務邏輯決定，此處取 min）
            must["point"] = float(point_min)
    elif point_min is not None:
        must["point"] = float(point_min)
    elif point_max is not None:
        must["point"] = float(point_max)

    # kind
    kind_include = getattr(constraints, "kind_include", None) or []
    if kind_include:
        must["kind"] = list(kind_include)

    result = ConditionResult(must=must, must_not=must_not)
    logger.debug(f"rule_v0 解析結果：{result.to_dict()}")
    return result
