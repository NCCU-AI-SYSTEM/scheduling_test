"""
課程搜尋條件轉 SQL WHERE 子句模組。

將 ConditionResult 轉換為可用於 PostgreSQL 查詢的 WHERE 字串和參數列表。
"""

from __future__ import annotations

import logging
from typing import Any

from src.parsers.base import ConditionResult

logger = logging.getLogger(__name__)

# 課程類型（kind）的字串到數字對應
KIND_TO_INT: dict[str, int] = {
    "必修": 1,
    "選修": 2,
    "通識": 3,
    "體育": 4,
}


def _build_lang_clause(langs: list[str], negate: bool) -> tuple[str, list[Any]]:
    """
    建立課程語言的 SQL 片段。
    must  → lang = ANY(%s)
    must_not → NOT (lang = ANY(%s))

    參數：
        langs: 語言列表
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    if negate:
        return "NOT (lang = ANY(%s))", [langs]
    return "lang = ANY(%s)", [langs]


def _build_weekday_clause(weekdays: list[int], negate: bool) -> tuple[str, list[Any]]:
    """
    建立星期的 SQL 片段（使用 GIN index 的陣列重疊運算子 &&）。
    must  → weekdays && %s
    must_not → NOT (weekdays && %s)

    參數：
        weekdays: 星期數字列表（1~7）
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    if negate:
        return "NOT (weekdays && %s)", [weekdays]
    return "weekdays && %s", [weekdays]


def _build_hour_range_clause(hour_range: list[int], negate: bool) -> tuple[str, list[Any]]:
    """
    建立時間範圍的 SQL 片段（使用 JSONB sessions 的重疊判斷）。
    判斷條件：session 的 [start_hour, end_hour] 與 [hr_start, hr_end] 有重疊，
    即 start_hour < hr_end AND end_hour > hr_start。

    must  → EXISTS (SELECT 1 FROM jsonb_array_elements(sessions) AS s
                    WHERE (s->>'start_hour')::int < %s AND (s->>'end_hour')::int > %s)
    must_not → NOT EXISTS (...)

    參數：
        hour_range: [開始小時, 結束小時]
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    hr_start, hr_end = hour_range[0], hour_range[1]
    inner = (
        "EXISTS ("
        "SELECT 1 FROM jsonb_array_elements(sessions) AS s "
        "WHERE (s->>'start_hour')::int < %s AND (s->>'end_hour')::int > %s"
        ")"
    )
    if negate:
        return f"NOT {inner}", [hr_end, hr_start]
    return inner, [hr_end, hr_start]


def _build_point_clause(point: float, negate: bool) -> tuple[str, list[Any]]:
    """
    建立學分數的 SQL 片段。
    must  → point = %s
    must_not → NOT (point = %s)

    參數：
        point: 學分數
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    if negate:
        return "NOT (point = %s)", [point]
    return "point = %s", [point]


def _build_kind_clause(kinds: list[str], negate: bool) -> tuple[str, list[Any]]:
    """
    建立課程類型的 SQL 片段（字串轉數字對應）。
    must  → kind = ANY(%s)
    must_not → NOT (kind = ANY(%s))

    參數：
        kinds: 課程類型字串列表（必修/選修/通識/體育）
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    int_kinds = [KIND_TO_INT[k] for k in kinds if k in KIND_TO_INT]
    if not int_kinds:
        logger.warning(f"kind 清單中有未知值：{kinds}，已略過")
        return "", []
    if negate:
        return "NOT (kind = ANY(%s))", [int_kinds]
    return "kind = ANY(%s)", [int_kinds]


def _build_lmt_kind_clause(lmt_kinds: list[str], negate: bool) -> tuple[str, list[Any]]:
    """
    建立通識細分的 SQL 片段。
    must  → lmt_kind = ANY(%s)
    must_not → NOT (lmt_kind = ANY(%s))

    參數：
        lmt_kinds: 通識細分列表
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    if negate:
        return "NOT (lmt_kind = ANY(%s))", [lmt_kinds]
    return "lmt_kind = ANY(%s)", [lmt_kinds]


def _build_unit_clause(units: list[str], negate: bool) -> tuple[str, list[Any]]:
    """
    建立開課系所的 SQL 片段（使用 LIKE 模糊比對）。
    must  → unit LIKE ANY(%s)
    must_not → NOT (unit LIKE ANY(%s))

    參數：
        units: 系所名稱列表
        negate: 是否加 NOT

    回傳：
        (clause_str, params): SQL 片段和參數
    """
    like_units = [f"%{u}%" for u in units]
    if negate:
        return "NOT (unit LIKE ANY(%s))", [like_units]
    return "unit LIKE ANY(%s)", [like_units]


def conditions_to_sql(conditions: ConditionResult) -> tuple[str, list]:
    """
    將 ConditionResult 轉換為 SQL WHERE 子句字串和參數列表。

    支援維度：
    - course_lang → lang = ANY(%s)
    - weekday     → weekdays && %s（GIN index）
    - hour_range  → EXISTS (JSONB sessions 重疊判斷)
    - point       → point = %s
    - kind        → kind = ANY(%s)（字串轉數字）
    - lmt_kind    → lmt_kind = ANY(%s)
    - unit        → unit LIKE ANY(%s)

    must_not 的所有條件加上 NOT。

    參數：
        conditions: 課程搜尋條件

    回傳：
        (where_clause, params): WHERE 子句字串（不含 WHERE 關鍵字）和對應參數列表
        若無任何條件，回傳 ("TRUE", [])
    """
    clauses: list[str] = []
    params: list[Any] = []

    # ── must 條件 ──────────────────────────────────────────────────────────
    must = conditions.must

    course_lang = must.get("course_lang")
    if course_lang:
        c, p = _build_lang_clause(course_lang, negate=False)
        clauses.append(c)
        params.extend(p)

    weekday = must.get("weekday")
    if weekday:
        c, p = _build_weekday_clause(weekday, negate=False)
        clauses.append(c)
        params.extend(p)

    hour_range = must.get("hour_range")
    if hour_range and isinstance(hour_range, list) and len(hour_range) == 2:
        c, p = _build_hour_range_clause(hour_range, negate=False)
        clauses.append(c)
        params.extend(p)

    point = must.get("point")
    if point is not None:
        c, p = _build_point_clause(float(point), negate=False)
        clauses.append(c)
        params.extend(p)

    kind = must.get("kind")
    if kind:
        c, p = _build_kind_clause(kind, negate=False)
        if c:
            clauses.append(c)
            params.extend(p)

    lmt_kind = must.get("lmt_kind")
    if lmt_kind:
        c, p = _build_lmt_kind_clause(lmt_kind, negate=False)
        clauses.append(c)
        params.extend(p)

    unit = must.get("unit")
    if unit:
        c, p = _build_unit_clause(unit, negate=False)
        clauses.append(c)
        params.extend(p)

    # ── must_not 條件 ──────────────────────────────────────────────────────
    must_not = conditions.must_not

    mn_course_lang = must_not.get("course_lang")
    if mn_course_lang:
        c, p = _build_lang_clause(mn_course_lang, negate=True)
        clauses.append(c)
        params.extend(p)

    mn_weekday = must_not.get("weekday")
    if mn_weekday:
        c, p = _build_weekday_clause(mn_weekday, negate=True)
        clauses.append(c)
        params.extend(p)

    mn_hour_range = must_not.get("hour_range")
    if mn_hour_range and isinstance(mn_hour_range, list) and len(mn_hour_range) == 2:
        c, p = _build_hour_range_clause(mn_hour_range, negate=True)
        clauses.append(c)
        params.extend(p)

    mn_point = must_not.get("point")
    if mn_point is not None:
        c, p = _build_point_clause(float(mn_point), negate=True)
        clauses.append(c)
        params.extend(p)

    mn_kind = must_not.get("kind")
    if mn_kind:
        c, p = _build_kind_clause(mn_kind, negate=True)
        if c:
            clauses.append(c)
            params.extend(p)

    mn_lmt_kind = must_not.get("lmt_kind")
    if mn_lmt_kind:
        c, p = _build_lmt_kind_clause(mn_lmt_kind, negate=True)
        clauses.append(c)
        params.extend(p)

    mn_unit = must_not.get("unit")
    if mn_unit:
        c, p = _build_unit_clause(mn_unit, negate=True)
        clauses.append(c)
        params.extend(p)

    # ── 組合結果 ──────────────────────────────────────────────────────────
    if not clauses:
        logger.debug("conditions_to_sql：無任何條件，回傳 TRUE")
        return "TRUE", []

    where_clause = " AND ".join(clauses)
    logger.debug(f"conditions_to_sql WHERE：{where_clause}，params：{params}")
    return where_clause, params
