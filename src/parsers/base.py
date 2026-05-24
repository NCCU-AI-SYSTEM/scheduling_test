"""
課程搜尋條件的基礎資料結構。
定義 ConditionResult dataclass，供所有 parser 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConditionResult:
    """
    課程搜尋條件的結構化表示。

    欄位說明：
    - must：必須符合的條件（AND 邏輯）
    - must_not：必須排除的條件（NOT 邏輯）
    - should：選擇性偏好條件（BOOST 邏輯）

    must 和 must_not 的 key 為維度名稱，支援以下維度：
    - course_lang: list[str]        e.g. ["英文"]
    - weekday: list[int]            e.g. [1, 2, 3]（1=週一，7=週日）
    - hour_range: list[int] | None  e.g. [13, 18]（start inclusive, end inclusive）
    - point: float | None           e.g. 3.0
    - kind: list[str]               e.g. ["選修"]
    - lmt_kind: list[str]           e.g. ["社會通識"]
    - unit: list[str]               e.g. ["企管系"]
    """

    must: dict = field(default_factory=dict)
    must_not: dict = field(default_factory=dict)
    should: dict = field(default_factory=dict)

    def has_any_condition(self) -> bool:
        """
        檢查 must 和 must_not 是否有任何非空的條件值。

        回傳：
            bool：若有任何非空條件則為 True，否則為 False
        """
        def _has_value(d: dict) -> bool:
            for v in d.values():
                if v is None:
                    continue
                if isinstance(v, list) and len(v) == 0:
                    continue
                return True
            return False

        return _has_value(self.must) or _has_value(self.must_not)

    def to_dict(self) -> dict:
        """
        將 ConditionResult 序列化為可 JSON 輸出的 dict。

        回傳：
            dict：包含 must、must_not、should 三個 key 的字典
        """
        def _clean(d: dict) -> dict:
            """移除空值，保留有意義的資料。"""
            result: dict[str, Any] = {}
            for k, v in d.items():
                if v is None:
                    result[k] = None
                elif isinstance(v, list):
                    result[k] = v
                else:
                    result[k] = v
            return result

        return {
            "must": _clean(self.must),
            "must_not": _clean(self.must_not),
            "should": _clean(self.should),
        }
