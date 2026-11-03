"""
P1 Parser：Regex + 否定詞清單。

使用正規表示式配合否定詞偵測，解析自然語言查詢中的課程搜尋條件。
不依賴任何外部 API，純本地計算。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.parsers.base import ConditionResult

logger = logging.getLogger(__name__)

# 否定詞前綴（由長到短排列，避免短詞遮蔽長詞）
NEG_WORDS = ["不要", "不含", "不是", "排除", "除了", "沒有", "避免", "非", "不"]

# 漢字星期對應數字
WEEKDAY_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "日": 7, "天": 7,
}

# 課程類型
KIND_MAP = {
    "必修": "必修",
    "選修": "選修",
    "通識": "通識",
    "體育": "體育",
}

# 通識細分
LMT_KIND_LIST = [
    "外文通識", "社會通識", "人文通識", "自然通識",
    "資訊通識", "書院通識", "中文通識",
]

# 語言列表
LANG_LIST = [
    "中文", "英文", "日文", "韓文", "法文", "德文",
    "西班牙文", "阿拉伯文", "土耳其文", "越南文",
    "泰文", "印尼文", "俄文",
]


def _has_neg_prefix(text: str, match_start: int, window: int = 10) -> bool:
    """
    檢查比對到的字串前方（window 字元內）是否有否定詞。

    參數：
        text: 完整查詢字串
        match_start: 正規比對開始位置
        window: 向前查找的字元數

    回傳：
        bool：若有否定詞則為 True
    """
    prefix_start = max(0, match_start - window)
    prefix = text[prefix_start:match_start]
    for neg in NEG_WORDS:
        if neg in prefix:
            return True
    return False


def _parse_course_lang(query: str, must: dict, must_not: dict) -> None:
    """解析課程語言條件。"""
    pattern = r"(" + "|".join(LANG_LIST) + r")[授課上]?"
    for m in re.finditer(pattern, query):
        lang = m.group(1)
        if _has_neg_prefix(query, m.start()):
            must_not.setdefault("course_lang", [])
            if lang not in must_not["course_lang"]:
                must_not["course_lang"].append(lang)
        else:
            must.setdefault("course_lang", [])
            if lang not in must["course_lang"]:
                must["course_lang"].append(lang)


def _parse_weekday(query: str, must: dict, must_not: dict) -> None:
    """
    解析星期條件。
    支援「星期一」「週二」「周三」以及「一到三」「一至五」範圍寫法。
    """
    # 範圍：「週一到週三」「星期一到五」「一到三」「一至五」
    range_pattern = r"(?:星期|週|周)?([一二三四五六日天])(?:到|至)(?:星期|週|周)?([一二三四五六日天])"
    range_neg: list[int] = []
    range_pos: list[int] = []

    for m in re.finditer(range_pattern, query):
        start_day = WEEKDAY_MAP.get(m.group(1))
        end_day = WEEKDAY_MAP.get(m.group(2))
        if start_day is None or end_day is None:
            continue
        days = list(range(start_day, end_day + 1))
        if _has_neg_prefix(query, m.start()):
            range_neg.extend(days)
        else:
            range_pos.extend(days)

    # 移除範圍已涵蓋的位置，避免重複解析（標記已處理的 span）
    handled_spans: list[tuple[int, int]] = [
        m.span() for m in re.finditer(range_pattern, query)
    ]

    # 單一星期
    single_pattern = r"(?:星期|週|周)([一二三四五六日天])"
    for m in re.finditer(single_pattern, query):
        # 確認不在已處理的範圍span內
        if any(s <= m.start() < e for s, e in handled_spans):
            continue
        day = WEEKDAY_MAP.get(m.group(1))
        if day is None:
            continue
        if _has_neg_prefix(query, m.start()):
            range_neg.append(day)
        else:
            range_pos.append(day)

    if range_pos:
        must["weekday"] = sorted(set(range_pos))
    if range_neg:
        must_not["weekday"] = sorted(set(range_neg))


def _parse_hour_range(query: str, must: dict, must_not: dict) -> None:
    """
    解析時段條件。
    支援時段名稱（早上/下午等）以及精確時間（9點到12點、9:00-12:00）。
    """
    # 精確時間：9點到12點 或 9:00-12:00
    precise_pattern = r"(\d{1,2})(?:點|:00)\s*(?:到|-)\s*(\d{1,2})(?:點|:00)?"
    for m in re.finditer(precise_pattern, query):
        h_start = int(m.group(1))
        h_end = int(m.group(2))
        hr = [h_start, h_end]
        if _has_neg_prefix(query, m.start()):
            must_not["hour_range"] = hr
        else:
            must["hour_range"] = hr

    # 時段名稱（若精確時間已設定則不覆蓋）
    time_slots = [
        (r"早上|上午", [8, 12]),
        (r"中午", [12, 13]),
        (r"下午|午後", [13, 18]),
        (r"晚上|晚間|夜間", [18, 24]),
    ]
    for pattern, hr in time_slots:
        for m in re.finditer(pattern, query):
            if _has_neg_prefix(query, m.start()):
                if "hour_range" not in must_not:
                    must_not["hour_range"] = hr
            else:
                if "hour_range" not in must:
                    must["hour_range"] = hr


def _parse_point(query: str, must: dict, must_not: dict) -> None:
    """解析學分條件。"""
    # 支援：1學分、2.0學分、三學分
    cn_digit = {"一": "1", "二": "2", "三": "3", "四": "4"}
    # 先把中文數字替換成阿拉伯數字（只替換學分前的數字）
    normalized = query
    for cn, ar in cn_digit.items():
        normalized = re.sub(cn + r"(?=\s*學分)", ar, normalized)

    pattern = r"([1-4](?:\.0)?)\s*學分"
    for m in re.finditer(pattern, normalized):
        point_val = float(m.group(1))
        if _has_neg_prefix(normalized, m.start()):
            must_not["point"] = point_val
        else:
            must["point"] = point_val


def _parse_kind(query: str, must: dict, must_not: dict) -> None:
    """
    解析課程類型條件。
    注意：通識細分（lmt_kind）優先於通識（kind），避免衝突。
    """
    # 先移除通識細分的部分，避免「社會通識」被誤判為「通識」
    lmt_kind_pattern = "|".join(LMT_KIND_LIST)
    masked = re.sub(lmt_kind_pattern, "　　", query)  # 用全形空白佔位

    for keyword, kind_val in KIND_MAP.items():
        for m in re.finditer(re.escape(keyword), masked):
            if _has_neg_prefix(masked, m.start()):
                must_not.setdefault("kind", [])
                if kind_val not in must_not["kind"]:
                    must_not["kind"].append(kind_val)
            else:
                must.setdefault("kind", [])
                if kind_val not in must["kind"]:
                    must["kind"].append(kind_val)


def _parse_lmt_kind(query: str, must: dict, must_not: dict) -> None:
    """解析通識細分條件。"""
    for lmt in LMT_KIND_LIST:
        for m in re.finditer(re.escape(lmt), query):
            if _has_neg_prefix(query, m.start()):
                must_not.setdefault("lmt_kind", [])
                if lmt not in must_not["lmt_kind"]:
                    must_not["lmt_kind"].append(lmt)
            else:
                must.setdefault("lmt_kind", [])
                if lmt not in must["lmt_kind"]:
                    must["lmt_kind"].append(lmt)


def _parse_unit(query: str, must: dict, must_not: dict) -> None:
    """
    解析開課系所條件。
    設計上 unit 的否定不做處理（跳過），只解析正向條件。
    """
    pattern = r"([\u4e00-\u9fff]{2,6}(?:系|所|院|中心|學程))"
    # 排除通識細分關鍵字被誤判
    lmt_kind_pattern = "|".join(LMT_KIND_LIST)

    for m in re.finditer(pattern, query):
        unit_name = m.group(1)
        # 若是通識細分的一部分，跳過
        if any(lmt in unit_name for lmt in LMT_KIND_LIST):
            continue
        # unit 的否定不做
        if not _has_neg_prefix(query, m.start()):
            must.setdefault("unit", [])
            if unit_name not in must["unit"]:
                must["unit"].append(unit_name)


def parse(query: str) -> ConditionResult:
    """
    使用 Regex + 否定詞清單解析查詢字串。

    參數：
        query: 使用者輸入的自然語言查詢

    回傳：
        ConditionResult：結構化的搜尋條件
    """
    must: dict = {}
    must_not: dict = {}

    _parse_lmt_kind(query, must, must_not)   # 優先解析，避免被 kind 覆蓋
    _parse_course_lang(query, must, must_not)
    _parse_weekday(query, must, must_not)
    _parse_hour_range(query, must, must_not)
    _parse_point(query, must, must_not)
    _parse_kind(query, must, must_not)
    _parse_unit(query, must, must_not)

    result = ConditionResult(must=must, must_not=must_not)
    logger.debug(f"regex_v1 解析結果：{result.to_dict()}")
    return result
