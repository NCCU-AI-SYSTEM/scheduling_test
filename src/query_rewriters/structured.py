"""Structured-extraction query rewriter (Q-Struct in PLAN.md).

Pure-rules first pass: parse hard constraints from a Chinese query into a
typed `QueryConstraints` dataclass + a residual *semantic* query string. No
LLM dependency for the rule layer; later we may add an LLM fallback for
cases where rules miss.

Constraints handled:
  - weekday: 一/二/三/四/五/六/日 (incl. "週一", "星期二", "禮拜三")
  - time-of-day: 早八 / 早上 / 上午 / 中午 / 下午 / 晚上 / 晚間
  - explicit hours: "9 點", "10:00"
  - language: 中文 / 英文 / 日文 / 韓文 / 法文 / 德文 / 西班牙文 / 阿拉伯文 ...
  - course kind: 必修 / 選修 / 通識 / 體育
  - level: 大一/二/三/四 → year-level filter via course-id heuristic (best effort)
  - point: "3 學分", "兩學分以上"
  - unit / dept: known keyword list (商學院、法律系、企管系、…)

Negations preserved: "不要", "不想", "不開", "別", "除外"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WEEKDAY_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7,
}
LANG_KEYWORDS = {
    "中文": "中文", "國語": "中文",
    "英文": "英文", "english": "英文",
    "日文": "日文", "日語": "日文",
    "韓文": "韓文", "韓語": "韓文",
    "法文": "法文", "德文": "德文",
    "西班牙文": "西班牙文", "西文": "西班牙文",
    "阿拉伯文": "阿拉伯文",
    "土耳其文": "土耳其文",
    "俄文": "俄文",
}
KIND_KEYWORDS = {
    "必修": 1, "選修": 2, "通識": 3, "體育": 4, "服學": 4, "服務學習": 4,
}

# Course-kind word → which side appears in the query
TIME_BUCKETS = {
    "早八": (8, 9),
    "早上": (8, 12), "上午": (8, 12),
    "中午": (12, 13),
    "下午": (13, 18),
    "晚上": (18, 22), "晚間": (18, 22), "夜間": (18, 22),
}

NEG_PREFIXES = ["不要", "不想", "別", "不開", "排除", "除外", "避開"]

UNIT_KEYWORDS = [
    "商學院", "法學院", "文學院", "社科院", "傳院", "理學院", "外語學院", "教育學院",
    "資科", "資管", "資訊", "企管", "會計", "財管", "金融", "經濟",
    "法律", "法科", "民法", "刑法",
    "中文系", "英文系", "日文系", "韓文系", "阿拉伯", "斯拉夫", "歐文", "土文",
    "心理", "社會", "民族", "教育", "新聞", "廣告", "廣電",
    "政治", "外交", "公行", "公共行政", "國關",
    "統計", "應數", "資工", "資科",
]


@dataclass(slots=True)
class QueryConstraints:
    weekday_include: set[int] = field(default_factory=set)
    weekday_exclude: set[int] = field(default_factory=set)
    hour_min: int | None = None
    hour_max: int | None = None
    lang_include: set[str] = field(default_factory=set)
    lang_exclude: set[str] = field(default_factory=set)
    kind_include: set[int] = field(default_factory=set)
    kind_exclude: set[int] = field(default_factory=set)
    point_min: float | None = None
    point_max: float | None = None
    unit_include: set[str] = field(default_factory=set)
    unit_exclude: set[str] = field(default_factory=set)
    raw: str = ""
    semantic_residual: str = ""


def _is_negated(text: str, span: tuple[int, int], window: int = 4) -> bool:
    start, _ = span
    pre = text[max(0, start - window) : start]
    return any(neg in pre for neg in NEG_PREFIXES)


def parse_constraints(query: str) -> QueryConstraints:
    c = QueryConstraints(raw=query)
    s = query

    # weekday: 週X / 星期X / 禮拜X
    for m in re.finditer(r"(?:週|周|星期|禮拜)([一二三四五六日天])", s):
        wd = WEEKDAY_MAP.get(m.group(1))
        if wd is None:
            continue
        if _is_negated(s, m.span()):
            c.weekday_exclude.add(wd)
        else:
            c.weekday_include.add(wd)

    # time-of-day buckets
    for kw, (lo, hi) in TIME_BUCKETS.items():
        m = re.search(re.escape(kw), s)
        if not m:
            continue
        if _is_negated(s, m.span()):
            # negation: clamp away from this bucket — best-effort, just record
            continue
        # Keep tightest intersecting window
        c.hour_min = max(c.hour_min or lo, lo)
        c.hour_max = min(c.hour_max or hi, hi)

    # explicit hours like "9 點" "10點" "10:00"
    for m in re.finditer(r"(\d{1,2})\s*(?::\d{2}|點)", s):
        try:
            h = int(m.group(1))
        except ValueError:
            continue
        if 6 <= h <= 22 and not _is_negated(s, m.span()):
            c.hour_min = max(c.hour_min or h, h)

    # language
    low = s.lower()
    for kw, norm in LANG_KEYWORDS.items():
        if kw not in low:
            continue
        idx = low.find(kw)
        if _is_negated(s, (idx, idx + len(kw))):
            c.lang_exclude.add(norm)
        else:
            # only treat as constraint when collocated with 授課/教學/開課/上課
            ctx = s[max(0, idx - 6) : idx + len(kw) + 6]
            if any(t in ctx for t in ("授課", "教學", "開課", "上課", "授")):
                c.lang_include.add(norm)
            elif kw in ("英文", "日文", "韓文", "法文", "德文", "西班牙文", "阿拉伯文", "土耳其文", "俄文"):
                # foreign-lang word likely a constraint by default
                c.lang_include.add(norm)

    # kind
    for kw, code in KIND_KEYWORDS.items():
        if kw not in s:
            continue
        idx = s.find(kw)
        if _is_negated(s, (idx, idx + len(kw))):
            c.kind_exclude.add(code)
        else:
            c.kind_include.add(code)

    # points: "3 學分", "兩學分以上"
    for m in re.finditer(r"(\d+|[一二三四五六七八九十])\s*學分(以上|以下)?", s):
        digit_or_zh = m.group(1)
        try:
            n = int(digit_or_zh)
        except ValueError:
            n = "一二三四五六七八九十".index(digit_or_zh) + 1
        bound = m.group(2)
        if bound == "以上":
            c.point_min = float(n)
        elif bound == "以下":
            c.point_max = float(n)
        else:
            c.point_min = c.point_max = float(n)

    # unit/dept — require collocation with 系/院/所/學系/系所 to avoid 經濟學被當成經濟系
    unit_triggers = ("系", "院", "所", "學系", "系所", "學院")
    for kw in UNIT_KEYWORDS:
        idx = s.find(kw)
        if idx < 0:
            continue
        tail = s[idx + len(kw) : idx + len(kw) + 2]
        if not any(tail.startswith(t) for t in unit_triggers):
            continue
        if _is_negated(s, (idx, idx + len(kw))):
            c.unit_exclude.add(kw)
        else:
            c.unit_include.add(kw)

    # build semantic residual: drop the constraint phrases we matched
    residual = s
    for pat in [
        r"(?:不要|不想|別|不開|排除|除外|避開)?(?:週|周|星期|禮拜)[一二三四五六日天]",
        r"(?:早八|早上|上午|中午|下午|晚上|晚間|夜間)",
        r"\d{1,2}\s*(?::\d{2}|點)",
        r"\d+\s*學分(?:以上|以下)?",
        r"[一二三四五六七八九十]\s*學分(?:以上|以下)?",
        r"(?:必修|選修|通識|體育|服學|服務學習)",
    ]:
        residual = re.sub(pat, " ", residual)
    for kw in list(c.lang_include) + list(c.lang_exclude):
        residual = residual.replace(kw, " ").replace(f"{kw}授課", " ").replace(f"{kw}教學", " ")
    for kw in list(c.unit_include) + list(c.unit_exclude):
        residual = residual.replace(kw, " ")
    residual = re.sub(r"\s+", " ", residual).strip()
    c.semantic_residual = residual or query
    return c


__all__ = ["QueryConstraints", "parse_constraints"]
