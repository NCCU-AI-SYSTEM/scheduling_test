"""Target classes from the unit string.

`unit` is usually a concatenation of class labels: 法律三甲法律三乙, 外交碩一外交博一,
科智碩1A科智碩2A, 地三土管地三土資 (地政 year 3, groups 土管 and 土資). 77 of the 544
distinct values are unit names instead (體育, 外文中心, 商院選修) and yield no class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_YEAR = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
_TOKEN = re.compile(r"(\D+?)([一二三四五六七]|[1-7])([甲乙丙丁戊A-Z]?)")


@dataclass(frozen=True)
class TargetClass:
    dept_abbr: str
    level: str          # 學士 / 碩士 / 博士 / 專班
    year: int | None
    section: str
    raw: str


def _level(abbr: str) -> str:
    if "博" in abbr:
        return "博士"
    if "碩" in abbr:
        return "碩士"
    if "專" in abbr:
        return "專班"
    return "學士"


def _dept(abbr: str) -> str:
    """Drop a trailing level marker: 外交碩 -> 外交; 資碩工 keeps its inner 碩."""
    return re.sub(r"(碩|博|專)$", "", abbr) or abbr


def parse_unit(unit: str | None) -> list[TargetClass]:
    s = (unit or "").strip()
    raw_tokens = [(m.group(1), m.group(2), m.group(3), m.end()) for m in _TOKEN.finditer(s)]
    if not raw_tokens or raw_tokens[0][0] != s[: len(raw_tokens[0][0])]:
        return []
    first_abbr = raw_tokens[0][0]
    out: list[list] = []
    for abbr, year, sec, _ in raw_tokens:
        # 地三土管地三土資: the second abbr '土管地' is group '土管' of the previous class + '地'
        if out and abbr != first_abbr and abbr.endswith(first_abbr) and len(abbr) - len(first_abbr) <= 3:
            out[-1][3] = out[-1][3] or abbr[: -len(first_abbr)]
            abbr = first_abbr
        out.append([abbr, year, sec, ""])
    tail = s[raw_tokens[-1][3]:]
    if tail:
        if tail.startswith(first_abbr) and len(tail) > len(first_abbr):
            out.append([first_abbr, "", "", tail[len(first_abbr):]])   # 歐文德選
        elif len(tail) <= 3:
            out[-1][3] = out[-1][3] or tail                            # 地三土測
        else:
            return []
    return [TargetClass(_dept(a), _level(a), _YEAR.get(y) or (int(y) if y.isdigit() else None),
                        sec + grp, f"{a}{y}{sec}{grp}") for a, y, sec, grp in out]


def is_class_unit(unit: str | None) -> bool:
    return bool(parse_unit(unit))
