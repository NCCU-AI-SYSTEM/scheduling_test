"""Scalar course fields: codes turned into labels, numbers out of templates.

kind codes follow NCCUCourse/DB.py (1 必修, 2 選修, 3 群修, 4 lmtKind contains 通識,
0 other). General education is decided by lmtKind, not kind: the 18 跨領域 courses have
kind 0. teaching_approach is a fixed template `N%|講述|Lecture|N%|討論|...|其他：|Others:`.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.v4.structured.text import TEACH_HEAD, nfkc

ROOT = Path(__file__).resolve().parents[3]
UNIT_JSON = ROOT / "data" / "v4" / "unit.json"

KIND_LABEL = {1: "必修", 2: "選修", 3: "群修", 4: "通識", 0: "其他"}
GE_CATEGORIES = ("外文", "社會", "人文", "中文", "自然", "資訊", "跨領域", "書院")
SPAN = {"單學期科目": "單學期", "全學年科目": "全學年", "三學期科目": "三學期"}
TEACH_KEYS = {"講述": "lecture", "討論": "discussion", "小組活動": "group", "數位學習": "elearning", "其他": "other"}
_TEACH = re.compile(r"(\d+)?\s*%\|(講述|討論|小組活動|數位學習|其他)[:：]?\|[A-Za-z \-]+:?")
_ROOM = re.compile(r"^(\D*?)\s*\(?(\d{6})\)?$")


def kind_label(kind: int | None) -> str:
    return KIND_LABEL.get(kind or 0, "其他")


def general_ed(lmt_kind: str | None, core: int | None) -> tuple[str | None, str | None, bool]:
    """lmtKind -> (category, detail, core). 跨領域(人文、社會) -> ('跨領域', '人文、社會')."""
    s = nfkc(lmt_kind)
    if not s:
        return None, None, False
    cat = next((c for c in GE_CATEGORIES if s.startswith(c)), s)
    m = re.search(r"\((.+)\)", s)
    return cat, (m.group(1) if m else None), bool(core)


def teaching(text: str | None) -> dict:
    """Percent per teaching mode plus the free label after 其他 (e.g. 實作)."""
    body = TEACH_HEAD.sub("", nfkc(text))
    out: dict = {}
    for pct, label in _TEACH.findall(body):
        out[TEACH_KEYS[label]] = int(pct) if pct else None
    m = re.search(r"Others:\|?([^|]+)", body)
    out["other_label"] = m.group(1).strip() if m and m.group(1).strip() else None
    vals = [v for k, v in out.items() if k != "other_label" and v is not None]
    out["sum"] = sum(vals) if vals else None
    return out


def classroom(raw: str | None) -> tuple[str | None, str | None]:
    s = nfkc(raw)
    if not s:
        return None, None
    m = _ROOM.match(s)
    return (m.group(1).strip() or None, m.group(2)) if m else (s, None)


@lru_cache(maxsize=1)
def unit_names() -> dict[tuple[str, str, str], str]:
    """(dp1, dp2, dp3) -> department / channel name from qrysub unit.json."""
    data = json.loads(UNIT_JSON.read_text(encoding="utf-8"))
    out = {}
    for l1 in data:
        for l2 in l1.get("utL2", []):
            for l3 in l2.get("utL3", []):
                if l3["utCodL3"] != "0":
                    out[(l1["utCodL1"], l2["utCodL2"], l3["utCodL3"])] = l3["utL3Text"]
    return out


def channel_label(dp: tuple[str, str, str]) -> str | None:
    name = unit_names().get(dp)
    return re.split(r"\s+(?=[A-Za-z])", name, maxsplit=1)[0].strip() if name else None


def dept_name(dept_code: str, paths: list[tuple[str, str, str]]) -> str | None:
    """Name of the opening department: the dp3 equal to subNum[:3], else any college row
    in the catalogue tree carrying that code."""
    for dp in paths:
        if dp[2] == dept_code:
            label = channel_label(dp)
            if label:
                return label
    for (d1, d2, d3), _ in unit_names().items():
        if d3 == dept_code and d1[0].isdigit() and d1 not in ("01", "02", "03"):
            return channel_label((d1, d2, d3))
    return None
