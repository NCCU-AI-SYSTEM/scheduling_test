"""Result table: one row per class, with what a student needs to decide whether to click.

Columns: 課名 / 老師 / 時間 / 學分 / 課號與對象 / 身分 / 評分方式 / 授課方式 / 連結.
Details stay on the official syllabus page (teaSchmUrl), which the row links to.

評分方式 comes from the syllabus evaluation text. Teachers write it in free form, so
items with a percentage are listed with it, items only mentioned are listed without,
and a syllabus that names none says 大綱未註明 instead of a guess.
"""

from __future__ import annotations

import re

from src.v4.assessment import FORMAT_KEYS
from src.v4.catalog import CatalogCourse
from src.v4.pipeline import Hit

_WEEKDAY = "一二三四五六日"
_TEACH_ZH = {"lecture": "講述", "discussion": "討論", "group": "小組", "elearning": "數位"}


def _key(text: str) -> str | None:
    return next((k for k, pat in FORMAT_KEYS if re.search(pat, text, re.I)), None)


def grading_summary(c: CatalogCourse, limit: int = 5) -> str:
    pct: dict[str, int] = {}
    for label, p in c.grading.get("items", []):
        k = _key(label)
        if k:
            pct[k] = pct.get(k, 0) + p
    text = c.course.evaluation or ""
    mentioned = [k for k, pat in FORMAT_KEYS if k not in pct and re.search(pat, text, re.I)]
    parts = [f"{k} {v}%" for k, v in sorted(pct.items(), key=lambda kv: -kv[1])] + mentioned
    return "、".join(parts[:limit]) if parts else "大綱未註明"


def teaching_summary(c: CatalogCourse) -> str:
    parts = [f"{_TEACH_ZH[k]} {v}%" for k, v in sorted(c.teaching.items(), key=lambda kv: -kv[1]) if v]
    return "、".join(parts) if parts else "大綱未註明"


def time_text(c: CatalogCourse) -> str:
    ss = c.meta.get("sessions") or []
    if not ss:
        return c.course.time_raw or "未定"
    return "、".join(f"週{_WEEKDAY[s['weekday'] - 1]} {s['start']}–{s['end']}" for s in ss)


def row(h: Hit, cmap: dict[str, CatalogCourse]) -> dict:
    c = cmap[h.course_id]
    a = h.access
    if a is None:
        status = "未提供學號"
    else:
        notes = [f.detail for f in a.flags if f.level in ("warn", "info")]
        status = "符合" if not notes else "；".join(notes)
    others = []
    if h.offering and len(h.offering.channels) > 1:
        others = [f"{ch.sub_num}（{ch.unit}）" for ch in h.offering.channels if ch.course_id != h.course_id]
    url = a.channel.url if a else next((ch.url for ch in (h.offering.channels if h.offering else ())
                                        if ch.course_id == h.course_id), "")
    return {"課名": c.course.name, "老師": c.course.teacher, "時間": time_text(c), "學分": c.course.point,
            "課號": c.course.sub_num, "對象": c.course.unit, "其他課號": others, "身分": status,
            "評分方式": grading_summary(c) + ("（無法確認是否符合評分條件）"
                                            if any(f.kind == "assessment_unknown" for f in h.flags) else ""),
            "授課方式": teaching_summary(c), "連結": url}


def to_markdown(rows: list[dict]) -> str:
    head = "| # | 課名 | 老師 | 時間 | 學分 | 課號（對象） | 身分 | 評分方式 | 授課方式 | 大綱 |\n|---|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for i, r in enumerate(rows, 1):
        code = f"{r['課號']}（{r['對象']}）" + (f"<br>另有 {'、'.join(r['其他課號'])}" if r["其他課號"] else "")
        link = f"[連結]({r['連結']})" if r["連結"] else ""
        lines.append(f"| {i} | {r['課名']} | {r['老師']} | {r['時間']} | {r['學分']:g} | {code} | {r['身分']} | "
                     f"{r['評分方式']} | {r['授課方式']} | {link} |")
    return "\n".join(lines)
