"""Build textual representations of courses for retrieval indexing.

Each builder returns a `RetrievalDoc` whose `text` is what BM25 / dense
retrievers see. `metadata` carries structured fields for the SQL filter layer.

Variants:
  D-Base  : name + teacher + time   (reproduces CourseLangChain baseline)
  D-Obj   : + objective + classroom + unit + lang
  D-V2    : + LLM summary + keywords + topic_tags + weekly_topics  (needs meta_gen)
  D-V3    : multi-field — returns multiple sub-docs per course, each tied to a
            field with its own weight. Caller is responsible for fusion.

W5 ships D-Base and D-Obj only; D-V2/V3 land in W6 once meta_gen finishes.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.loader import Course

ROOT = Path(__file__).resolve().parents[2]
META_DB = ROOT / "data" / "processed" / "course_meta.db"


@dataclass(slots=True)
class RetrievalDoc:
    course_id: str  # gold target
    text: str
    metadata: dict = field(default_factory=dict)
    field_name: str = "all"  # for multi-field (V3) builders
    weight: float = 1.0


def _course_metadata(c: Course) -> dict:
    return {
        "course_id": c.course_id,
        "name": c.name,
        "teacher": c.teacher,
        "lang": c.lang,
        "kind": c.kind,
        "lmt_kind": c.lmt_kind,
        "point": c.point,
        "unit": c.unit,
        "time_raw": c.time_raw,
        "sessions": [
            {"weekday": s.weekday, "start": s.start_hour, "end": s.end_hour}
            for s in c.sessions
        ],
    }


def _time_str(c: Course) -> str:
    if not c.sessions:
        return "未定"
    parts: list[str] = []
    weekday_zh = ["", "一", "二", "三", "四", "五", "六", "日"]
    for s in c.sessions:
        parts.append(f"星期{weekday_zh[s.weekday]} {s.start_hour}:00-{s.end_hour}:00")
    return "、".join(parts)


def build_d_base(courses: list[Course]) -> list[RetrievalDoc]:
    """Reproduce current CourseLangChain build.py page_content."""
    out: list[RetrievalDoc] = []
    for c in courses:
        text = f"課程名稱是{c.name}, 上課時間是{_time_str(c)}, 這堂課的老師是{c.teacher}"
        out.append(RetrievalDoc(course_id=c.course_id, text=text, metadata=_course_metadata(c)))
    return out


def build_d_obj(courses: list[Course]) -> list[RetrievalDoc]:
    """+ objective + classroom + unit + lang. No LLM dependency."""
    out: list[RetrievalDoc] = []
    for c in courses:
        objective = (c.objective or "").strip()
        text = (
            f"課名: {c.name}\n"
            f"教師: {c.teacher}\n"
            f"開課單位: {c.unit}\n"
            f"語言: {c.lang}　學分: {c.point}\n"
            f"上課時間: {_time_str(c)}\n"
            f"課程目標: {objective[:600]}"
        )
        out.append(RetrievalDoc(course_id=c.course_id, text=text, metadata=_course_metadata(c)))
    return out


def _load_meta(db_path: Path = META_DB) -> dict[str, dict]:
    if not db_path.exists():
        return {}
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT course_id, summary_100, keywords_json, topic_tags_json FROM course_meta_v1"
    ).fetchall()
    con.close()
    out: dict[str, dict] = {}
    for cid, summ, kw, tags in rows:
        out[cid] = {
            "summary": summ or "",
            "keywords": json.loads(kw) if kw else [],
            "topic_tags": json.loads(tags) if tags else [],
        }
    return out


def build_d_v2(courses: list[Course]) -> list[RetrievalDoc]:
    """D-Obj + LLM summary/keywords/tags. Falls back to D-Obj when meta missing."""
    meta = _load_meta()
    out: list[RetrievalDoc] = []
    for c in courses:
        m = meta.get(c.course_id, {})
        kws_raw = m.get("keywords") or []
        kw = "、".join(str(k) for k in kws_raw if isinstance(k, str)) if m else ""
        tags_raw = m.get("topic_tags") or []
        tags = "、".join(str(t) for t in tags_raw if isinstance(t, str)) if m else ""
        summary = m.get("summary") or ""
        objective = (c.objective or "").strip()
        topic_lines = "、".join(c.weekly_topics) if c.weekly_topics else ""
        text = (
            f"課名: {c.name}\n"
            f"教師: {c.teacher}\n"
            f"開課單位: {c.unit}　語言: {c.lang}　學分: {c.point}\n"
            f"上課時間: {_time_str(c)}\n"
            f"主題標籤: {tags}\n"
            f"關鍵字: {kw}\n"
            f"摘要: {summary}\n"
            f"週次主題: {topic_lines}\n"
            f"課程目標: {objective[:400]}"
        )
        meta_dict = _course_metadata(c)
        meta_dict["meta_v1_present"] = bool(m)
        out.append(RetrievalDoc(course_id=c.course_id, text=text, metadata=meta_dict))
    return out


# multi-field (D-V3) builder will arrive in W6
BUILDERS = {
    "d-base": build_d_base,
    "d-obj": build_d_obj,
    "d-v2": build_d_v2,
}
