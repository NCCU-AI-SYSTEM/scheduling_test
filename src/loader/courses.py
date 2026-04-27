"""Course loader: read SQLite -> dedupe -> clean -> Course dataclass.

Dedupe rule: a logical course = unique courseId. Multiple rows differing only in
(dp1, dp2, dp3) represent cross-listing across departments; we collapse these into
`cross_listed: list[(dp1, dp2, dp3)]` and keep all other fields from the first row.

Cleaning:
- NFKC normalise text fields
- Strip HTML entities and `@異動資訊:` / `@備註:` prefixes when present
- Split syllabus weekly schedule rows (heuristic: lines containing ``\\d+/\\d+``)
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .time_parser import Session, parse_time_str

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "1142.db"

# Fields to NFKC + strip HTML entities
TEXT_FIELDS = (
    "name",
    "nameEn",
    "teacher",
    "teacherEn",
    "classroom",
    "unit",
    "info",
    "note",
    "syllabus",
    "objective",
    "schedule",
    "evaluation",
    "textbook",
    "teaching_approach",
    "ai_policy",
)


@dataclass(slots=True)
class Course:
    course_id: str  # e.g. 1142000348021
    year: str
    semester: str
    sub_num: str
    name: str
    name_en: str
    teacher: str
    teacher_en: str
    kind: int  # 必修=1 選修=2 通識=3 體育=4 其他=0
    lmt_kind: str
    core: int
    lang: str
    point: float
    classroom: str
    classroom_id: str
    unit: str
    cross_listed: list[tuple[str, str, str]]  # (dp1, dp2, dp3)
    time_raw: str
    sessions: list[Session]
    info: str
    note: str
    syllabus: str
    objective: str
    schedule: str
    evaluation: str
    textbook: str
    teaching_approach: str
    ai_policy: str
    weekly_topics: list[str] = field(default_factory=list)


_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_INFO_PREFIX = re.compile(r"^[＠@](異動資訊|備註|Information|Note)\s*[:：]\s*", re.MULTILINE)
_WEEK_LINE = re.compile(r"\d+/\d+")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _HTML_ENTITY.sub("", s)
    s = _INFO_PREFIX.sub("", s)
    return s.strip()


def _extract_weekly_topics(syllabus: str, schedule: str) -> list[str]:
    """Pull weekly schedule rows. NCCU stores schedule as table-pipe text:
    'Week|Date|Topic|...|1|2/27|...|2|3/6|Course Overview|...'
    We extract entries where a date pattern appears.
    """
    src = schedule or syllabus
    if not src:
        return []
    parts = re.split(r"\|", src)
    topics: list[str] = []
    for i, p in enumerate(parts):
        if _WEEK_LINE.fullmatch(p.strip()):
            # Next non-empty cell is usually the topic
            j = i + 1
            while j < len(parts) and not parts[j].strip():
                j += 1
            if j < len(parts):
                topic = parts[j].strip()
                if topic and len(topic) <= 80:
                    topics.append(topic)
    # dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_courses(
    db_path: Path | str = DEFAULT_DB,
    year: str = "114",
    semester: str = "2",
) -> list[Course]:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM COURSE WHERE y = ? AND s = ?",
        (year, semester),
    ).fetchall()
    con.close()

    grouped: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        grouped.setdefault(r["id"], []).append(r)

    out: list[Course] = []
    for cid, group in grouped.items():
        primary = group[0]
        cleaned: dict[str, str] = {f: _clean(primary[f]) for f in TEXT_FIELDS}
        sessions = parse_time_str(primary["time"] or "")
        weekly = _extract_weekly_topics(cleaned["syllabus"], cleaned["schedule"])

        out.append(
            Course(
                course_id=cid,
                year=primary["y"],
                semester=primary["s"],
                sub_num=primary["subNum"] or "",
                name=cleaned["name"],
                name_en=cleaned["nameEn"],
                teacher=cleaned["teacher"],
                teacher_en=cleaned["teacherEn"],
                kind=primary["kind"] or 0,
                lmt_kind=primary["lmtKind"] or "",
                core=primary["core"] or 0,
                lang=primary["lang"] or "",
                point=float(primary["point"]) if primary["point"] is not None else 0.0,
                classroom=cleaned["classroom"],
                classroom_id=primary["classroomId"] or "",
                unit=cleaned["unit"],
                cross_listed=[(g["dp1"], g["dp2"], g["dp3"]) for g in group],
                time_raw=primary["time"] or "",
                sessions=sessions,
                info=cleaned["info"],
                note=cleaned["note"],
                syllabus=cleaned["syllabus"],
                objective=cleaned["objective"],
                schedule=cleaned["schedule"],
                evaluation=cleaned["evaluation"],
                textbook=cleaned["textbook"],
                teaching_approach=cleaned["teaching_approach"],
                ai_policy=cleaned["ai_policy"],
                weekly_topics=weekly,
            )
        )
    return out


def courses_to_records(courses: Iterable[Course]) -> list[dict]:
    """Flatten Course list to JSON-serialisable dicts (for pandas / persistence)."""
    out: list[dict] = []
    for c in courses:
        out.append(
            {
                "course_id": c.course_id,
                "name": c.name,
                "teacher": c.teacher,
                "kind": c.kind,
                "lmt_kind": c.lmt_kind,
                "lang": c.lang,
                "point": c.point,
                "unit": c.unit,
                "classroom": c.classroom,
                "n_cross_listed": len(c.cross_listed),
                "time_raw": c.time_raw,
                "sessions": [
                    {"weekday": s.weekday, "start": s.start_hour, "end": s.end_hour}
                    for s in c.sessions
                ],
                "objective_len": len(c.objective),
                "syllabus_len": len(c.syllabus),
                "schedule_len": len(c.schedule),
                "n_weekly_topics": len(c.weekly_topics),
                "weekly_topics": c.weekly_topics,
                "objective": c.objective,
                "syllabus": c.syllabus,
                "schedule": c.schedule,
                "evaluation": c.evaluation,
                "textbook": c.textbook,
                "teaching_approach": c.teaching_approach,
                "ai_policy": c.ai_policy,
                "info": c.info,
                "note": c.note,
            }
        )
    return out
