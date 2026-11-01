"""Rule pass over one course code: raw COURSE rows -> Parsed (no LLM here)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.v4.offering import COLLEGES, LEVELS, Dept
from src.v4.structured import aipolicy, grading, sessions, weekly
from src.v4.structured.aliases import base_name, nk
from src.v4.structured.clauses import Change, Restriction, parse_info, parse_note
from src.v4.structured.scalars import (SPAN, channel_label, classroom, dept_name, general_ed, kind_label,
                                       teaching)
from src.v4.structured.targets import TargetClass, parse_unit
from src.v4.structured.text import clean, merge_fragments, nfkc, split_teachers, text_lang, textbook_clean


@dataclass(frozen=True)
class Raw:
    row: dict
    paths: tuple[tuple[str, str, str], ...]

    @property
    def course_id(self) -> str:
        return self.row["id"]


@dataclass(frozen=True)
class Parsed:
    course: dict
    paths: tuple[tuple[str, str, str], ...]
    sessions: tuple[sessions.Session, ...]
    targets: tuple[TargetClass, ...]
    restrictions: tuple[Restriction, ...]
    unmatched: tuple[str, ...]
    changes: tuple[Change, ...]
    grading: grading.Grading
    weeks: tuple[weekly.Week, ...]
    weekly_method: str
    ai: aipolicy.AiPolicy
    note: str | None
    extra: dict = field(default_factory=dict)

    @property
    def course_id(self) -> str:
        return self.course["course_id"]


def load_raw(con: sqlite3.Connection, y: str = "114", s: str = "2") -> list[Raw]:
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM COURSE WHERE y = ? AND s = ? ORDER BY id, dp1, dp2, dp3", (y, s)).fetchall()
    con.row_factory = None
    grouped: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        grouped.setdefault(r["id"], []).append(r)
    return [Raw(dict(g[0]), tuple((r["dp1"], r["dp2"], r["dp3"]) for r in g)) for g in grouped.values()]


def load_result(con: sqlite3.Connection, ys: str = "1142") -> dict[str, tuple[int, int]]:
    return {cid: (lim, cnt) for cid, lim, cnt in
            con.execute("SELECT courseId, studentLimit, studentCount FROM RESULT WHERE yearsem = ?", (ys,))}


def _org(raw: Raw, depts: dict[str, Dept], has_targets: bool) -> tuple[str, str | None, str | None, str | None]:
    """dept code, opening unit name, college, level. Units outside the college tree (通識中心,
    外文中心, 體育室, 學程) have no department name in unit.json; the unit string names them."""
    code = (raw.row["subNum"] or "")[:3]
    college_rows = [(d1, d2) for d1, d2, _ in raw.paths if d1 in COLLEGES]
    dept = depts.get(code)
    college = COLLEGES[college_rows[0][0]] if college_rows else (dept.college if dept else None)
    level = LEVELS.get(college_rows[0][1]) if college_rows else (dept.level if dept else None)
    name = dept_name(code, list(raw.paths))
    if name is None and not has_targets:
        name = nfkc(raw.row["unit"]) or None
    if name is None:
        name = next((lab for p in raw.paths if (lab := channel_label(p))), None)
    return code, name, (None if college == "X" else college), level


def _course_row(raw: Raw, depts: dict[str, Dept], result: dict, g: grading.Grading,
                ai: aipolicy.AiPolicy, n_weeks: int, ses: list, targets: list) -> dict:
    r = raw.row
    code, dname, college, level = _org(raw, depts, bool(targets))
    ge_cat, ge_detail, ge_core = general_ed(r["lmtKind"], r["core"])
    teach = teaching(r["teaching_approach"])
    building, room = classroom(r["classroom"])
    lim, cnt = result.get(raw.course_id, (None, None))
    syllabus = merge_fragments(nfkc(r["syllabus"])) or None
    objective = merge_fragments(nfkc(r["objective"])) or None
    return {
        "course_id": raw.course_id, "sub_num": r["subNum"], "course_group": (r["subNum"] or "")[:6],
        "name": nfkc(r["name"]), "name_en": nfkc(r["nameEn"]) or None, "name_base": base_name(r["name"]),
        "teachers": split_teachers(r["teacher"]), "teacher_raw": nfkc(r["teacher"]),
        "credits": float(r["point"]) if r["point"] is not None else None,
        "kind_code": r["kind"], "kind_label": kind_label(r["kind"]),
        "ge_category": ge_cat, "ge_detail": ge_detail, "ge_core": ge_core,
        "semester_span": SPAN.get(nfkc(r["smtQty"]), nfkc(r["smtQty"]) or None),
        "enroll_mode": nfkc(r["tranTpe"]) or None, "cancelled": "停開" in (r["info"] or ""),
        "lang": nfkc(r["lang"]) or None, "english_taught": nfkc(r["lang"]) == "英文",
        "syllabus_lang": text_lang(f"{syllabus or ''} {objective or ''}"),
        "dept_code": code, "dept_name": dname, "college": college, "level": level,
        "unit_raw": nfkc(r["unit"]) or None, "unit_is_class": bool(targets),
        "is_pe": any(d1 == "03" and d2 == "A6" for d1, d2, _ in raw.paths) or nfkc(r["unit"]) == "體育",
        "time_raw": nfkc(r["time"]) or None, "time_tba": not any(s.source == "time" for s in ses),
        "classroom_raw": nfkc(r["classroom"]) or None, "building": building, "room_id": room,
        "ai_policy": ai.category,
        "teach_lecture": teach.get("lecture"), "teach_discussion": teach.get("discussion"),
        "teach_group": teach.get("group"), "teach_elearning": teach.get("elearning"),
        "teach_other": teach.get("other"), "teach_other_label": teach.get("other_label"),
        "teach_sum": teach.get("sum"),
        "eval_pct_sum": g.pct_sum, "eval_reliable": g.reliable, "eval_method": g.method,
        "n_weeks": n_weeks, "enroll_limit": lim, "enroll_count": cnt,
        "url": r["teaSchmUrl"], "note_clean": clean(r["note"]), "info_clean": clean(r["info"]),
        "syllabus": syllabus, "objective": objective,
        "evaluation": merge_fragments(nfkc(r["evaluation"])) or None,
        "textbook_clean": textbook_clean(r["textbook"]),
    }


def parse_course(raw: Raw, depts: dict[str, Dept], result: dict) -> Parsed:
    r = raw.row
    note = clean(r["note"])
    main = sessions.parse_time_code(r["time"])
    ses = main + sessions.note_sessions(note, main)
    targets = parse_unit(nfkc(r["unit"]))
    restr, unmatched = parse_note(note)
    g = grading.parse(r["evaluation"])
    weeks = weekly.parse(r["schedule"])
    ai = aipolicy.parse(r["ai_policy"], r["evaluation"], r["syllabus"])
    row = _course_row(raw, depts, result, g, ai, len(weeks), ses, targets)
    row["schedule_clean"] = weekly.schedule_clean(r["schedule"], weeks)
    row["weekly_method"] = "rule" if weeks else "none"
    return Parsed(row, raw.paths, tuple(ses), tuple(targets), tuple(restr), tuple(unmatched),
                  tuple(parse_info(r["info"])), g, tuple(weeks), "rule" if weeks else "none", ai, note)


def path_rows(p: Parsed) -> list[dict]:
    return [{"course_id": p.course_id, "dp1": a, "dp2": b, "dp3": c, "label": channel_label((a, b, c))}
            for a, b, c in p.paths]
