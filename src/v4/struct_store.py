"""Runtime loader: everything the search needs from data/v4/course_struct.db alone.

The online system ships this one SQLite file. 1142.db (raw scrape, 311 MB, 28 terms
of reviews) is needed only at build time: scripts/v4/build_structured.py turns it into
course_struct.db, and scripts/v4/build_runtime.py adds the derived tables the search
reads at runtime (dept, aspect, embedding).

Objects are the same ones the pipeline already uses (CatalogCourse, Offering, Dept), so
src/v4/pipeline.py does not need to know where they came from. Field mapping is checked
course by course against src/v4/catalog.py in scripts/v4/analysis/check_struct_catalog.py.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.doc_builders import _course_metadata
from src.eval.conditions import course_categories
from src.loader.courses import Course
from src.loader.time_parser import Session
from src.v4.assessment import EXAM_ITEMS, FORMAT_KEYS, AssessmentProfile, profile_from
from src.v4.catalog import CatalogCourse, _norm_name
from src.v4.offering import Channel, Dept, Offering
from src.v4.semester import Semester, parse as parse_semester, struct_db

ROOT = Path(__file__).resolve().parents[2]
STRUCT_DB = struct_db()   # data/v4/course_struct.db unless V4_STRUCT_DB says otherwise
TRUSTED_ALIAS_SOURCES = ("rate", "seed")   # jieba-generated aliases were 27/33 correct (user: do not use)
_ITEM_NAMES = {k for k, _ in FORMAT_KEYS}
# grading["by_cat"] keys the rest of the code reads (src.v4.catalog._EVAL_CAT, src.v4.aspects)
_BY_CAT_KEY = {"期中考": "exam", "期末考": "exam", "小考": "exam", "考試": "exam", "書面報告": "report",
               "口頭報告": "presentation", "作業": "homework", "專題實作": "project",
               "出席參與": "participation", "分組": "group"}
# 期中 45%: a bare 期中／期末 label with a percentage is the exam (刑法(二))
_BARE_EXAM_LABEL = re.compile(r"(期中|期末)(成績)?")
_BARE_EXAM = {"期中": "期中考", "期末": "期末考"}
# 測驗40% / Examinations (40%): a graded exam even when the item parser missed it
_GRADED_EXAM = re.compile(r"(考試|測驗|[Ee]xam(ination)?s?)\W{0,3}\(?\s*\d{1,3}\s*[%％]")


@dataclass
class StructBundle:
    catalog: list[CatalogCourse]
    offerings: list[Offering]
    depts: dict[str, Dept]
    aliases: dict[str, tuple[str, ...]]            # alias -> canonical course names
    assess: dict[str, AssessmentProfile]
    aspects: dict[str, dict[str, float]] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)   # model -> (n, d) aligned with catalog
    meta: dict[str, str] = field(default_factory=dict)
    units: "UnitTable | None" = None               # names <-> codes for identity resolution


def _hour(hhmm: str, end: bool) -> int:
    h, m = (int(x) for x in hhmm.split(":"))
    return h + 1 if end and m > 0 else h   # 13:10 starts in hour 13, 16:00 ends at 16


def _lmt_kind(cat: str | None, detail: str | None) -> str:
    if not cat:
        return ""
    return f"跨領域({detail})" if cat == "跨領域" and detail else f"{cat}通識"


def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _grouped(con: sqlite3.Connection, sql: str) -> dict[str, list[sqlite3.Row]]:
    out: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in con.execute(sql):
        out[r[0]].append(r)
    return out


def _course(r: sqlite3.Row, sessions: list[sqlite3.Row], paths: list[sqlite3.Row], topics: list[str],
            sem: Semester) -> Course:
    sess = [Session(weekday=s["weekday"], start_hour=_hour(s["start"], False), end_hour=_hour(s["end"], True),
                    raw=s["periods"] or "") for s in sessions if s["source"] == "time"]
    return Course(
        course_id=r["course_id"], year=sem.year, semester=sem.term, sub_num=r["sub_num"] or "", name=r["name"] or "",
        name_en=r["name_en"] or "", teacher=r["teacher_raw"] or "", teacher_en="", kind=r["kind_code"] or 0,
        lmt_kind=_lmt_kind(r["ge_category"], r["ge_detail"]), core=r["ge_core"] or 0, lang=r["lang"] or "",
        point=float(r["credits"] or 0), classroom=r["classroom_raw"] or "", classroom_id=r["room_id"] or "",
        unit=r["unit_raw"] or "", cross_listed=[(p["dp1"], p["dp2"], p["dp3"]) for p in paths],
        time_raw=r["time_raw"] or "", sessions=sess, info=r["info_clean"] or "", note=r["note_clean"] or "",
        syllabus=r["syllabus"] or "", objective=r["objective"] or "", schedule=r["schedule_clean"] or "",
        evaluation=r["evaluation"] or "", textbook=r["textbook_clean"] or "", teaching_approach="",
        ai_policy=r["ai_policy"] or "", weekly_topics=topics)


def _catalog_course(r: sqlite3.Row, c: Course, evals: list[sqlite3.Row], restr: list[sqlite3.Row],
                    topics: list[str]) -> CatalogCourse:
    meta = _course_metadata(c)
    meta["dept_code"] = r["dept_code"] or ""
    items = [(e["label"], int(e["pct"])) for e in evals if e["pct"] is not None]
    by_cat: dict[str, int] = defaultdict(int)
    for e in evals:
        if e["pct"] is not None and e["category"] in _BY_CAT_KEY:
            by_cat[_BY_CAT_KEY[e["category"]]] += int(e["pct"])
    teaching = {k: r[f"teach_{k}"] for k in ("lecture", "discussion", "group", "elearning") if r[f"teach_{k}"] is not None}
    enroll = {}
    if r["enroll_limit"]:
        enroll = {"limit": r["enroll_limit"], "count": r["enroll_count"],
                  "fill": round((r["enroll_count"] or 0) / r["enroll_limit"], 3)}
    return CatalogCourse(
        course=c, meta=meta, cancelled=bool(r["cancelled"]), categories=course_categories(meta),
        dept_code=r["dept_code"] or "", topics=topics, schedule_clean=c.schedule, textbook_clean=c.textbook,
        teaching=teaching,
        grading={"items": items[:12], "by_cat": dict(by_cat), "pct_sum": r["eval_pct_sum"] or 0,
                 "reliable": bool(r["eval_reliable"])},
        restrictions=sorted({x["type"] for x in restr if x["method"] == "rule"}),
        family=(_norm_name(c.name), c.teacher), enrollment=enroll)


def _assessment(evals: list[sqlite3.Row], evaluation_text: str, reliable: bool) -> AssessmentProfile:
    """Union of the keyword scan over labels and text and the eval_item categories.

    eval_item gives each graded item one category, so a combined label such as
    Quizzes and assignments 20% or 平時成績（出席/作業） loses its second part; the text
    scan keeps it. eval_item adds the exam type where the text says only 考試."""
    labels = [e["label"] or "" for e in evals]
    scan = profile_from(labels, evaluation_text)
    cats = {e["category"] for e in evals}
    bare = {_BARE_EXAM[m.group(1)] for lbl in labels if (m := _BARE_EXAM_LABEL.fullmatch(lbl.strip()))}
    items = scan.items | (cats & _ITEM_NAMES) | bare
    if reliable:  # graded items are complete; only an untyped exam that is actually graded counts
        generic_mention = ("考試" in cats or profile_from(labels, "").generic_exam
                           or bool(_GRADED_EXAM.search(evaluation_text)))
    else:
        generic_mention = scan.generic_exam or "考試" in cats
    generic = generic_mention and not items & EXAM_ITEMS
    return AssessmentProfile(frozenset(items), generic)


def _targets(raw: str | None) -> frozenset:
    """'["學士3", "碩士1"]' -> {("學士", 3), ("碩士", 1)}, the form src.v4.offering.unit_targets returns."""
    out = set()
    for t in json.loads(raw or "[]"):
        m = re.fullmatch(r"(\D+)(\d+)", str(t))
        if m:
            out.add((m.group(1), int(m.group(2))))
    return frozenset(out)


def _offerings(con: sqlite3.Connection) -> list[Offering]:
    notes = dict(con.execute("SELECT course_id, note_clean FROM course"))
    chans = _grouped(con, "SELECT offering_id, * FROM channel ORDER BY offering_id, rowid")
    out = []
    for o in con.execute("SELECT * FROM offering ORDER BY offering_id"):
        rows = chans.get(o["offering_id"], [])
        channels = tuple(Channel(
            course_id=c["course_id"], sub_num=c["sub_num"], unit=c["unit"] or "", dept_code=c["dept_code"] or "",
            college=c["college"] or "", level=c["level"] or "",
            targets=_targets(c["targets"]),
            enroll_mode=c["enroll_mode"] or "", audience_level=c["audience_level"] or "",
            colleges_only=frozenset(json.loads(c["colleges_only"] or "[]")),
            status_only=frozenset(json.loads(c["status_only"] or "[]")),
            special_class=c["special_class"] or "", url=c["url"] or "",
            note=notes.get(c["course_id"]) or "") for c in rows)
        key = (o["name_norm"], o["teacher"], o["time_raw"] if o["time_raw"] else o["primary_course_id"])
        out.append(Offering(key=key, course_ids=tuple(c.course_id for c in channels), channels=channels))
    return out


def _aliases(con: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    q = (f"SELECT alias, members FROM alias WHERE source IN ({','.join('?' * len(TRUSTED_ALIAS_SOURCES))}) "
         "AND in_1142 = 1 ORDER BY confidence DESC")
    for alias, members in con.execute(q, TRUSTED_ALIAS_SOURCES):
        out.setdefault(alias, tuple(json.loads(members or "[]")))
    return out


def load_struct(db: Path = STRUCT_DB, include_cancelled: bool = False,
                embedding_models: tuple[str, ...] = ()) -> StructBundle:
    if not db.exists():
        raise FileNotFoundError(f"{db} missing: run scripts.v4.build_structured then scripts.v4.build_runtime")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    sessions = _grouped(con, "SELECT course_id, * FROM session ORDER BY course_id, seq")
    paths = _grouped(con, "SELECT course_id, * FROM course_path")
    evals = _grouped(con, "SELECT course_id, * FROM eval_item ORDER BY course_id, seq")
    restr = _grouped(con, "SELECT course_id, * FROM restriction")
    weekly = _grouped(con, "SELECT course_id, topic FROM weekly ORDER BY course_id, week")
    meta = dict(con.execute("SELECT key, value FROM meta"))
    sem = parse_semester(meta.get("semester", "114-2"))
    cat, assess = [], {}
    for r in con.execute("SELECT * FROM course ORDER BY course_id"):
        if r["cancelled"] and not include_cancelled:
            continue
        cid = r["course_id"]
        topics = [w["topic"] for w in weekly.get(cid, []) if w["topic"]][:24]
        c = _course(r, sessions.get(cid, []), paths.get(cid, []), topics, sem)
        cat.append(_catalog_course(r, c, evals.get(cid, []), restr.get(cid, []), topics))
        assess[cid] = _assessment(evals.get(cid, []), r["evaluation"] or "", bool(r["eval_reliable"]))
    tables = _tables(con)
    depts = ({code: Dept(code, college, level) for code, college, level in con.execute("SELECT * FROM dept")}
             if "dept" in tables else {})
    aspects: dict[str, dict[str, float]] = defaultdict(dict)
    if "aspect" in tables:
        for cid, aspect, value in con.execute("SELECT course_id, aspect, value FROM aspect"):
            aspects[cid][aspect] = value
    embs = {}
    if "embedding" in tables:
        pos = {c.course_id: i for i, c in enumerate(cat)}
        for model in embedding_models:
            rows = con.execute("SELECT course_id, dim, vec FROM embedding WHERE model = ?", (model,)).fetchall()
            if not rows:
                raise ValueError(f"no embeddings for {model} in {db}: run scripts.v4.build_runtime --encoder {model}")
            m = np.zeros((len(cat), rows[0][1]), dtype=np.float32)
            for cid, dim, vec in rows:
                if cid in pos:
                    m[pos[cid]] = np.frombuffer(vec, dtype=np.float32, count=dim)
            embs[model] = m
    offerings = [o for o in _offerings(con) if any(cid in assess for cid in o.course_ids)]
    aliases = _aliases(con)
    units = None
    if "unit" in tables:
        from src.v4.identity import UnitTable

        units = UnitTable.from_rows(con.execute("SELECT code, name, college, level, source FROM unit"))
    con.close()
    return StructBundle(cat, offerings, depts, aliases, assess, dict(aspects), embs, meta, units)


def expand_aliases(text: str, aliases: dict[str, tuple[str, ...]]) -> tuple[str, list[str]]:
    """'想修個經' -> ('想修個經 個體經濟學', ['個體經濟學']). Longest alias first; only whole
    aliases that are not already part of a written-out course name."""
    found: list[str] = []
    for a in sorted(aliases, key=len, reverse=True):
        if a in text and not any(a in n for n in found):
            found.extend(n for n in aliases[a] if n not in found and n not in text)
    return (text + " " + " ".join(found)).strip() if found else text, found

