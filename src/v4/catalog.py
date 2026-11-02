"""Normalized 114-2 course catalogue for the v4 pipeline.

Builds on src.loader (dedupe, time parsing) and adds what docs/v4/03_data_analysis.md
found missing:
  - cancelled courses (info contains 停開; 140 in 114-2) are flagged and excluded
  - categories from the real kind codes (src.eval.conditions.course_categories)
  - department code = subNum[:3]
  - schedule with the two fixed boilerplate blocks stripped; weekly topics taken from
    every table cell that reads like a topic (the old extractor found none for 76.5%)
  - teaching approach and grading breakdowns as numbers
  - restriction flags from note text (regex detection only; values need an LLM)
  - review history: past offerings by subNum or (name, teacher) -> RATE comments
  - enrollment pressure from RESULT (1142 rows cover every course)
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.doc_builders import _course_metadata
from src.eval.conditions import course_categories
from src.loader import Course, load_courses
from src.v4.semester import current, source_db

ROOT = Path(__file__).resolve().parents[2]
DB = source_db()   # data/1142.db unless V4_SOURCE_DB says otherwise

_SCHED_HEAD = re.compile(r"^每周課程進度與作業要求\|Course Schedule & Requirements\|")
_TEXTBOOK_TAIL = re.compile(r"已申請之圖書館指定參考書目.*?Respect Copyright\.?", re.S)
_HEADER_CELLS = {"週次", "Week", "課程主題", "Topic", "課程內容與指定閱讀", "Content and Reading Assignment",
                 "教學活動與作業", "Teaching Activities and Homework", "學習投入時間", "Student workload expectation",
                 "課堂講授", "In-class Hours", "課程前後", "Outside-of-class Hours", "小時", "hours", "Hours"}
_DATE_OR_NUM = re.compile(r"^(W?\d{1,2}|第?\d{1,2}週?|\d{1,2}/\d{1,2}|\d+(\.\d+)?|[（(]?\d{1,2}/\d{1,2}[）)]?|周（|）)$")
_HAN = re.compile(r"[一-鿿]")

_TEACH = re.compile(r"(\d+)?%\|(講述|討論|小組活動|數位學習|其他：?)")
_TEACH_KEY = {"講述": "lecture", "討論": "discussion", "小組活動": "group", "數位學習": "elearning"}
_EVAL_ITEM = re.compile(r"([一-鿿A-Za-z][一-鿿A-Za-z /&()（）\-]{0,24}?)[\s:：|]*?(\d{1,3})\s*[%％]")
_EVAL_CAT = [
    ("exam", r"考|exam|test|quiz|測驗"),
    ("report", r"報告|report|paper|essay|論文|寫作|writing"),
    ("presentation", r"簡報|發表|presentation|口頭"),
    ("homework", r"作業|homework|assignment|習題|練習"),
    ("project", r"專題|project|期末作品|實作"),
    ("participation", r"出席|參與|participation|attendance|課堂表現|點名|討論"),
    ("group", r"分組|小組|group|team"),
]
_RESTRICT = {
    "grade_min": r"限大[一二三四]|[一二三四]年級以上|限碩|限博士生|限學士班",
    "prerequisite": r"先修|擋修|預修|須修過",
    "dept_only": r"(僅限|只限|限).{0,20}(系|所|學院|學程|ETP)",
    "priority": r"優先",
    "no_repeat": r"不得重複|請勿重複",
    "not_for_graduation": r"不(計入|列入|採計).{0,8}畢業學分",
    "first_class_mandatory": r"第一(堂|次|週).{0,8}(務必|必須)",
    "english_taught_note": r"英語授課|英文授課|全英|EMI",
    "add_permit": r"加簽",
    "exchange_only": r"交換生|外籍(學位)?生",
}


def _norm_name(name: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", name or "")


def weekly_topics(schedule: str, limit: int = 24) -> list[str]:
    cells = [c.strip() for c in _SCHED_HEAD.sub("", schedule or "").split("|")]
    out: list[str] = []
    for c in cells:
        if not c or c in _HEADER_CELLS or _DATE_OR_NUM.match(c):
            continue
        han = len(_HAN.findall(c))
        if (han >= 3 or len(re.findall(r"[A-Za-z]{3,}", c)) >= 2) and len(c) <= 80 and c not in out:
            out.append(c)
        if len(out) >= limit:
            break
    return out


def teaching(text: str) -> dict[str, int]:
    out = {}
    for pct, label in _TEACH.findall(text or ""):
        key = _TEACH_KEY.get(label.rstrip("："))
        if key:
            out[key] = int(pct) if pct else 0
    return out


def grading(text: str) -> dict:
    items = [(lbl.strip(" |:："), int(p)) for lbl, p in _EVAL_ITEM.findall(text or "") if 0 < int(p) <= 100]
    by_cat: dict[str, int] = defaultdict(int)
    for lbl, p in items:
        for cat, pat in _EVAL_CAT:
            if re.search(pat, lbl, re.I):
                by_cat[cat] += p
                break
    total = sum(p for _, p in items)
    return {"items": items[:12], "by_cat": dict(by_cat), "pct_sum": total, "reliable": 90 <= total <= 110}


@dataclass
class CatalogCourse:
    course: Course
    meta: dict
    cancelled: bool
    categories: set[str]
    dept_code: str
    topics: list[str]
    schedule_clean: str
    textbook_clean: str
    teaching: dict[str, int]
    grading: dict
    restrictions: list[str]
    family: tuple[str, str]
    past_ids: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)
    enrollment: dict = field(default_factory=dict)

    @property
    def course_id(self) -> str:
        return self.course.course_id


def load_catalog(year: str | None = None, semester: str | None = None, with_reviews: bool = True,
                 include_cancelled: bool = False, db: Path | None = None) -> list[CatalogCourse]:
    """Defaults: the semester and raw DB from src.v4.semester (V4_SEMESTER, V4_SOURCE_DB)."""
    sem = current()
    year, semester, db = year or sem.year, semester or sem.term, db or DB
    con = sqlite3.connect(db)
    out: list[CatalogCourse] = []
    for c in load_courses(db_path=db, year=year, semester=semester):
        meta = _course_metadata(c)
        meta["dept_code"] = (c.sub_num or "")[:3]
        restr = [k for k, pat in _RESTRICT.items() if re.search(pat, c.note or "")]
        out.append(CatalogCourse(
            course=c, meta=meta, cancelled="停開" in (c.info or ""),
            categories=course_categories(meta), dept_code=meta["dept_code"],
            topics=weekly_topics(c.schedule), schedule_clean=_SCHED_HEAD.sub("", c.schedule or ""),
            textbook_clean=_TEXTBOOK_TAIL.sub("", c.textbook or "").strip(),
            teaching=teaching(c.teaching_approach), grading=grading(c.evaluation),
            restrictions=restr, family=(_norm_name(c.name), c.teacher)))
    if with_reviews:
        _attach_history(con, out, f"{year}{semester}")
    _attach_enrollment(con, out, f"{year}{semester}")
    con.close()
    return out if include_cancelled else [x for x in out if not x.cancelled]


def _attach_history(con: sqlite3.Connection, cat: list[CatalogCourse], ys: str) -> None:
    by_sub: dict[str, set[str]] = defaultdict(set)
    by_fam: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cid, sub, name, teacher in con.execute(
            "SELECT DISTINCT substr(id,1,13), subNum, name, teacher FROM COURSE WHERE y||s <> ?", (ys,)):
        by_sub[sub].add(cid)
        by_fam[(_norm_name(name), teacher)].add(cid)
    rate: dict[str, list[str]] = defaultdict(list)
    for cid, text in con.execute("SELECT courseId, content FROM RATE"):
        t = (text or "").strip()
        if len(t) > 4 and t != "無":
            rate[cid[:13]].append(t)
    for x in cat:
        past = by_sub.get(x.course.sub_num, set()) | by_fam.get(x.family, set())
        x.past_ids = sorted(past)
        x.reviews = [t for p in x.past_ids for t in rate.get(p, [])]


def _attach_enrollment(con: sqlite3.Connection, cat: list[CatalogCourse], ys: str) -> None:
    res = {cid: (lim, cnt) for cid, lim, cnt in con.execute(
        "SELECT courseId, studentLimit, studentCount FROM RESULT WHERE yearsem = ?", (ys,))}
    for x in cat:
        lim, cnt = res.get(x.course_id, (None, None))
        if lim:
            x.enrollment = {"limit": lim, "count": cnt, "fill": round(cnt / lim, 3)}
