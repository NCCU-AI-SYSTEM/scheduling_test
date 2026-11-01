"""Rule-based eligibility flags for (course, student profile).

Each flag has a level:
  block  the student cannot or should not take it (already passed, time clash with
         a course already enrolled, explicit grade minimum not met)
  warn   probably restricted, needs a human check (department-only text that does not
         mention the student's department, prerequisite not found in the transcript)
  info   worth showing (course is aimed at another class, first class mandatory)

Restriction text lives only in the free-text note (845 distinct phrasings in 114-2),
so regex covers the frequent patterns and anything else is passed through as text.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.v4.catalog import CatalogCourse
from src.v4.profile import StudentProfile

_GRADE_MIN = re.compile(r"限(?:大)?([一二三四])(?:年級)?以上|限大([一二三四])")
_CN = {"一": 1, "二": 2, "三": 3, "四": 4}
_DEPT_ONLY = re.compile(r"(?:僅限|只限|限)([^，。；;,\s]{1,20}?)(?:系|所|學院|學程)")
_PREREQ = re.compile(r"(?:先修(?:科目)?(?:為|：|:)?|擋修|須修過)\s*([^，。；;]{2,40})")


@dataclass
class Flag:
    level: str
    kind: str
    detail: str


def _n(name: str) -> str:
    """NFKC + no spaces, so 初級會計學（二） and 初級會計學(二) compare equal."""
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", name or ""))


def _sessions_overlap(a: list[dict], b: list[dict]) -> bool:
    return any(x["weekday"] == y["weekday"] and x["start"] < y["end"] and y["start"] < x["end"]
               for x in a for y in b)


def check(c: CatalogCourse, p: StudentProfile, current: list[CatalogCourse] | None = None) -> list[Flag]:
    flags: list[Flag] = []
    note = c.course.note or ""
    sub = c.course.sub_num or ""
    taken = {_n(t) for t in p.taken_names}
    if sub in p.taken or _n(c.course.name) in taken:
        flags.append(Flag("block", "already_taken", c.course.name))
    for cur in current or []:
        if cur.course_id != c.course_id and _sessions_overlap(c.meta["sessions"], cur.meta["sessions"]):
            flags.append(Flag("block", "time_conflict", cur.course.name))
    m = _GRADE_MIN.search(note)
    if m and p.year:
        need = _CN[m.group(1) or m.group(2)]
        if p.degree == "學士" and p.year < need:
            flags.append(Flag("block", "grade_min", f"限大{m.group(1) or m.group(2)}以上，你是{p.year}年級"))
    if re.search(r"限碩|限博士生", note) and p.degree == "學士":
        flags.append(Flag("block", "degree", "限研究生"))
    for dm in _DEPT_ONLY.finditer(note):
        target = dm.group(1)
        mine = {p.dept_abbr, p.major.replace("學系", "").replace("系", "")}
        if p.double_major:
            mine.add(re.sub(r"[（(].*", "", p.double_major))
        if not any(t and (t in target or target in t) for t in mine):
            flags.append(Flag("warn", "dept_only", dm.group(0)))
    pm = _PREREQ.search(note)
    if pm:
        names = [n.strip("（）() ") for n in re.split(r"[、及與和或/]", pm.group(1)) if len(n.strip()) >= 2]
        if names and not all(any(_n(n) in t or t in _n(n) for t in taken) for n in names):
            flags.append(Flag("warn", "prerequisite", pm.group(0)))
    if "first_class_mandatory" in c.restrictions:
        flags.append(Flag("info", "first_class", "第一堂課必須出席"))
    return flags


def blocked(flags: list[Flag]) -> bool:
    return any(f.level == "block" for f in flags)
