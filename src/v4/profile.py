"""Student profile from the 全人 transcript export, keeping only what course search needs.

Reads exactly: major / double major / minor names, current year, 身分別 reduced to the
statuses course notes restrict on, the 大英免修 flag, the list of course codes taken
(with pass/fail, not scores), and this semester's enrolled codes. Personal data
sections (個人資料, contact info, ranks, averages, scores, student number) are never read.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_YEAR = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}


@dataclass
class StudentProfile:
    major: str = ""                # e.g. 資訊科學系
    dept_abbr: str = ""            # e.g. 資科
    double_major: str = ""
    degree: str = "學士"
    year: int = 0
    taken: set[str] = field(default_factory=set)        # passed subNum (9 digits)
    taken_names: set[str] = field(default_factory=set)
    current: set[str] = field(default_factory=set)      # subNum enrolled this semester
    current_semester: str = ""
    other_majors: tuple[str, ...] = ()                   # double major / minor names (may carry a code)
    statuses: frozenset = frozenset()                    # 僑生 / 外籍生 / 陸生 / 交換生 from 身分別
    english_exempt: bool = False                         # 大英免修


def _n(s: str) -> str:
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", s or ""))


def parse_statuses(student_identity: str) -> frozenset:
    """全人 身分別 text -> the statuses course notes restrict on. 港澳 counts as 僑生 (notes say 僑生（含港澳生）)."""
    s = _n(student_identity)
    out = set()
    if "僑" in s or "港澳" in s:
        out.add("僑生")
    if "外籍" in s or "外國" in s:
        out.add("外籍生")
    if "陸" in s:
        out.add("陸生")
    if "交換" in s:
        out.add("交換生")
    return frozenset(out)


def parse_exempt(text: str) -> bool:
    """大英免修 field: 免修 -> True; 未免修 / 不免修 / 否 / empty -> False."""
    s = _n(text)
    return "免" in s and not re.search(r"[未不非無否]", s)


def _grade_token(token: str) -> tuple[str, str, int]:
    """'資訊三' -> ('資訊', '學士', 3); '資科碩一' -> ('資科', '碩士', 1)."""
    m = re.match(r"(.+?)(碩|博)?([一二三四五六七])$", token or "")
    if not m:
        return token, "學士", 0
    degree = {"碩": "碩士", "博": "博士"}.get(m.group(2) or "", "學士")
    return m.group(1), degree, _YEAR[m.group(3)]


def load_profile(path: Path) -> StudentProfile:
    data = json.loads(path.read_text(encoding="utf-8"))
    study = next(x["課業學習"] for x in data if "課業學習" in x)
    me = study.get("aboutMe", {})
    abbr, degree, year = _grade_token(me.get("departmentProgramGrade", ""))
    hist = sorted(study.get("enrollmentHistoryList", []), key=lambda h: h.get("academicYearSemester", ""))
    current_sem = hist[-1]["academicYearSemester"].replace("-", "") if hist else ""
    others = tuple(x for x in (me.get("doubleMajor") or me.get("registerDoubleMajor"),
                               me.get("minor1"), me.get("minor2"), me.get("registerMinor")) if x)
    p = StudentProfile(major=me.get("registerMajor", ""), dept_abbr=abbr, degree=degree, year=year,
                       double_major=me.get("registerDoubleMajor", ""), current_semester=current_sem,
                       other_majors=others, statuses=parse_statuses(me.get("studentIdentity", "")),
                       english_exempt=parse_exempt(me.get("collegeEnglishExemption", "")))
    for group in study.get("gradeRecordList", []):
        for g in group.get("GradeRecords", []):
            code, sem = g.get("courseCode", ""), str(g.get("academicYearSemester", ""))
            score = str(g.get("score", ""))
            if sem == current_sem:
                p.current.add(code)
            elif score == "通過" or (score.replace(".", "").isdigit() and float(score) >= 60):
                p.taken.add(code)
                p.taken_names.add(g.get("courseName", ""))
    return p
