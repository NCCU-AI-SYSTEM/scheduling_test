"""Student profile from the 全人 transcript export, keeping only what course search needs.

Reads exactly: major / double major names, current year, the list of course codes
taken (with pass/fail, not scores), and this semester's enrolled codes. Personal
data sections (個人資料, contact info, ranks, averages, scores) are never read.
"""

from __future__ import annotations

import json
import re
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
    p = StudentProfile(major=me.get("registerMajor", ""), dept_abbr=abbr, degree=degree, year=year,
                       double_major=me.get("registerDoubleMajor", ""), current_semester=current_sem)
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
