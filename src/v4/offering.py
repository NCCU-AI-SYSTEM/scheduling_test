"""One class, several enrolment channels.

26% of 114-2 course codes share name, teacher and time with another code: the same
class opened to several audiences (186 groups are undergraduate + master's, e.g. 資科
703xxx for 資訊三資訊四 and 753xxx for 資碩). Search returns the class once; whether a
student can enrol depends on which channel (course code) fits them.

Channel facts, structured fields first:
  dp1 / dp2   college and degree level of the opening department (dept rows only)
  unit        target classes, 資訊三資訊四 -> {(學士, 3), (學士, 4)}
  tranTpe     enrolment mode: 一般 / 分派 / 上下轉檔 / 必修灌檔
  note        X學院修讀, 僑生／外國學生修讀, 專班, 此科目代碼為大學部選課用 ...
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.v4.catalog import DB, CatalogCourse

# dp1 of department rows -> college. dp1 01/02/03 are 通識 / 共同 / 體育 categories.
COLLEGES = {"100": "文", "200": "社會科學", "300": "商", "400": "傳播", "500": "外語", "600": "法",
            "700": "理", "800": "國際事務", "900": "教育", "Z23": "創新國際", "ZA0": "資訊", "ZC0": "X"}
LEVELS = {"1": "學士", "2": "碩士", "3": "博士"}
# longest first so 國際事務 wins over a bare 國際 and 社會科學 over 社會
_COLLEGE_RE = re.compile("|".join(sorted(set(COLLEGES.values()) - {"X"}, key=len, reverse=True)))
_YEAR_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}


@dataclass(frozen=True)
class Dept:
    code: str
    college: str
    level: str


@dataclass(frozen=True)
class Channel:
    course_id: str
    sub_num: str
    unit: str
    dept_code: str
    college: str            # college of the opening department, "" for 通識／共同
    level: str              # 學士 / 碩士 / 博士 / "" when the code is open to any level
    targets: frozenset      # {(level, year)} from the unit string
    enroll_mode: str        # tranTpe
    audience_level: str     # from 此科目代碼為大學部／碩士班選課用, else ""
    colleges_only: frozenset  # from X學院修讀
    status_only: frozenset  # 僑生 / 外籍生 / 交換生 required
    special_class: str      # name of a 專班 the channel is limited to, "" if none
    url: str
    note: str


@dataclass(frozen=True)
class Offering:
    key: tuple[str, str, str]   # (normalised name, teacher, time)
    course_ids: tuple[str, ...]
    channels: tuple[Channel, ...]

    @property
    def primary(self) -> str:
        return self.course_ids[0]


def norm_name(name: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", name or "")


def dept_registry(db: Path = DB) -> dict[str, Dept]:
    """Department code (subNum[:3]) -> college and level, by majority within the latest
    term the code appears in. Colleges were reorganised over time (資科 703 sat under
    理學院 before 資訊學院 existed), so votes from older terms must not outweigh it."""
    votes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    con = sqlite3.connect(db)
    for code, ys, dp1, dp2 in con.execute("SELECT substr(subNum,1,3), y||s, dp1, dp2 FROM COURSE"):
        if dp1 in COLLEGES and dp2 in LEVELS:
            votes[code][ys][(dp1, dp2)] += 1
    con.close()
    out = {}
    for code, by_term in votes.items():
        latest = max(by_term, key=lambda ys: (int(ys[:-1]), ys[-1]))
        (dp1, dp2), _ = by_term[latest].most_common(1)[0]
        out[code] = Dept(code, COLLEGES[dp1], LEVELS[dp2])
    return out


def unit_targets(unit: str) -> frozenset:
    """資訊三資訊四 -> {(學士,3),(學士,4)}; 資碩工一資碩計一 -> {(碩士,1)}; 外文中 -> {}."""
    out = set()
    for prefix, y in re.findall(r"([^一二三四五六七]*?)([一二三四五六七])", unit or ""):
        level = "博士" if "博" in prefix else "碩士" if "碩" in prefix else "學士"
        out.add((level, _YEAR_CN[y]))
    return frozenset(out)


def colleges_only(note: str) -> frozenset:
    """Colleges named in a 修讀 clause: 文、理、教育、資訊學院修讀 / 外國學生修讀（商學院）."""
    m = re.search(r"([^。；;]{0,40}?)學院修讀|修讀[（(]([^）)]{2,40})[）)]", note or "")
    if not m:
        return frozenset()
    return frozenset(_COLLEGE_RE.findall(m.group(1) or m.group(2)))


def status_only(note: str) -> frozenset:
    n = note or ""
    out = set()
    if re.search(r"僑生.{0,12}修讀|限.{0,4}僑生", n):
        out.add("僑生")
    if re.search(r"外國學生修讀|(僅供|限)外籍(交換生|學位生|生)", n):
        out.add("外籍生")
    if re.search(r"(僅供|限)(國際)?交換生", n):
        out.add("交換生")
    return frozenset(out)


def special_class(note: str) -> str:
    m = re.search(r"(國防|領導|健康中心)專班|不開放非原?專班", note or "")
    if not m:
        return ""
    return m.group(1) or "原專班"


def audience_level(note: str) -> str:
    m = re.search(r"此科目代碼為(大學部|學士班|碩士班|博士班)選課用", note or "")
    return {"大學部": "學士", "學士班": "學士", "碩士班": "碩士", "博士班": "博士"}.get(m.group(1), "") if m else ""


def _course_extra(year: str, term: str, db: Path = DB) -> dict[str, tuple[str, str]]:
    con = sqlite3.connect(db)
    rows = {cid: (t or "", u or "") for cid, t, u in
            con.execute("SELECT substr(id,1,13), tranTpe, teaSchmUrl FROM COURSE WHERE y = ? AND s = ?", (year, term))}
    con.close()
    return rows


def make_channel(c: CatalogCourse, depts: dict[str, Dept], extra: dict[str, tuple[str, str]]) -> Channel:
    x = c.course
    dept = depts.get(c.dept_code)
    dept_rows = [(d1, d2) for d1, d2, _ in x.cross_listed if d1 in COLLEGES]
    level = LEVELS.get(dept_rows[0][1], "") if dept_rows else ""
    mode, url = extra.get(c.course_id, ("", ""))
    return Channel(
        course_id=c.course_id, sub_num=x.sub_num, unit=x.unit, dept_code=c.dept_code,
        college=COLLEGES.get(dept_rows[0][0], "") if dept_rows else (dept.college if dept else ""),
        level=level, targets=unit_targets(x.unit), enroll_mode=mode,
        audience_level=audience_level(x.note), colleges_only=colleges_only(x.note),
        status_only=status_only(x.note), special_class=special_class(x.note), url=url, note=x.note or "")


def group_offerings(catalog: list[CatalogCourse], depts: dict[str, Dept] | None = None,
                    db: Path = DB) -> list[Offering]:
    depts = depts if depts is not None else dept_registry(db)
    first = catalog[0].course if catalog else None
    extra = _course_extra(first.year, first.semester, db) if first else {}
    groups: dict[tuple, list[CatalogCourse]] = defaultdict(list)
    for c in catalog:
        x = c.course
        # a course without a time slot is never merged: 未定 rows are unrelated classes
        key = (norm_name(x.name), x.teacher, x.time_raw) if x.time_raw else (norm_name(x.name), x.teacher, c.course_id)
        groups[key].append(c)
    return [Offering(key=k, course_ids=tuple(c.course_id for c in v),
                     channels=tuple(make_channel(c, depts, extra) for c in v)) for k, v in groups.items()]
