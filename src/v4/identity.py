"""Who is asking: one StudentIdentity from three sources, best first (user, 2026/11/02).

  1. 全人 export   current major and grade, double major / minor, 身分別, 大英免修, transcript
  2. form input    department (code or name), level, year, self-declared statuses
  3. student number  entry year + department code only; wrong for students who changed
                     department, blind to double majors and statuses, so a last resort

Department names become codes through the course system's own unit tree
(data/v4/unit.json: college -> 學士/碩士/博士 -> code + name), so 資訊科學系 at 碩士 level is
753 while the same name at 學士 level is 703. data/v4/nccu_units.tsv (the school's
administrative unit list) supplies names for codes the tree lacks.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from src.v4.access import StudentIdentity, from_student_id
from src.v4.offering import Dept
from src.v4.profile import StudentProfile

ROOT = Path(__file__).resolve().parents[2]
UNIT_TREE = ROOT / "data" / "v4" / "unit.json"
UNIT_LIST = ROOT / "data" / "v4" / "nccu_units.tsv"
_LEVEL = {"1": "學士", "2": "碩士", "3": "博士"}
_COLLEGE = {"文學院": "文", "社會科學學院": "社會科學", "商學院": "商", "傳播學院": "傳播", "外國語文學院": "外語",
            "法學院": "法", "理學院": "理", "國際事務學院": "國際事務", "教育學院": "教育", "創新國際學院": "創新國際",
            "資訊學院": "資訊", "X實驗學院": "X"}
_LEVEL_SUFFIX = re.compile(r"(學士班|碩士班|博士班)$")


@dataclass(frozen=True)
class Unit:
    code: str
    name: str
    college: str
    level: str


def _n(s: str) -> str:
    return re.sub(r"\s", "", unicodedata.normalize("NFKC", s or ""))


class UnitTable:
    def __init__(self, units: list[Unit], names: dict[str, str]):
        self.by_code = {}
        for u in units:
            self.by_code.setdefault(u.code, u)
        self.by_name = {(_n(u.name), u.level): u for u in units}
        self.names = names

    @classmethod
    def load(cls, tree: Path = UNIT_TREE, listing: Path = UNIT_LIST) -> "UnitTable":
        units = []
        for l1 in json.loads(tree.read_text(encoding="utf-8")):
            college = _COLLEGE.get(l1["utL1Text"].split()[0], "")
            for l2 in l1.get("utL2", []):
                level = _LEVEL.get(l2["utCodL2"])
                if not level:
                    continue
                for l3 in l2.get("utL3", []):
                    if l3["utCodL3"] != "0":
                        units.append(Unit(l3["utCodL3"], l3["utL3Text"].split(" / ")[0].strip(), college, level))
        names = {}
        if listing.exists():
            with listing.open(encoding="utf-8") as f:
                names = {r["code"]: r["name"] for r in csv.DictReader(f, delimiter="\t")}
        return cls(units, names)

    def code_for(self, name: str, level: str) -> str | None:
        """資訊科學系 + 碩士 -> 753; 資訊科學系碩士班 -> 753; a 3-character code passes through."""
        s = _n(name)
        if re.fullmatch(r"[0-9A-Z]{3}", s):
            return s
        m = re.search(r"[（(]([0-9A-Z]{3})[)）]", s)       # 全人 doubleMajor carries the code
        if m:
            return m.group(1)
        base = _LEVEL_SUFFIX.sub("", s)
        for key in ((s, level), (base, level)):
            if key in self.by_name:
                return self.by_name[key].code
        hits = {u.code for (nm, lv), u in self.by_name.items() if lv == level and (base in nm or nm in base)}
        return hits.pop() if len(hits) == 1 else None

    def dept(self, code: str, level: str | None = None) -> Dept | None:
        u = self.by_code.get(code)
        if u is None:
            return None
        if level and (_n(u.name), level) in self.by_name:
            u = self.by_name[(_n(u.name), level)]
        return Dept(u.code, u.college, u.level)


def from_profile(p: StudentProfile, units: UnitTable) -> StudentIdentity:
    code = units.code_for(p.major, p.degree)
    if code is None:
        raise ValueError(f"全人資料的主修 {p.major!r} 對不到系所代碼")
    d = units.dept(code, p.degree)
    others = {c for c in (units.code_for(n, "學士") for n in p.other_majors) if c}
    return StudentIdentity(dept_code=code, college=d.college if d else "", level=p.degree, year=p.year,
                           statuses=p.statuses, other_depts=frozenset(others), english_exempt=p.english_exempt,
                           source="全人")


@dataclass(frozen=True)
class FormInput:
    dept: str                      # code (703) or name (資訊科學系)
    level: str = "學士"
    year: int = 1
    statuses: tuple[str, ...] = ()
    other_depts: tuple[str, ...] = ()   # double major / minor, code or name
    english_exempt: bool = False


def from_form(f: FormInput, units: UnitTable) -> StudentIdentity:
    if f.level not in ("學士", "碩士", "博士"):
        raise ValueError("學制應為 學士、碩士 或 博士")
    if not 1 <= f.year <= 7:
        raise ValueError("年級應為 1 到 7")
    code = units.code_for(f.dept, f.level)
    d = units.dept(code, f.level) if code else None
    if d is None:
        raise ValueError(f"找不到系所 {f.dept!r}")
    others = {c for c in (units.code_for(x, "學士") for x in f.other_depts) if c}
    return StudentIdentity(dept_code=d.code, college=d.college, level=f.level, year=f.year,
                           statuses=frozenset(f.statuses), other_depts=frozenset(others),
                           english_exempt=f.english_exempt, source="自行輸入")


def resolve(units: UnitTable, depts: dict[str, Dept], profile: StudentProfile | None = None,
            form: FormInput | None = None, student_id: str | None = None) -> StudentIdentity | None:
    """First source that works wins; self-declared statuses from the form are added to any source."""
    ident = None
    if profile is not None:
        ident = from_profile(profile, units)
    elif form is not None:
        ident = from_form(form, units)
    elif student_id:
        ident = replace(from_student_id(student_id, depts), source="學號")
    if ident is not None and form is not None and ident.source != "自行輸入":
        ident = replace(ident, statuses=ident.statuses | frozenset(form.statuses))
    return ident
