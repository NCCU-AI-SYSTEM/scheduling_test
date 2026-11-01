"""Can this student enrol in this class? Judged per channel, then per offering.

Student identity comes from the student number alone: NCCU numbers are
<entry year (ROC, 3)><department code (3)><serial (3)>, and the department code is the
same code that starts course numbers (703 資科學士班, 753 資科碩士班), so it maps to a
college and degree level through the catalogue (offering.dept_registry). The number is
used only to derive these fields and is never stored.

Two levels, as agreed with the user (2026/10/19):
  exclude  the student cannot enrol through this channel at all
  warn     shown, with a note saying why the student may not get in
  info     shown as context
An offering is excluded only when every channel excludes the student.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.v4.offering import Channel, Dept, Offering

_SID = re.compile(r"^(\d{3})([0-9A-Z]{3})(\d{3})$")
_GRADE_MIN = re.compile(r"限(?:大)?([一二三四])(?:年級)?以上|限大([一二三四])")
_CN = {"一": 1, "二": 2, "三": 3, "四": 4}

# rule -> level. Domain call: which restrictions are absolute at NCCU.
SEVERITY = {
    "college_only": "exclude",    # 大學英文 X學院修讀
    "status_only": "exclude",     # 僑生／外國學生修讀, 僅供外籍生
    "special_class": "exclude",   # 國防／領導／健康中心專班, 不開放非原專班
    "audience_level": "exclude",  # 此科目代碼為碩士班選課用
    "degree": "exclude",          # 限碩士生 / 限學士班
    "grade_min": "exclude",       # 限大三以上
    "dept_only": "warn",          # 限本系: 加簽 can still let others in (user, 2026/10/19)
    "dept_or_minor": "warn",      # 限本系及輔雙生: a minor or double major may still enrol
    "assigned": "warn",           # 必修灌檔, student is not the assigned class: self-select, few seats
    "assigned_target": "info",    # 必修灌檔 and the student is the assigned class
    "level_target": "warn",       # unit is another degree level, no explicit limit
    "english_exempt": "exclude",  # 大英免修: the school does not assign 大學英文 to exempt students
    "allocated": "info",          # 分派
    "target": "info",             # unit is another department or year
}


@dataclass(frozen=True)
class StudentIdentity:
    dept_code: str
    college: str
    level: str
    year: int
    statuses: frozenset = frozenset()   # self-declared: 僑生, 外籍生, 交換生, 國防 ...
    other_depts: frozenset = frozenset()  # double major / minor department codes (全人 or form)
    english_exempt: bool = False          # 大英免修 (全人 collegeEnglishExemption or form)
    source: str = ""                      # 全人 / 自行輸入 / 學號, see src/v4/identity.py


def from_student_id(sid: str, depts: dict[str, Dept], academic_year: int = 114,
                    statuses: frozenset = frozenset()) -> StudentIdentity:
    m = _SID.match((sid or "").strip().upper())
    if not m:
        raise ValueError("學號格式應為 9 碼，例如 112703001")
    entry, code = int(m.group(1)), m.group(2)
    dept = depts.get(code)
    if dept is None:
        raise ValueError(f"找不到系所代碼 {code}")
    return StudentIdentity(dept_code=code, college=dept.college, level=dept.level,
                           year=max(1, academic_year - entry + 1), statuses=statuses)


@dataclass(frozen=True)
class Flag:
    level: str
    kind: str
    detail: str


def _flag(kind: str, detail: str) -> Flag:
    return Flag(SEVERITY[kind], kind, detail)


def _is_target(ch: Channel, s: StudentIdentity) -> bool:
    """The student belongs to the class this channel is opened for (大學英文 targets a college)."""
    if ch.targets:
        return s.dept_code == ch.dept_code and (s.level, s.year) in ch.targets
    return bool(ch.colleges_only) and s.college in ch.colleges_only


def check_channel(ch: Channel, s: StudentIdentity) -> tuple[Flag, ...]:
    out: list[Flag] = []
    note = ch.note
    if ch.colleges_only and s.college not in ch.colleges_only:
        out.append(_flag("college_only", f"限{'、'.join(sorted(ch.colleges_only))}學院學生"))
    if ch.status_only and not ch.status_only & s.statuses:
        out.append(_flag("status_only", f"限{'、'.join(sorted(ch.status_only))}"))
    if ch.special_class and ch.special_class not in s.statuses:
        out.append(_flag("special_class", f"{ch.special_class}專班課程"))
    if ch.audience_level and ch.audience_level != s.level:
        out.append(_flag("audience_level", f"此課號供{ch.audience_level}班選課"))
    if re.search(r"限碩|限博士生|限研究生", note) and s.level == "學士":
        out.append(_flag("degree", "限研究生"))
    elif re.search(r"限學士班|限大學部", note) and s.level != "學士":
        out.append(_flag("degree", "限學士班"))
    m = _GRADE_MIN.search(note)
    if m and s.level == "學士" and s.year < _CN[m.group(1) or m.group(2)]:
        out.append(_flag("grade_min", f"限大{m.group(1) or m.group(2)}以上"))
    minor_ok = bool(re.search(r"輔|雙", note))
    if "限本系" in note and s.dept_code != ch.dept_code and not (minor_ok and ch.dept_code in s.other_depts):
        out.append(_flag("dept_or_minor" if minor_ok else "dept_only",
                         "限本系" + ("及輔系、雙主修生" if minor_ok else "學生")))
    if ch.enroll_mode == "必修灌檔":
        if _is_target(ch, s):
            out.append(_flag("assigned_target", "必修灌檔：學校直接分發，不用自己選"))
        else:
            out.append(_flag("assigned", f"必修灌檔：先分發給{ch.unit or '開課對象'}，其他學生需自行選課，名額可能不多"))
    elif ch.enroll_mode == "分派":
        out.append(_flag("allocated", "由系所分派"))
    levels = {lv for lv, _ in ch.targets}
    if levels and s.level not in levels:
        out.append(_flag("level_target", f"開課對象為{'、'.join(sorted(levels))}班（{ch.unit}）"))
    elif ch.targets and (s.dept_code != ch.dept_code or (s.level, s.year) not in ch.targets) \
            and ch.dept_code not in s.other_depts:
        out.append(_flag("target", f"主要開給{ch.unit}"))
    return tuple(out)


@dataclass(frozen=True)
class Access:
    channel: Channel
    flags: tuple[Flag, ...]
    excluded: bool
    others: tuple[tuple[Channel, tuple[Flag, ...]], ...] = field(default=())


def _rank(item: tuple[Channel, tuple[Flag, ...]], s: StudentIdentity) -> tuple:
    ch, flags = item
    lv = [f.level for f in flags]
    return ("exclude" in lv, lv.count("warn"), lv.count("info"), ch.dept_code != s.dept_code)


def judge(off: Offering, s: StudentIdentity) -> Access:
    """Pick the channel that suits the student best; excluded only if all channels exclude."""
    extra: tuple[Flag, ...] = ()
    if s.english_exempt and off.key[0].startswith("大學英文"):
        extra = (_flag("english_exempt", "你已大英免修，不用修這門"),)
    items = sorted(((ch, check_channel(ch, s) + extra) for ch in off.channels), key=lambda it: _rank(it, s))
    best_ch, best_flags = items[0]
    return Access(channel=best_ch, flags=best_flags,
                  excluded=any(f.level == "exclude" for f in best_flags), others=tuple(items[1:]))
