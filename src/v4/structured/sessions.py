"""Weekly sessions with minute precision, from the time code and from notes.

Period table (docs/v4/03_data_analysis.md 1.2, checked against timeEn and the 幼教所
114-2 timetable): each period is 50 minutes, starting at hh:10 and ending at (hh+1):00.
Order A B 1 2 3 4 C D 5 6 7 8 E F G H, so 三23456 is two blocks (4 and 5 are not adjacent).

Notes add lab and discussion sessions that the time code lacks (49 courses), e.g.
實習課為四、78 / 五4為實習課 / 四E節為TA討論課 / 實習課時間為星期一17:10~19:00.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

PERIODS = "AB1234CD5678EFGH"
WEEKDAYS = "一二三四五六日"
_START_HOUR = {p: 6 + i for i, p in enumerate(PERIODS)}


@dataclass(frozen=True)
class Session:
    weekday: int          # 1 = Monday ... 7 = Sunday
    start: str            # 'HH:MM'
    end: str              # 'HH:MM'
    periods: str          # e.g. 'D56'
    kind: str = "main"    # main / lab / discussion
    source: str = "time"  # time / note
    tentative: bool = False
    in_time_field: bool = True   # a note session already covered by the time code


def _blocks(periods: str) -> list[str]:
    """Split a period string into contiguous blocks in PERIODS order."""
    idx = sorted({PERIODS.index(p) for p in periods if p in PERIODS})
    out: list[list[int]] = []
    for i in idx:
        if out and i == out[-1][-1] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return ["".join(PERIODS[i] for i in b) for b in out]


def _span(block: str) -> tuple[str, str]:
    return f"{_START_HOUR[block[0]]:02d}:10", f"{_START_HOUR[block[-1]] + 1:02d}:00"


def parse_time_code(code: str | None, **kw) -> list[Session]:
    """'一23四56' -> two sessions. '未定或彈性' and empty -> []."""
    out: list[Session] = []
    for day, periods in re.findall(rf"([{WEEKDAYS}])([{PERIODS}]+)", code or ""):
        for block in _blocks(periods):
            start, end = _span(block)
            out.append(Session(WEEKDAYS.index(day) + 1, start, end, block, **kw))
    return out


def is_tba(code: str | None) -> bool:
    return not parse_time_code(code)


# --- sessions written in notes -------------------------------------------------------
_P = f"[{PERIODS}]"
_LAB = r"(?:TA)?(實習課|實驗課|討論課)"
_DAYP = rf"([{WEEKDAYS}])\s*[、,]?\s*第?({_P}(?:[、,]?{_P})*)(?![\d:])節?"
_NOTE_PATTERNS = [
    # 實習課為四、78 / 實習課：一C / 實習課為三、C及五、78 / 另有TA討論課五CD
    re.compile(rf"{_LAB}(?:時間)?(?:暫定)?(?:為|:)?\s*((?:(?:週|星期)?[{WEEKDAYS}]\s*[、,]?\s*第?{_P}+(?![\d:])節?(?:\s*[、,及和]\s*)?)+)"),
    # 五4為實習課 / 四E節為TA討論課 / 三EF實習課
    re.compile(rf"((?:[{WEEKDAYS}]\s*第?{_P}+節?\s*[、,及]?\s*)+)(?:為|是)?\s*{_LAB}"),
]
_SAME_DAY = re.compile(rf"(?:第)?({_P}+)節?(?:為|是)\s*(?:TA)?(討論課|實習課)|(?:TA)?(討論課|實習課)時間為第?({_P}+)節")
_CLOCK = re.compile(rf"{_LAB}[^。]{{0,20}}?(?:週|星期)([{WEEKDAYS}])\s*(\d{{1,2}})[:：](\d{{2}})\s*[~～\-至]\s*(\d{{1,2}})[:：](\d{{2}})")
_LAB_CODE = re.compile(rf"{_LAB}[^。]{{0,30}}?上課時間(?:為|:)\s*([{WEEKDAYS}])\s*({_P}+)(?![\d:])")
_KIND = {"實習課": "lab", "實驗課": "lab", "討論課": "discussion"}


def _note_kind(text: str) -> str:
    m = re.search(_LAB, text)
    return _KIND[m.group(1)] if m else "lab"


def _covered(s: Session, main: list[Session]) -> bool:
    return any(m.weekday == s.weekday and set(s.periods) <= set(_all_periods(m, main)) for m in main)


def _all_periods(m: Session, main: list[Session]) -> str:
    return "".join(x.periods for x in main if x.weekday == m.weekday)


def note_sessions(note: str | None, main: list[Session]) -> list[Session]:
    """Lab / discussion sessions named in a note. `main` is the time-code result, used to
    resolve a weekday-less period (第E節為討論課) and to flag sessions it already covers."""
    text = (note or "").replace("：", ":")
    found: list[Session] = []
    tentative = "暫定" in text or "將於第一" in text
    for pat in _NOTE_PATTERNS:
        for m in pat.finditer(text):
            group = m.group(2) if pat is _NOTE_PATTERNS[0] else m.group(1)
            kind = _KIND[m.group(1) if pat is _NOTE_PATTERNS[0] else m.group(2)]
            for day, periods in re.findall(_DAYP, group):
                found += parse_time_code(day + re.sub(r"[、,]", "", periods), kind=kind, source="note",
                                         tentative=tentative)
    days = {s.weekday for s in main}
    if not found and len(days) == 1:
        for m in _SAME_DAY.finditer(text):
            periods, label = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            day = WEEKDAYS[next(iter(days)) - 1]
            found += parse_time_code(day + periods, kind=_KIND[label], source="note", tentative=tentative)
    for m in _LAB_CODE.finditer(text):
        found += parse_time_code(m.group(2) + m.group(3), kind=_KIND[m.group(1)], source="note", tentative=tentative)
    for m in _CLOCK.finditer(text):
        found.append(Session(WEEKDAYS.index(m.group(2)) + 1, f"{int(m.group(3)):02d}:{m.group(4)}",
                             f"{int(m.group(5)):02d}:{m.group(6)}", "", _KIND[m.group(1)], "note", tentative))
    uniq: dict[tuple, Session] = {}
    for s in found:
        flagged = replace(s, in_time_field=bool(s.periods) and _covered(s, main))
        uniq.setdefault((flagged.weekday, flagged.start, flagged.end, flagged.kind), flagged)
    return list(uniq.values())


def mentions_extra_session(note: str | None) -> bool:
    return bool(re.search(r"實習課|實驗課|討論課", note or ""))
