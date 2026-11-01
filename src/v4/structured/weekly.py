"""schedule table -> one row per week (week, date, topic, detail, hours).

The schedule column is a flattened table. Row layouts seen in 114-2:
  1|行政契約…|…|4|小時            bare week number, hours at the end
  W01|02/25|Syllabus & ITS|…|3|3  W-prefixed number, date, in-class and outside hours
  第|1|周（|2/24|）請求權基礎…      第 n 周 split over cells
  Week 1: Introduction to …       number and topic in one cell
Week markers are found as the sequence 1, 2, 3 … in cell order; when a number occurs in
several cells, the marker is the one followed by a date or text (hours are numbers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.v4.structured.text import merge_fragments, schedule_body

_CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九十", 0)}
_WK = r"(\d{1,2}|[一二三四五六七八九十]{1,3})"
_MARK_ONLY = re.compile(rf"^(?:W|Week\s*|Session\s*|Lecture\s*|Class\s*|第\s*)?0?{_WK}\s*(?:[-~–]\s*(\d{{1,2}}))?\s*(?:週|周|week)?\.?$", re.I)
_MARK_SPACE = re.compile(r"^0?(\d{1,2})(?:\s*[-~–]\s*(\d{1,2}))?(?:[.、:：]\s*|\s+)(?=\D)(.{2,})$")
_MARK_INLINE = re.compile(rf"^(?:Week|W|Session|Lecture|Class|第)\s*0?{_WK}\s*(?:週|周)?\s*(?:Week\s*\d{{1,2}})?\s*[:：.\-–)）]?\s*(.{{2,}})$", re.I)
_MARK_DATE = re.compile(r"^(?:第)?(\d{1,2}|[一二三四五六七八九十]{1,3})\s*(?:週|周)?\s*[(（](\d{1,2}/\d{1,2})[)）]$")
_DATE = re.compile(r"(\d{1,2}/\d{1,2}(?:\s*[~～\-–]\s*\d{1,2}(?:/\d{1,2})?)?|"
                   r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{1,2}(?:\s*[-–]\s*\d{1,2})?)", re.I)
_WEEKDAY_TAG = re.compile(r"[(（]?\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?\s*[)）]?|[(（][一二三四五六日][)）]|"
                          r"الاثنين|الثلاثاء|الأربعاء|الخميس|الجمعة|السبت|الأحد", re.I)
_MODE = re.compile(r"實體課程|線上課程|遠距課程|同步線上|非同步線上|實體|線上|In-person class|Online", re.I)
_NUM = re.compile(r"^\d+(?:\.\d+)?(?:\s*[-~–]\s*\d+(?:\.\d+)?)?$")
_JUNK = re.compile(r"^(第|周|週|週次|Week|W|周[（(]|週[（(]|[（(）)]|[–\-—:：•·]+|小時|hours?|hrs?|Hours?)$", re.I)
MIN_WEEKS = 6


@dataclass(frozen=True)
class Week:
    week: int
    week_end: int | None   # 1-8 written as one row
    date: str | None
    topic: str
    detail: str | None
    hours_in: float | None
    hours_out: float | None


def _num(tok: str) -> int:
    if tok.isdigit():
        return int(tok)
    if len(tok) == 1:
        return _CN_NUM[tok]
    if tok[0] == "十":
        return 10 + _CN_NUM.get(tok[1:], 0)
    return _CN_NUM[tok[0]] * 10 + (_CN_NUM.get(tok[2:], 0) if len(tok) > 2 else 0)


def _marker(cell: str) -> tuple[int, str, int | None, str] | None:
    """(week, inline text, range end, style). Inline text may start with the date '(3/9)'.
    style separates bare numbers, 第n週 / Week n, n(date) and 'n. text' so a numbered
    list in the header cannot splice into the real week run."""
    prefixed = bool(re.match(r"(W|Week|Session|Lecture|Class|第)", cell, re.I))
    if re.fullmatch(r"[一二三四五六七八九十]{1,3}", cell):
        return _num(cell), "", None, "cjk"
    m = _MARK_ONLY.match(cell)
    if m and (m.group(1).isdigit() or re.search(r"第|週|周", cell)):
        style = "list" if cell.endswith(".") else "prefix" if prefixed else "bare"
        return _num(m.group(1)), "", int(m.group(2)) if m.group(2) else None, style
    m = _MARK_DATE.match(cell)
    if m:
        return _num(m.group(1)), f"({m.group(2)})", None, "date"
    m = _MARK_SPACE.match(cell)
    if m:
        return int(m.group(1)), m.group(3).strip(), int(m.group(2)) if m.group(2) else None, "list"
    m = _MARK_INLINE.match(cell)
    # 第二課 is a lesson, not a week: 第 needs 週/周 after the number
    if m and cell.startswith("第") and not re.match(rf"第\s*{_WK}\s*[週周]", cell):
        m = None
    if m and (m.group(1).isdigit() or re.search(r"第|週|周", cell)):
        return _num(m.group(1)), m.group(2).strip(), None, "prefix"
    return None


def _next_is_content(cells: list[str], i: int) -> bool:
    j = i + 1
    while j < len(cells) and _JUNK.match(cells[j]):
        j += 1
    return j < len(cells) and not _NUM.match(cells[j])


def _run(cells: list[str], cands: list[tuple]) -> list[tuple[int, int, str, int | None]]:
    """Follow 1, 2, 3 ... through the candidates. The exact next week is preferred over a
    skipped one anywhere before the week after the skip, so hours like 3|4-5 sitting
    before the real marker 2 cannot be taken as week 3."""
    ok = [c for c in cands if c[2] or _next_is_content(cells, c[0])]
    start = next((k for k, c in enumerate(ok) if c[1] in (0, 1, 2)), None)
    if start is None:
        return []
    out = [ok[start]]
    k = start
    while True:
        expect = (out[-1][3] or out[-1][1]) + 1
        rest = ok[k + 1:]
        stop = next((j for j, c in enumerate(rest) if c[1] == expect + 2), len(rest))
        exact = next((j for j, c in enumerate(rest[:stop]) if c[1] == expect), None)
        skip = next((j for j, c in enumerate(rest[:stop]) if c[1] == expect + 1), None)
        j = exact if exact is not None else skip
        if j is None:
            return out
        out.append(rest[j])
        k += 1 + j


def find_markers(cells: list[str]) -> list[tuple[int, int, str, int | None]]:
    """(cell index, week, inline topic, range end) for the longest 1..N run of one style
    (bare numbers and W01 are kept apart, so hours cells cannot join a W run)."""
    cands = [(i, *mk) for i, c in enumerate(cells) if (mk := _marker(c))]
    groups = {"prefix": [c for c in cands if c[4] in ("prefix", "date")],
              "bare": [c for c in cands if c[4] in ("bare", "date")],
              "list": [c for c in cands if c[4] == "list"],
              "cjk": [c for c in cands if c[4] in ("cjk", "date")]}
    runs = [_run(cells, [c[:4] for c in g]) for g in groups.values()]
    return max(runs, key=len)   # ties keep the first group: prefixed markers win


_LEAD_DATE = re.compile(r"^[•·\-]?\s*(\d{1,2})/(\d{1,2})\b")


def date_markers(cells: list[str]) -> list[tuple[int, int, str, int | None]]:
    """No week numbers: cells that start with an increasing M/D date become weeks 1..n."""
    out, last = [], (0, 0)
    for i, c in enumerate(cells):
        m = _LEAD_DATE.match(c)
        if m and (int(m.group(1)), int(m.group(2))) > last and 1 <= int(m.group(1)) <= 12:
            last = (int(m.group(1)), int(m.group(2)))
            out.append((i, len(out) + 1, c.lstrip("•·- "), None))
    return out


def _hours(seg: list[str], two_cols: bool) -> tuple[float | None, float | None, list[str]]:
    nums: list[float] = []
    k = len(seg)
    while k > 0 and (_NUM.match(seg[k - 1]) or _JUNK.match(seg[k - 1])):
        k -= 1
        if _NUM.match(seg[k]):
            vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", seg[k])]
            nums.insert(0, sum(vals) / len(vals))
    nums = [x for x in nums if x <= 40]
    rest = seg[:k]
    if two_cols and len(nums) >= 2:
        return nums[-2], nums[-1], rest
    if len(nums) == 1:
        return nums[0], None, rest
    return None, None, rest


def _week(n: int, end: int | None, seg: list[str], inline: str, two_cols: bool) -> Week:
    h_in, h_out, rest = _hours(seg, two_cols)
    date = None
    body: list[str] = [inline] if inline else []
    after_date = False
    for c in rest:
        if date is None and not body and _DATE.fullmatch(c.strip("()（）")):
            date, after_date = c.strip("()（）"), True
            continue
        if after_date and not body and _WEEKDAY_TAG.fullmatch(c):
            continue
        if _JUNK.match(c):
            continue
        body.append(c.lstrip(")）") if after_date and not body else c)
    if date is None and body:
        m = (re.match(r"^[(（]?(\d{1,2}/\d{1,2})[)）]?\s*(.*)$", body[0])
             or re.match(r"^[(（](\d{1,2}\.\d{2})[)）]\s*(.*)$", body[0]))
        if m:
            date, body[0] = m.group(1), m.group(2)
    text = merge_fragments("|".join(b for b in body if b))
    text = re.sub(r"^((st|nd|rd|th)\s*week\s*:?\s*|(Class Topic|Topic|課程主題|主題)\s*:\s*)", "",
                  text.lstrip(":：–-—)） "), flags=re.I)
    topic, _, detail = text.partition(" | ")
    if len(topic) < 4 and detail:
        nxt, _, detail = detail.partition(" | ")
        topic = f"{topic} {nxt}".strip()
    topic = re.sub(r"^(Class Topic|Topic|課程主題)\s*:\s*", "", topic.lstrip(":：–-—)） "), flags=re.I)
    topic = re.sub(r"^[週周]\s*[:：]\s*", "", topic)
    if (_MODE.fullmatch(topic) or _WEEKDAY_TAG.fullmatch(topic)) and detail:
        topic, _, detail = detail.partition(" | ")
    return Week(n, end, date, topic[:120].strip(), detail[:300].strip() or None, h_in, h_out)


def _cells(schedule: str | None) -> list[str]:
    body = schedule_body(schedule)
    body = re.sub(r"第\s*\|?\s*(\d{1,2})\s*\|?\s*(週|周)\s*[（(]?\s*\|?", r"|第\1\2|", body)
    body = re.sub(r"(?:(?<=\|)|^)(Week|W|Session)\s*\|\s*(\d{1,2})(?=\|)", r"\1 \2", body, flags=re.I)
    body = re.sub(r"(?<![\d/])(\d{1,2})\s*\|?\s*月\s*\|?\s*(\d{1,2})\s*\|?\s*日", r"\1/\2", body)
    return [c.strip() for c in body.split("|") if c.strip()]


def parse(schedule: str | None) -> list[Week]:
    cells = _cells(schedule)
    marks = find_markers(cells)
    if len(marks) < MIN_WEEKS:
        marks = date_markers(cells)
    if len(marks) < MIN_WEEKS:
        return []
    head = " ".join(cells[: marks[0][0]])
    two_cols = bool(re.search(r"In-class|課堂講授", head, re.I) and re.search(r"Outside|課程前後|課後", head, re.I))
    ends = [m[0] for m in marks[1:]] + [len(cells)]
    weeks = [_week(n, rend, cells[i + 1:end], inline, two_cols) for (i, n, inline, rend), end in zip(marks, ends)]
    return [w for w in weeks if w.topic or w.date]


def schedule_clean(schedule: str | None, weeks: list[Week]) -> str | None:
    """Readable schedule: one line per parsed week, else the fragment-merged text."""
    if weeks:
        return "\n".join(f"第{w.week}週" + (f"（{w.date}）" if w.date else "") + f" {w.topic}" for w in weeks)
    body = merge_fragments(schedule_body(schedule))
    return body or None


def needs_llm(schedule: str | None, weeks: list[Week]) -> bool:
    body = schedule_body(schedule)
    return not weeks and len(body) >= 150
