"""evaluation text -> graded items with percentages.

Teachers write the breakdown in free form. The rule parser handles the three layouts
that cover most syllabi:
  label before percent   期中考 30%  /  Midterm Exam: 20%  /  Design Project 1 (30%)
  label after percent    25% Worksheets 25% Final exam
  nested breakdowns      平時 30% ... (1)出席率 15% (2)課堂參與 15%  -> children dropped
and drops total lines (Total 100%). A result is reliable when the kept items sum to
95-105; the rest goes to the LLM fallback (llm.py) with this sum as the check.
Categories reuse src.v4.assessment.FORMAT_KEYS, plus 考試 for a generic exam and other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.v4.assessment import FORMAT_KEYS
from src.v4.structured.text import merge_fragments, nfkc

_PCT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")
_TOTAL = re.compile(r"總計|合計|總分|共計|小計|總成績|total|sum", re.I)
_LEAD = re.compile(r"^[\s\-–—•·‧*#>:=、,.]*(?:[（(]?\s*(?:\d{1,2}|[一二三四五六七八九十]|[a-zA-Z])\s*[)）.、]\s*)?")
_TRAIL = " \t:=：-–—(（[【佔占為"
EXTRA_KEYS = [("考試", r"考試|exam|test|測驗"),
              ("出席參與", r"缺席|出缺|討論|課堂表現|表現|performance|discussion")]
CATEGORIES = [k for k, _ in FORMAT_KEYS] + ["考試", "other"]


@dataclass(frozen=True)
class EvalItem:
    category: str
    label: str
    pct: float | None


@dataclass(frozen=True)
class Grading:
    items: tuple[EvalItem, ...]
    pct_sum: float | None
    reliable: bool
    method: str           # rule / llm / none


_SYNONYM = [(r"mid-term|mid term|中間試験|中間テスト", "midterm"), (r"期末試験|期末テスト", "期末考")]
_GENERIC = re.compile(r"^(評分|配分|佔|占|比重|比例|權重|約|共|計|成績|分數|總成績之?)$")


def category(label: str) -> str:
    bare = re.sub(r"[(（][^)）]*[)）]?", "", label)
    for pat, rep in _SYNONYM:
        bare = re.sub(pat, rep, bare, flags=re.I)
    for k, pat in FORMAT_KEYS + EXTRA_KEYS:
        if re.search(pat, bare, re.I):
            return k
    return "other"


def _tidy(label: str) -> str:
    s = label.strip().strip(_TRAIL)
    s = [p for p in re.split(r"[:：]|(?<!\d)\d{1,2}[.、](?!\d)", s) if p.strip()][-1:] or [""]
    s = _LEAD.sub("", s[0].strip()).strip(_TRAIL + ")）]】")
    return re.sub(r"\s{2,}", " ", s)[:40]


def _before(text: str, start: int, lo: int) -> str:
    chunks = [c for c in re.split(r"[。;；\n|)）]|\s{2,}", text[lo:start]) if c.strip(_TRAIL)]
    # 期中報告(評分25%): the word before the percent is generic, the label sits before '('
    if chunks:
        parts = re.split(r"[(（]", chunks[-1])
        if _GENERIC.match(parts[-1].strip(_TRAIL)):
            head = "(".join(parts[:-1]).strip(_TRAIL)
            chunks = chunks[:-1] + ([head] if head else [])
    chunk = chunks[-1] if chunks else ""
    if len(chunk) > 40:
        cut = len(chunk) - 40
        if chunk[cut - 1].isalpha() and chunk[cut].isalpha() and chunk[cut].isascii():
            nxt = chunk.find(" ", cut)
            cut = nxt + 1 if nxt != -1 else cut
        chunk = chunk[cut:]
    return _tidy(chunk)


def _after(text: str, end: int, hi: int) -> str:
    tail = re.sub(r"^\s*[(（][^)）]{0,20}[)）]", "", text[end:hi].strip(" :："))
    chunks = [c for c in re.split(r"[。;；\n|]|\s{3,}", tail) if c.strip(_TRAIL)]
    return _tidy(chunks[0][:40]) if chunks else ""


def raw_items(text: str) -> list[tuple[str, float]]:
    """(label, pct) in text order, picking the orientation that labels more percents."""
    ms = [m for m in _PCT.finditer(text) if 0 < float(m.group(1)) <= 100]
    if not ms:
        return []
    bounds = [0] + [m.end() for m in ms]
    nexts = [m.start() for m in ms[1:]] + [len(text)]
    before = [_before(text, m.start(), bounds[i]) for i, m in enumerate(ms)]
    after = [_after(text, m.end(), nexts[i]) for i, m in enumerate(ms)]
    labelled = lambda xs: sum(1 for x in xs if re.search(r"[A-Za-z一-鿿]", x))
    labels = after if labelled(after) > labelled(before) else before
    return [(lab, float(m.group(1))) for lab, m in zip(labels, ms)]


def _drop_totals(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    out = [(l, p) for l, p in items if not (_TOTAL.search(l) and p == 100)]
    if len(out) > 1 and out[-1][1] == 100 and abs(sum(p for _, p in out[:-1]) - 100) < 1:
        out = out[:-1]
    return out


def _collapse_children(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Drop runs of items that add up exactly to the item before them."""
    out: list[tuple[str, float]] = []
    i = 0
    while i < len(items):
        out.append(items[i])
        acc, j = 0.0, i + 1
        while j < len(items) and acc < items[i][1]:
            acc += items[j][1]
            j += 1
        i = j if j > i + 2 and abs(acc - items[i][1]) < 0.5 else i + 1
    return out


def _prefix_100(items: list[tuple[str, float]]) -> list[tuple[str, float]] | None:
    acc = 0.0
    for k, (_, p) in enumerate(items):
        acc += p
        if abs(acc - 100) < 0.5 and k >= 1:
            return items[: k + 1]
        if acc > 100.5:
            return None
    return None


_CJK = re.compile(r"[一-鿿]")


def _bilingual_half(items: list[tuple[str, float]]) -> list[tuple[str, float]] | None:
    """出席 50% 測驗 40% 其他 10% Attendance (50%) Examinations (40%) Others (10%):
    the same breakdown written twice. Without this, _collapse_children reads 測驗 + 其他
    (40 + 10) as children of the first 50% and keeps only the two attendance items.
    Needs equal percent sequences, each half summing to ~100, and labels in different
    scripts in the two halves; keeps the half with Chinese labels."""
    n = len(items)
    if n < 4 or n % 2:
        return None
    a, b = items[: n // 2], items[n // 2:]
    if [p for _, p in a] != [p for _, p in b] or not 95 <= sum(p for _, p in a) <= 105:
        return None
    cjk = lambda half: sum(bool(_CJK.search(lbl)) for lbl, _ in half)  # noqa: E731
    if cjk(a) == cjk(b):
        return None
    return a if cjk(a) > cjk(b) else b


def select(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    items = _drop_totals(items)
    total = sum(p for _, p in items)
    if 95 <= total <= 105 or total < 95:
        return items
    for cand in (_bilingual_half(items), _collapse_children(items), _prefix_100(items)):
        if cand and 95 <= sum(p for _, p in cand) <= 105:
            return cand
    return items


def mentioned(text: str, have: set[str]) -> list[EvalItem]:
    """Categories named without a percentage (pct NULL), skipping ones already scored."""
    out = []
    for k, pat in FORMAT_KEYS:
        m = re.search(pat, text, re.I)
        if m and k not in have:
            out.append(EvalItem(k, m.group(0), None))
    return out


_CN_DIGIT = {c: i for i, c in enumerate("零一二三四五六七八九")}


def _cn_number(s: str) -> int | None:
    """四十 -> 40, 十五 -> 15, 一百 -> 100, 五 -> 5."""
    if s == "一百" or s == "百":
        return 100
    if "十" in s:
        tens, _, ones = s.partition("十")
        return (_CN_DIGIT.get(tens, 1) if tens else 1) * 10 + (_CN_DIGIT.get(ones, 0) if ones else 0)
    return _CN_DIGIT.get(s) if len(s) == 1 else None


def cn_percent(text: str) -> str:
    """百分之四十 -> 40% (PE syllabi write percents in words)."""
    def rep(m: re.Match) -> str:
        s, tail = m.group(1), ""
        nxt = m.string[m.end():m.end() + 1]
        if len(s) > 1 and nxt in ".、．" and s[-1] != "十":   # 百分之四十二.出缺席: 二. numbers the next item
            s, tail = s[:-1], s[-1]
        n = _cn_number(s)
        return f"{n}%{tail}" if n is not None else m.group(0)
    return re.sub(r"百分之([零一二三四五六七八九十百]{1,3})", rep, text)


def parse(evaluation: str | None) -> Grading:
    text = cn_percent(merge_fragments(nfkc(evaluation)))
    if not text:
        return Grading((), None, False, "none")
    picked = select(raw_items(text))
    scored = tuple(EvalItem(category(l), l, p) for l, p in picked)
    total = round(sum(p for _, p in picked), 1) if picked else None
    reliable = total is not None and 95 <= total <= 105
    items = scored if reliable else scored + tuple(mentioned(text, {i.category for i in scored}))
    return Grading(items, total, reliable, "rule" if items else "none")


def needs_llm(g: Grading, evaluation: str | None) -> bool:
    """Percents exist but do not add up, or the text has scores the rules found none of."""
    if g.reliable:
        return False
    text = nfkc(evaluation)
    return bool(_PCT.search(text)) or bool(re.search(r"\d+\s*(分|points?|pts)", text, re.I))
