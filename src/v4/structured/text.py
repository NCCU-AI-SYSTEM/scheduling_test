"""Text cleaning shared by every extractor.

COURSE text columns are scraped HTML: the new syllabus page joins each HTML block with
`|`, so a table becomes `週次|Week|課程主題|...` and CJK phrases get cut into pieces
(`GNSS|、|GIS|與|Map`). Two blocks of boilerplate repeat in every row (the schedule
header and the library notice in textbook, 2,786 of 2,795 courses each).
"""

from __future__ import annotations

import html
import re
import unicodedata

_PREFIX = re.compile(r"^\s*@\s*(異動資訊|備註|Information of alteration|Information|Note|Remarks?)\s*:\s*", re.I)
_EMPTY = {"", "無", "N/A", "n/a", "NA", "無。", "-", "none", "None"}
SCHED_HEAD = re.compile(r"^每周課程進度與作業要求\|Course Schedule & Requirements\|?")
TEXTBOOK_TAIL = re.compile(r"\|?已申請之圖書館指定參考書目.*?(Respect Copyright\.?|$)", re.S)
TEACH_HEAD = re.compile(r"^授課方式\|Teaching Approach\|?")

_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
_CJK_PUNCT = "，。、；：「」『』（）《》〈〉【】！？…—～·,.;:()!?/-~%&"
_LATIN = re.compile(r"[A-Za-z]")
_GLUE_END = "與及和的之或對、"
_GLUE_START = "與及和的之"
_OTHER_LETTER = re.compile(r"[Ѐ-ӿ؀-ۿ฀-๿Ā-ɏ]")


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def nfkc(text: str | None) -> str:
    """NFKC, entities decoded, control characters (C0 except tab/newline, C1) removed."""
    return _CONTROL.sub("", unicodedata.normalize("NFKC", html.unescape(text or ""))).strip()


def clean(text: str | None) -> str | None:
    """NFKC + entity decode + 備註/異動資訊 prefix removed; placeholder values -> None."""
    s = _PREFIX.sub("", nfkc(text)).strip()
    return None if s in _EMPTY else s


def _is_cjk_edge(s: str, end: bool) -> bool:
    ch = s[-1] if end else s[0]
    return bool(_CJK.match(ch)) or ch in _CJK_PUNCT


def _joiner(left: str, right: str) -> str:
    """Separator between two `|` cells: glue fragments of one phrase, keep real column breaks."""
    if left[-1].isdigit() and right[0].isdigit():
        return " "
    if re.fullmatch(r"\d{1,2}[.、]", right) and left[-1] in "。.!?！？":
        return " | "
    if right[0] in _CJK_PUNCT and right[0] not in "(（「『《〈【":
        return ""
    if left[-1] in "(（「『《〈【、，":
        return ""
    short = min(len(left), len(right)) <= 2
    if short and (_is_cjk_edge(left, True) or _is_cjk_edge(right, False)):
        return ""
    if left[-1] in _GLUE_END or right[0] in _GLUE_START:
        return ""
    if _is_cjk_edge(left, True) and _is_cjk_edge(right, False) and min(len(left), len(right)) <= 3:
        return ""
    if left[-1].isdigit() and right in ("st", "nd", "rd", "th"):
        return ""
    if _LATIN.search(left[-1]) and _LATIN.search(right[0]):
        if re.search(r"[a-z,]$", left) and right[0].islower() or min(len(left), len(right)) <= 3:
            return " "
        if re.search(r"\b(to|of|and|the|for|in|with|on|a|an|&)$", left, re.I):   # Introduction to|Narrative
            return " "
    return " | "


def merge_fragments(text: str | None) -> str:
    """Join `|`-separated HTML blocks back into phrases. Empty cells are dropped."""
    cells = [c.strip() for c in (text or "").split("|")]
    cells = [c for c in cells if c]
    if not cells:
        return ""
    out = cells[0]
    for c in cells[1:]:
        out += _joiner(out, c) + c
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def schedule_body(schedule: str | None) -> str:
    return SCHED_HEAD.sub("", nfkc(schedule))


def textbook_clean(textbook: str | None) -> str | None:
    s = TEXTBOOK_TAIL.sub("", nfkc(textbook)).strip(" |")
    return merge_fragments(s) or None


def text_lang(text: str | None) -> str | None:
    """'en' when Latin letters are > 70% of letters, 'zh' when < 20% and CJK leads,
    'other' when another script leads (Arabic, Cyrillic, Thai ...), else 'mixed'."""
    s = text or ""
    cjk, lat, oth = len(_CJK.findall(s)), len(_LATIN.findall(s)), len(_OTHER_LETTER.findall(s))
    total = cjk + lat + oth
    if total < 10:
        return None
    if oth / total > 0.5:
        return "other"
    en = lat / total
    if en > 0.7:
        return "en"
    if en < 0.2:
        return "zh"
    return "mixed"


def split_teachers(teacher: str | None) -> list[str]:
    s = nfkc(teacher)
    if not s or s == "詳備註":
        return []
    return [t.strip() for t in re.split(r"[、,，/]", s) if t.strip()]
