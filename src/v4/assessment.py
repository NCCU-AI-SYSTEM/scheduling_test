"""Assessment format as a search condition: 不要期中考, 要有小考.

A syllabus that names no assessment item says nothing either way: 8% of 114-2 syllabi
name none, and "not written" does not mean "not done". The user's rule (2026/10/19):
  violate  the syllabus clearly contradicts the condition -> excluded
  unknown  the syllabus does not say -> kept, ranked after every clear match, tagged 大綱未註明
  ok       the syllabus clearly matches
A generic 考試 / 期中 / 期末 mention without the exam type counts as unknown for exam items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FORMAT_KEYS = [
    # 期中評量 / 期末評量 are exams (user, 2026/11/01)
    ("期中考", r"期中(考|測驗|評量|筆試)|midterm"),
    ("期末考", r"期末(考|測驗|評量|筆試)|final\s*exam"),
    ("小考", r"小考|quiz|隨堂測驗"),
    ("作業", r"作業|homework|assignment|習題"),
    ("口頭報告", r"口頭|簡報|presentation|發表"),
    ("書面報告", r"報告|paper|essay|寫作|論文"),
    ("分組", r"分組|小組|group|team"),
    ("專題實作", r"專題|project|實作"),
    ("出席參與", r"出席|參與|participation|attendance|點名"),
]
# items the planner may put in an assessment condition; 分組 and 出席 stay preferences
# (group_work, attendance) so the two mechanisms do not compete for the same words
CONDITION_ITEMS = ("期中考", "期末考", "小考", "作業", "口頭報告", "書面報告", "專題實作")
EXAM_ITEMS = {"期中考", "期末考", "小考"}
# 期中／期末 alone may be an exam; 期末報告, 期中作業, 期末專題 ... are not
_GENERIC_EXAM = re.compile(r"考試|exam|測驗|\btest|期中(?![報作專書提呈發])|期末(?![報作專書提呈發])", re.I)


@dataclass(frozen=True)
class AssessmentProfile:
    items: frozenset
    generic_exam: bool

    @property
    def stated(self) -> bool:
        return bool(self.items) or self.generic_exam


def profile_from(labels: list[str], text: str) -> AssessmentProfile:
    """Items named in the parsed grading labels or anywhere in the evaluation text."""
    blob = " ".join(labels) + " " + (text or "")
    items = frozenset(k for k, pat in FORMAT_KEYS if re.search(pat, blob, re.I))
    return AssessmentProfile(items, bool(_GENERIC_EXAM.search(blob)) and not items & EXAM_ITEMS)


# ── query side ───────────────────────────────────────────────────────────────
# Rules, not the planner: adding this to the planner prompt cost 0.048 condition overlap
# on the R-set (0.741 -> 0.693), and the wording is a closed vocabulary.
_QUERY_ITEMS = [  # longest first; 上台報告 must win over 報告
    ("口頭報告", r"上台報告|口頭報告|上台|簡報|presentation"),
    ("書面報告", r"書面報告|期末報告|寫報告|報告"),
    ("期中考", r"期中考|期中測驗|期中評量|期中(?![報作專])"),
    ("期末考", r"期末考|期末測驗|期末評量|期末(?![報作專])"),
    ("小考", r"小考|隨堂考|quiz"),
    ("ALL_EXAMS", r"考試|考"),
    ("作業", r"作業|功課"),
    ("專題實作", r"專題|實作"),
]
_TRIGGER = re.compile(r"(不要|不用|不需要|不必|沒有|免|避開|不想|別|無)|(要有|需要有|想要有|包含|有)")
_STOP = re.compile(r"[，,。；;！!？?的課]|也|但")
_QUANT = re.compile(r"太多|很多|多|少|一點")


def parse_query(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """不要期中考 -> ((), ('期中考',)); 要有期中期末考 -> (('期中考','期末考'), ()).
    Amount words (作業不要太多, 少一點報告) are workload preferences, not conditions."""
    q = re.sub(r"有沒有|有無|有些|有點|有趣|有用|沒有很", "  ", query or "")
    must: list[str] = []
    must_not: list[str] = []
    for m in _TRIGGER.finditer(q):
        stop = _STOP.search(q, m.end())
        window = q[m.end():stop.start() if stop else len(q)][:12]
        if _QUANT.search(window):
            continue
        found, rest = [], window
        for item, pat in _QUERY_ITEMS:
            for hit in re.finditer(pat, rest, re.I):
                found.append((hit.start(), item))
            rest = re.sub(pat, lambda x: " " * len(x.group()), rest, flags=re.I)
        items = [it for _, it in sorted(found)]
        items = [x for it in items for x in (sorted(EXAM_ITEMS) if it == "ALL_EXAMS" else [it])]
        (must_not if m.group(1) else must).extend(items)
    return tuple(dict.fromkeys(must)), tuple(dict.fromkeys(i for i in must_not if i not in must))


def _one(item: str, p: AssessmentProfile) -> str:
    """Does the course have this item: yes / no / unknown."""
    if item in p.items:
        return "yes"
    if not p.stated or (item in EXAM_ITEMS and p.generic_exam):
        return "unknown"
    return "no"


def verdict(p: AssessmentProfile, must: tuple[str, ...], must_not: tuple[str, ...]) -> str:
    answers = [("yes", _one(i, p)) for i in must] + [("no", _one(i, p)) for i in must_not]
    if any(got not in (want, "unknown") for want, got in answers):
        return "violate"
    if any(got == "unknown" for _, got in answers):
        return "unknown"
    return "ok"
