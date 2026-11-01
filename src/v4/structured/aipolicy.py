"""Generative-AI policy: category from the fixed labels, detail from the text.

ai_policy starts with one of four Chinese labels and ends with the English label glued
on (有條件開放使用:需經教師同意始得使用Conditional Permitted to Use). The conditional
detail is free text; when the field only points elsewhere (參見簡介, 依教師課堂說明) or
is empty, the AI sentences of evaluation and syllabus are used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.v4.structured.text import merge_fragments, nfkc

CATEGORIES = [("none", "本課程無涉及AI使用"), ("full", "完全開放使用"),
              ("conditional", "有條件開放使用"), ("prohibited", "禁止使用")]
LABEL_ZH = {"none": "無涉及AI", "full": "完全開放", "conditional": "有條件開放", "prohibited": "禁止", "other": "其他"}
_EN_TAIL = re.compile(r"(Conditional Permitted to Use|Prohibited Uses?|Completely Permitted to Use|"
                      r"This Course Does Not Involve the Use of AI\.?)\s*$", re.I)
_POINTER = re.compile(r"^(參見|詳見|請見|見|依|依照|依據|按|according to|(please\s+)?(see|refer to))[^。]{0,40}$", re.I)
_AI = re.compile(r"生成式|人工智慧|ChatGPT|Gemini|Claude|Copilot|Grammarly|\bAI\b|\bA\.I\.|G-AI|GenAI", re.I)


@dataclass(frozen=True)
class AiPolicy:
    category: str | None     # none / full / conditional / prohibited / other / None (empty)
    detail: str | None
    detail_source: str | None   # ai_policy / evaluation / syllabus


def _category(text: str) -> str | None:
    if not text:
        return None
    for key, label in CATEGORIES:
        if text.startswith(label):
            return key
    return "other"


def _detail(text: str, cat: str | None) -> str | None:
    label = dict(CATEGORIES).get(cat or "", "")
    body = text[len(label):] if label and text.startswith(label) else text
    body = _EN_TAIL.sub("", body.lstrip(":：").strip()).strip()
    return body or None


def ai_sentences(text: str | None, limit: int = 400) -> str | None:
    """Sentences mentioning AI tools, joined, capped at `limit` characters."""
    s = merge_fragments(nfkc(text))
    parts = [p.strip() for p in re.split(r"(?<=[。.!?！？;；])\s*|\n", s) if p.strip()]
    hits = [p for p in parts if _AI.search(p)]
    return " ".join(hits)[:limit] or None


def parse(ai_policy: str | None, evaluation: str | None = None, syllabus: str | None = None) -> AiPolicy:
    text = nfkc(ai_policy)
    cat = _category(text)
    detail = _detail(text, cat) if cat in ("conditional", "prohibited", "other") else None
    if cat == "other" and detail and detail.startswith("Course Name:"):
        detail = None   # scraper stored a page header here instead of the policy
    source = "ai_policy" if detail else None
    if cat in ("conditional", "prohibited", "other", None) and (not detail or _POINTER.match(detail)):
        for name, field in (("evaluation", evaluation), ("syllabus", syllabus)):
            found = ai_sentences(field)
            if found:
                detail = f"{detail}；{found}" if detail else found
                source = name if source is None else f"ai_policy+{name}"
                break
    return AiPolicy(cat, detail, source)
