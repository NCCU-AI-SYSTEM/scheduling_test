"""note / info -> restriction clauses and change events, by rules.

A note is split into clauses at 。；; and list numbers (1. 2.). Every rule runs over a
clause; one clause can yield several restrictions (英語授課，學碩合開 -> english_taught
and joint_offering). The span stored with a restriction is the comma-delimited part
of the clause holding the match, so it can be shown or checked against the source.
Parts no rule touches are returned separately for the LLM fallback (llm.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.v4.structured.rules import RULES, Rule

_CLAUSE_SPLIT = re.compile(r"[。；;\n]+|(?:^|(?<=[\s。；;，,]))(?:\d{1,2}[\.、)]|[（(]\d[）)])\s*")
_PART_SPLIT = re.compile(r"[，,]|(?<!\d)/(?!\d)")
_SOFT = re.compile(r"建議|鼓勵|較妥|最好|宜|優先")
_HARD = re.compile(r"限|僅|不開放|不得|不接受|擋修|須|需|必須|務必|必到|一律|不收")
_CHANGE = re.compile(r"(停開|教室異動|時間異動|教師異動)於(\d{4}/\d{2}/\d{2})")
CHANGE_TYPE = {"停開": "cancelled", "教室異動": "classroom", "時間異動": "time", "教師異動": "teacher"}


@dataclass(frozen=True)
class Restriction:
    type: str
    value: str
    polarity: str     # only = limited to value; not = value is excluded; info = no limit
    hard: bool
    source: str       # note / info / llm-note
    span: str
    method: str = "rule"


@dataclass(frozen=True)
class Change:
    type: str         # cancelled / classroom / time / teacher
    date: str         # YYYY/MM/DD as written
    raw: str


def clauses(text: str | None) -> list[str]:
    parts = [p.strip(" ，,、:：") for p in _CLAUSE_SPLIT.split(text or "")]
    return [p for p in parts if p]


def _part_of(clause: str, pos: int) -> str:
    start = 0
    for m in _PART_SPLIT.finditer(clause):
        if m.start() >= pos:
            return clause[start:m.start()].strip()
        start = m.end()
    return clause[start:].strip()


def _hard(rule: Rule, span: str) -> bool:
    if rule.hard is not None:
        return rule.hard
    if _SOFT.search(span):
        return False
    return bool(_HARD.search(span))


def apply_rules(clause: str, source: str = "note") -> tuple[list[Restriction], list[str]]:
    """Restrictions found in one clause, plus the comma parts no rule matched."""
    found: list[Restriction] = []
    covered: set[str] = set()
    for rule in RULES:
        kept = 0
        for m in rule.pattern.finditer(clause):
            span = _part_of(clause, m.start())
            if rule.once and kept:
                covered.add(span)
                continue
            value = rule.value(m, clause)
            if value is None:
                continue
            covered.add(span)
            polarity = rule.polarity(m, clause) if callable(rule.polarity) else rule.polarity
            found.append(Restriction(rule.type, value, polarity, _hard(rule, span), source, span))
            kept += 1
    parts = [p.strip() for p in _PART_SPLIT.split(clause)]
    rest = [p for p in parts if len(p) >= 4 and p not in covered
            and not any(p in c or c in p for c in covered)]
    uniq: dict[tuple, Restriction] = {}
    for r in found:
        uniq.setdefault((r.type, r.value, r.polarity), r)
    return list(uniq.values()), rest


def parse_note(note: str | None, source: str = "note") -> tuple[list[Restriction], list[str]]:
    out: list[Restriction] = []
    unmatched: list[str] = []
    seen: set[tuple] = set()
    for c in clauses(note):
        found, rest = apply_rules(c, source)
        for r in found:
            if (r.type, r.value, r.polarity) not in seen:   # same rule hit in two clauses
                seen.add((r.type, r.value, r.polarity))
                out.append(r)
        unmatched += rest
    return out, unmatched


def parse_info(info: str | None) -> list[Change]:
    return [Change(CHANGE_TYPE[t], d, f"{t}於{d}") for t, d in _CHANGE.findall(info or "")]
