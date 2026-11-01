"""Merge cached LLM answers into the rule results. Inputs are built here so the cache keys
are the same in every run; `llm.run` asks only for keys missing from the cache."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from src.v4.structured import grading, weekly
from src.v4.structured.aliases import Resolution, Resolver
from src.v4.structured.assemble import Parsed
from src.v4.structured.clauses import Restriction
from src.v4.structured.llm import Cache, run

VALID_TYPES = {"dept_only", "college_only", "grade_min", "degree_only", "status_only", "special_class", "cohort",
               "prerequisite", "no_repeat", "not_for_graduation", "priority", "add_permit",
               "first_class_mandatory", "capacity", "fee", "intensive", "location", "english_taught",
               "pass_fail", "online", "joint_offering", "program", "assigned", "id_parity", "common_exam",
               "equipment", "schedule_info", "credit_rule", "open_to", "teaching_staff", "extra_session", "other"}
# a course states these once; an LLM item of the same type restates a continuation part
SINGLE = {"common_exam", "first_class_mandatory", "english_taught", "fee", "online", "pass_fail", "intensive",
          "location", "assigned", "id_parity", "cohort", "grade_min", "degree_only", "special_class"}
MAX_WEEK = 24   # a semester has 16-18 weeks; 34 means the model numbered something else
_SPLIT = re.compile(r"[、,，/]|以及|及|和|或者?是?|或")


# ── input builders (stable text -> stable cache key) ─────────────────────────
def notes_input(p: Parsed) -> str | None:
    if not p.unmatched:
        return None
    parts = "\n".join(f"{i}. {u}" for i, u in enumerate(p.unmatched, 1))
    return f"備註全文：{p.note}\n未處理片段：\n{parts}"


def grading_input(p: Parsed, raw_eval: str | None) -> str | None:
    return p.course["evaluation"] if grading.needs_llm(p.grading, raw_eval) else None


def weekly_input(p: Parsed, raw_schedule: str | None) -> str | None:
    return weekly.schedule_body(raw_schedule) if weekly.needs_llm(raw_schedule, list(p.weeks)) else None


# ── merging ──────────────────────────────────────────────────────────────────
def merge_notes(p: Parsed, cache: Cache) -> Parsed:
    text = notes_input(p)
    out = cache.get(text) if text else None
    if not out:
        return p
    add = []
    seen = {(r.type, r.value) for r in p.restrictions}
    single = {r.type for r in p.restrictions} & SINGLE
    for it in out.get("items") or []:
        if not isinstance(it, dict) or it.get("type") not in VALID_TYPES:
            continue
        idx = it.get("part")
        span = p.unmatched[idx - 1] if isinstance(idx, int) and 0 < idx <= len(p.unmatched) else ""
        value = str(it.get("value") or "").strip()[:60]
        if not value:
            continue
        pol = it.get("polarity") if it.get("polarity") in ("only", "not", "info") else "info"
        if (it["type"], value) in seen or it["type"] in single:
            continue
        seen.add((it["type"], value))
        if it["type"] in SINGLE:
            single.add(it["type"])
        add.append(Restriction(it["type"], value, pol, bool(it.get("hard")), "note", span, "llm"))
    return replace(p, restrictions=p.restrictions + tuple(add)) if add else p


def merge_grading(p: Parsed, cache: Cache, raw_eval: str | None) -> Parsed:
    text = grading_input(p, raw_eval)
    out = cache.get(text) if text else None
    if not out:
        return p
    items = []
    for it in out.get("items") or []:
        if not isinstance(it, dict) or not str(it.get("label") or "").strip():
            continue
        pct = it.get("pct")
        pct = float(pct) if isinstance(pct, (int, float)) and 0 < pct <= 100 else None
        label = str(it["label"]).strip()[:40]
        items.append(grading.EvalItem(grading.category(label), label, pct))
    scored = [i.pct for i in items if i.pct is not None]
    total = round(sum(scored), 1) if scored else None
    old = p.grading.pct_sum
    if not items or (old is not None and total is not None and abs(old - 100) < abs(total - 100)):
        return p
    g = grading.Grading(tuple(items), total, total is not None and 95 <= total <= 105, "llm")
    course = {**p.course, "eval_pct_sum": g.pct_sum, "eval_reliable": g.reliable, "eval_method": "llm"}
    return replace(p, grading=g, course=course)


def merge_weekly(p: Parsed, cache: Cache, raw_schedule: str | None) -> Parsed:
    text = weekly_input(p, raw_schedule)
    out = cache.get(text) if text else None
    rows = [w for w in (out or {}).get("weeks") or [] if isinstance(w, dict) and isinstance(w.get("week"), int)
            and 0 <= w["week"] <= MAX_WEEK and str(w.get("topic") or "").strip()]
    if len(rows) < 4:
        return p
    weeks = tuple(weekly.Week(w["week"], None, (str(w["date"]) if w.get("date") else None),
                              str(w.get("topic") or "")[:120], None, None, None) for w in rows)
    course = {**p.course, "n_weeks": len(weeks), "weekly_method": "llm",
              "schedule_clean": weekly.schedule_clean(raw_schedule, list(weeks))}
    return replace(p, weeks=weeks, weekly_method="llm", course=course)


# ── prerequisites ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PrereqMention:
    mention: str
    kind: str              # course / knowledge
    res: Resolution
    method: str            # rule / llm


def _clean_value(v: str) -> str:
    return re.sub(r"^[:：\s]+|(課程|等課程|之課程)$|[「」『』\"“”]", "", v).strip()


_MULTI_NUM = re.compile(r"^(.+?)((?:[(（][一二三四上下][)）]){2,})$")


def _expand(piece: str) -> list[str]:
    """寫作與口語訓練(一)(二) -> 寫作與口語訓練(一), 寫作與口語訓練(二)."""
    m = _MULTI_NUM.match(piece)
    if not m:
        return [piece]
    return [f"{m.group(1)}({n})" for n in re.findall(r"[一二三四上下]", m.group(2))]


def rule_split(value: str, resolver: Resolver) -> list[PrereqMention] | None:
    """Split a prerequisite value on 、 及 和 或 and resolve each piece; None if any fails."""
    v = _clean_value(value)
    if len(_expand(v)) == 1:
        whole = resolver.resolve(v)
        if whole.method != "unresolved":
            return [PrereqMention(v, "course", whole, "rule")]
    pieces = [y for x in _SPLIT.split(v) if x.strip() for y in _expand(x.strip())]
    out = [PrereqMention(x, "course", resolver.resolve(x), "rule") for x in pieces]
    return out if out and all(m.res.method != "unresolved" for m in out) else None


def prereq_spans(parsed: list[Parsed], resolver: Resolver) -> list[str]:
    return [r.span for p in parsed for r in p.restrictions
            if r.type == "prerequisite" and rule_split(r.value, resolver) is None]


def resolve_prereq(r: Restriction, resolver: Resolver, cache: Cache) -> list[PrereqMention]:
    found = rule_split(r.value, resolver)
    if found is not None:
        return found
    out = cache.get(r.span) or {}
    mentions = []
    for c in out.get("courses") or []:
        if isinstance(c, dict) and str(c.get("name") or "").strip():
            kind = "knowledge" if c.get("kind") == "knowledge" else "course"
            for name in _expand(_clean_value(str(c["name"]))):
                k = "see_syllabus" if re.search(r"詳|大綱|請見", name) else kind
                mentions.append(PrereqMention(name, k, resolver.resolve(name), "llm"))
    if not mentions:
        v = _clean_value(r.value)
        kind = "see_syllabus" if re.search(r"詳|大綱|請見", v) else "course"
        mentions = [PrereqMention(v, kind, resolver.resolve(v), "rule")]
    return mentions


def run_jobs(parsed: list[Parsed], raws: dict[str, dict], resolver: Resolver, mode: str, log=print) -> dict:
    """Ask (or read) every job; return the caches keyed by job."""
    notes = [t for p in parsed if (t := notes_input(p))]
    evals = [t for p in parsed if (t := grading_input(p, raws[p.course_id]["evaluation"]))]
    sched = [t for p in parsed if (t := weekly_input(p, raws[p.course_id]["schedule"]))]
    caches = {"notes": run("notes", notes, mode, log)}
    merged = [merge_notes(p, caches["notes"]) for p in parsed]
    caches["prereq"] = run("prereq", prereq_spans(merged, resolver), mode, log)
    caches["grading"] = run("grading", evals, mode, log)
    caches["weekly"] = run("weekly", sched, mode, log)
    return caches
