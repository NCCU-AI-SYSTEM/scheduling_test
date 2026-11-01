"""Orchestration: raw COURSE -> rule pass -> cached LLM fallback -> aliases -> tables."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from src.v4.catalog import load_catalog
from src.v4.offering import dept_registry, group_offerings
from src.v4.structured import fallback, store
from src.v4.structured.aipolicy import LABEL_ZH
from src.v4.structured.aliases import Resolver, build_aliases, load_catalogue
from src.v4.structured.assemble import Parsed, load_raw, load_result, parse_course, path_rows
from src.v4.structured.card import render_card

ROOT = Path(__file__).resolve().parents[3]
SRC_DB = ROOT / "data" / "1142.db"
OUT_DB = ROOT / "data" / "v4" / "course_struct.db"


def _session_rows(p: Parsed) -> list[dict]:
    return [{"course_id": p.course_id, "seq": i, **asdict(s)} for i, s in enumerate(p.sessions)]


def _target_rows(p: Parsed) -> list[dict]:
    return [{"course_id": p.course_id, **asdict(t)} for t in p.targets]


def _eval_rows(p: Parsed) -> list[dict]:
    return [{"course_id": p.course_id, "seq": i, "category": e.category, "label": e.label, "pct": e.pct,
             "method": p.grading.method} for i, e in enumerate(p.grading.items)]


def _weekly_rows(p: Parsed) -> list[dict]:
    return [{"course_id": p.course_id, **asdict(w), "method": p.weekly_method} for w in p.weeks]


def _restriction_rows(parsed: list[Parsed], resolver: Resolver, cache) -> tuple[list[dict], list[dict]]:
    restr, prereq = [], []
    for p in parsed:
        for r in p.restrictions:
            rid = len(restr) + 1
            value = r.value
            if r.type == "prerequisite":
                ms = fallback.resolve_prereq(r, resolver, cache)
                for m in ms:
                    prereq.append({"restriction_id": rid, "course_id": p.course_id, "mention": m.mention,
                                   "kind": m.kind, "resolved_base": m.res.base, "resolved_names": list(m.res.names),
                                   "method": f"{m.method}/{m.res.method}", "in_1142": m.res.in_1142})
                names = [m.res.names[0] if len(m.res.names) == 1 else (m.res.base or m.mention) for m in ms]
                value = "、".join(dict.fromkeys(names)) if names else value
            restr.append({"id": rid, "course_id": p.course_id, "type": r.type, "value": value,
                          "polarity": r.polarity, "hard": r.hard, "source": r.source, "span": r.span,
                          "method": r.method})
    return restr, prereq


def _offering_rows(depts) -> tuple[list[dict], list[dict]]:
    offs = group_offerings(load_catalog(with_reviews=False), depts)
    o_rows, c_rows = [], []
    for i, o in enumerate(offs, 1):
        o_rows.append({"offering_id": i, "name_norm": o.key[0], "teacher": o.key[1],
                       "time_raw": o.key[2] if not o.key[2].startswith("1142") else None,
                       "primary_course_id": o.primary, "n_channels": len(o.channels)})
        for ch in o.channels:
            c_rows.append({"offering_id": i, "course_id": ch.course_id, "sub_num": ch.sub_num, "unit": ch.unit,
                           "dept_code": ch.dept_code, "college": ch.college, "level": ch.level,
                           "targets": sorted(f"{lv}{y}" for lv, y in ch.targets), "enroll_mode": ch.enroll_mode,
                           "audience_level": ch.audience_level, "colleges_only": ch.colleges_only,
                           "status_only": ch.status_only, "special_class": ch.special_class, "url": ch.url})
    return o_rows, c_rows


def build(mode: str = "off", out: Path = OUT_DB, log=print) -> dict[str, int]:
    t0 = time.time()
    con = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    depts = dept_registry(SRC_DB)
    raws = load_raw(con)
    result = load_result(con)
    parsed = [parse_course(r, depts, result) for r in raws]
    raw_by_id = {r.course_id: r.row for r in raws}
    log(f"rule pass: {len(parsed)} courses in {time.time() - t0:.1f}s")

    cat = load_catalogue(con)
    aliases = build_aliases(con, cat)
    resolver = Resolver(aliases, cat)
    log(f"aliases: {len(aliases)}")

    caches = fallback.run_jobs(parsed, raw_by_id, resolver, mode, log)
    merged = []
    for p in parsed:
        row = raw_by_id[p.course_id]
        p = fallback.merge_notes(p, caches["notes"])
        p = fallback.merge_grading(p, caches["grading"], row["evaluation"])
        p = fallback.merge_weekly(p, caches["weekly"], row["schedule"])
        merged.append(p)

    restr, prereq = _restriction_rows(merged, resolver, caches["prereq"])
    tables = {"course": [], "course_path": [], "session": [], "target_class": [], "restriction": restr,
              "prereq": prereq, "change": [], "eval_item": [], "ai_policy": [], "weekly": [], "alias": []}
    by_course: dict[str, list[dict]] = {}
    for r in restr:
        by_course.setdefault(r["course_id"], []).append(r)
    for p in merged:
        ses, tgt, ev, wk = _session_rows(p), _target_rows(p), _eval_rows(p), _weekly_rows(p)
        card = render_card(p.course, ses, tgt, by_course.get(p.course_id, []), ev, wk)
        tables["course"].append({**p.course, "card": card})
        tables["course_path"] += path_rows(p)
        tables["session"] += ses
        tables["target_class"] += tgt
        tables["eval_item"] += ev
        tables["weekly"] += wk
        tables["change"] += [{"course_id": p.course_id, **asdict(c)} for c in p.changes]
        tables["ai_policy"].append({"course_id": p.course_id, "category": p.ai.category,
                                    "label_zh": LABEL_ZH.get(p.ai.category or "", None),
                                    "detail": p.ai.detail, "detail_source": p.ai.detail_source})
    tables["alias"] = [{**asdict(a), "members": list(a.members)} for a in aliases]
    tables["offering"], tables["channel"] = _offering_rows(depts)
    con.close()
    meta = {"built_at": time.strftime("%Y/%m/%d %H:%M"), "source_db": str(SRC_DB.name), "semester": "114-2",
            "llm_mode": mode, "seconds": f"{time.time() - t0:.1f}"}
    store.write(out, tables, meta)
    counts = {k: len(v) for k, v in tables.items()}
    log(f"wrote {out} {counts} in {time.time() - t0:.1f}s")
    return counts
