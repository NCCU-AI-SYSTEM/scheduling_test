"""Precision samples, judgement files and consistency checks for course_struct.db.

Samples are drawn with a fixed seed so the same rows come back after a rebuild with the
same content. Each sample row carries the raw source text the value came from; a human
(or the session that built the data) writes a judgement line {"i", "ok", "why"} per
sample row into data/v4/structured_audit/judged_<field>.jsonl.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

from src.v4.structured.sessions import parse_time_code
from src.v4.structured.text import clean, nfkc

SEED = 20261019
N = 40
_WD = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}


def _sample(rows: list, n: int = N) -> list:
    rng = random.Random(SEED)
    return rng.sample(rows, min(n, len(rows)))


def samples(out: sqlite3.Connection, raw: dict) -> dict[str, list[dict]]:
    q = lambda sql: out.execute(sql).fetchall()
    res = {}
    res["restriction"] = [{"course_id": c, "type": t, "value": v, "polarity": p, "hard": h, "method": m,
                           "span": sp, "source": clean(raw[c]["note"])}
                          for c, t, v, p, h, m, sp in _sample(q(
                              "SELECT course_id, type, value, polarity, hard, method, span FROM restriction ORDER BY id"))]
    res["restriction_llm"] = [{"course_id": c, "type": t, "value": v, "polarity": p, "hard": h, "span": sp,
                               "source": clean(raw[c]["note"])}
                              for c, t, v, p, h, sp in _sample(q(
                                  "SELECT course_id, type, value, polarity, hard, span FROM restriction "
                                  "WHERE method='llm' ORDER BY id"), 30)]
    res["eval_item"] = [{"course_id": c, "category": cat, "label": lab, "pct": pct, "method": m,
                         "source": nfkc(raw[c]["evaluation"])[:1500]}
                        for c, cat, lab, pct, m in _sample(q(
                            "SELECT course_id, category, label, pct, method FROM eval_item ORDER BY course_id, seq"))]
    res["weekly"] = [{"course_id": c, "week": w, "date": d, "topic": t, "method": m,
                      "source": nfkc(raw[c]["schedule"])[:2500]}
                     for c, w, d, t, m in _sample(q(
                         "SELECT course_id, week, date, topic, method FROM weekly ORDER BY course_id, week"))]
    res["weekly_llm"] = [{"course_id": c, "week": w, "date": d, "topic": t, "source": nfkc(raw[c]["schedule"])[:2500]}
                         for c, w, d, t in _sample(q(
                             "SELECT course_id, week, date, topic FROM weekly WHERE method='llm' "
                             "ORDER BY course_id, week"), 30)]
    res["alias"] = [{"alias": a, "canonical": c, "source": s, "evidence": e, "confidence": cf, "note": nt}
                    for a, c, s, e, cf, nt in _sample(q(
                        "SELECT alias, canonical, source, evidence, confidence, note FROM alias ORDER BY alias"))]
    res["ai_detail"] = [{"course_id": c, "category": cat, "detail": d, "detail_source": ds,
                         "source": nfkc(raw[c]["ai_policy"])[:600]}
                        for c, cat, d, ds in _sample(q(
                            "SELECT course_id, category, detail, detail_source FROM ai_policy "
                            "WHERE detail IS NOT NULL ORDER BY course_id"))]
    res["prereq"] = [{"course_id": c, "mention": m, "resolved_base": b, "resolved_names": n, "method": me,
                      "source": clean(raw[c]["note"])}
                     for c, m, b, n, me in _sample(q(
                         "SELECT course_id, mention, resolved_base, resolved_names, method FROM prereq "
                         "ORDER BY restriction_id, mention"))]
    return res


def write_samples(res: dict[str, list[dict]], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in res.items():
        with (folder / f"sample_{name}.jsonl").open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"i": i, **r}, ensure_ascii=False) + "\n")


def precision(folder: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(folder.glob("judged_*.jsonl")):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        ok = sum(1 for r in rows if r["ok"])
        out[p.stem.removeprefix("judged_")] = {"ok": ok, "n": len(rows),
                                               "errors": [r for r in rows if not r["ok"]]}
    return out


# ── consistency checks ───────────────────────────────────────────────────────
def _time_en(sessions) -> str:
    """Sessions in the official English form: 三78四CD -> wed16-18,thu12-14; two blocks on one
    day are joined with & (三23456 -> wed09-12&14-16)."""
    inv = {v: k for k, v in _WD.items()}
    days: dict[int, list[str]] = {}
    for s in sessions:
        days.setdefault(s.weekday, []).append(f"{int(s.start[:2]):02d}-{int(s.end[:2]):02d}")
    return ",".join(f"{inv[d]}{'&'.join(v)}" for d, v in days.items())


def surprises(out: sqlite3.Connection, raw: dict) -> dict[str, list]:
    st = {r[0]: r for r in out.execute("SELECT course_id, lang, kind_code, ge_category, teach_sum, eval_pct_sum, "
                                       "cancelled, enroll_count, syllabus_lang, ai_policy FROM course")}
    eng = {c for (c,) in out.execute("SELECT course_id FROM restriction WHERE type='english_taught' AND value='英文'")}
    zh = {c for (c,) in out.execute("SELECT course_id FROM restriction WHERE type='english_taught' AND value='中文'")}
    res: dict[str, list] = {}
    res["note_english_but_lang_not"] = sorted(c for c in eng if st[c][1] != "英文")
    res["note_chinese_but_lang_english"] = sorted(c for c in zh if st[c][1] == "英文")
    res["kind4_without_lmtkind"] = sorted(c for c, v in st.items() if v[2] == 4 and not v[3])
    res["lmtkind_kind_not4"] = sorted(c for c, v in st.items() if v[3] and v[2] != 4)
    res["teach_sum_not_100"] = sorted(c for c, v in st.items() if v[4] is not None and v[4] != 100)
    res["eval_sum_over_105"] = sorted(c for c, v in st.items() if (v[5] or 0) > 105)
    res["cancelled_with_students"] = sorted(c for c, v in st.items() if v[6] and (v[7] or 0) > 0)
    res["lang_en_syllabus_zh"] = sorted(c for c, v in st.items() if v[1] == "英文" and v[8] == "zh")
    res["ai_policy_other"] = sorted(c for c, v in st.items() if v[9] == "other")
    mism = []
    for cid, r in raw.items():
        en = nfkc(r["timeEn"]).lower().replace(" ", "")
        if not en or en == "n/a":
            continue
        mine = _time_en(parse_time_code(r["time"]))
        if mine != en:
            mism.append((cid, r["time"], r["timeEn"], mine))
    res["time_vs_timeEn"] = mism
    dup_w = out.execute("SELECT course_id, week, COUNT(*) FROM weekly GROUP BY 1, 2 HAVING COUNT(*) > 1").fetchall()
    res["weekly_duplicate_week"] = sorted({c for c, _, _ in dup_w})
    dup_r = out.execute("SELECT course_id, type, value, COUNT(*) FROM restriction GROUP BY 1, 2, 3 "
                        "HAVING COUNT(*) > 1").fetchall()
    res["restriction_duplicates"] = [(c, t, v) for c, t, v, _ in dup_r]
    same_name = Counter(n for (n,) in out.execute("SELECT name_base FROM course"))
    res["most_repeated_names"] = same_name.most_common(5)
    return res


def time_en_format_note(raw: dict) -> list[str]:
    """A few timeEn strings, to document the format the round-trip check assumes."""
    return sorted({nfkc(r["timeEn"]) for r in raw.values() if re.search(r"\d", nfkc(r["timeEn"]))})[:5]
