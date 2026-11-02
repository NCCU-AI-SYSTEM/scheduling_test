"""Re-check course_struct.db against the raw COURSE rows.

Coverage of a field = among courses whose raw source carries the information (the
`raw` predicate, written independently of the extractor where possible), the share
whose structured field got a value (the `got` predicate). Reported overall, per
general-education category (+ core) and per college.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

from src.v4.structured.text import TEXTBOOK_TAIL, clean, nfkc, schedule_body

# independent detectors, from docs/v4/03_data_analysis.md 1.7 (the 150 / 142 / ... counts)
NOTE_SIGNALS = {
    "prerequisite": r"先修|擋修|預修|須修過",
    "add_permit": r"加簽",
    "english_taught": r"英語授課|英文授課|全英|EMI",
    "priority": r"優先",
    "not_for_graduation": r"不(計入|列入|採計).{0,8}畢業學分",
    "first_class_mandatory": r"第一(堂|次|週).{0,8}(務必|必須)",
    "college_only": r"學院修讀",
    "intensive": r"密集|集中授課|隔週",
    "status_only": r"交換生|外籍(學位)?生",
    "no_repeat": r"不得重複|請勿重複",
    "grade_min": r"限大[一二三四]|[一二三四]年級以上",
    "common_exam": r"會考",
    "fee": r"收費",
    "joint_offering": r"合開",
    "location": r"上課地點|集合地點",
    "online": r"遠距|非同步|線上",
}


@dataclass(frozen=True)
class Field:
    name: str
    raw: Callable[[dict], bool]
    got: Callable[[dict], bool]
    note: str = ""


def _has(s) -> bool:
    return bool(nfkc(s)) and nfkc(s) not in ("N/A",)


def _weekly_table(schedule: str | None) -> bool:
    """>= 6 week markers in order (1|, 第2週, W03, Week 4) or >= 6 M/D dates: a week-by-week table."""
    body = schedule_body(schedule)
    nums = [int(x) for x in re.findall(r"(?:^|\|)\s*(?:第|W|Week\s*|Session\s*)?0?(\d{1,2})\s*(?:週|周)?\s*(?=\||$)", body)]
    run, best = 0, 0
    for n in nums:
        run = n if n == run + 1 else (1 if n == 1 else run)
        best = max(best, run)
    return best >= 6 or len(re.findall(r"(?<![\d/])\d{1,2}/\d{1,2}(?![\d/])", body)) >= 6


def fields() -> list[Field]:
    f = [
        Field("sessions", lambda r: bool(re.search(r"[一二三四五六日][AB1-8C-H]", r["time"] or "")),
              lambda s: s["n_main_sessions"] > 0),
        Field("extra_sessions", lambda r: bool(re.search(r"(實習課|實驗課|討論課)", r["note"] or "")
                                               and re.search(r"[一二三四五六日]\s*[、,]?\s*[AB1-8C-H]|第[A-H1-8]節|\d{1,2}[:：]\d{2}", r["note"] or "")),
              lambda s: s["n_note_sessions"] > 0,
              "note names a lab or discussion with a day/period or clock time; got counts sessions the time code already holds"),
        Field("target_class", lambda r: bool(re.search(r"[一二三四五六七1-7]", r["unit"] or "")),
              lambda s: s["n_targets"] > 0, "unit contains a year character"),
        Field("dept_name", lambda r: bool(r["subNum"]), lambda s: bool(s["dept_name"])),
        Field("college", lambda r: any(d in r["dp1s"] for d in ("100", "200", "300", "400", "500", "600", "700",
                                                                  "800", "900", "Z23", "ZA0")),
              lambda s: bool(s["college"]), "course has a college row in dp1"),
        Field("ge_category", lambda r: _has(r["lmtKind"]), lambda s: bool(s["ge_category"])),
        Field("teaching", lambda r: "%" in (r["teaching_approach"] or ""), lambda s: s["teach_sum"] == 100,
              "got = the five percents add up to 100"),
        Field("ai_policy", lambda r: _has(r["ai_policy"]), lambda s: s["ai_policy"] not in (None, "other")),
        Field("ai_detail", lambda r: nfkc(r["ai_policy"]).startswith(("有條件開放使用", "禁止使用")),
              lambda s: bool(s["ai_detail"]), "conditional / prohibited courses with a detail text"),
        Field("restriction", lambda r: clean(r["note"]) is not None, lambda s: s["n_restr"] > 0,
              "note has content"),
        Field("change", lambda r: clean(r["info"]) is not None, lambda s: s["n_change"] > 0),
        Field("eval_item", lambda r: _has(r["evaluation"]), lambda s: s["n_eval"] > 0),
        Field("eval_pct", lambda r: bool(re.search(r"\d\s*[%％﹪]", r["evaluation"] or "")),
              lambda s: bool(s["eval_reliable"]), "got = items sum to 95-105"),
        Field("weekly", lambda r: _weekly_table(r["schedule"]), lambda s: s["n_weeks"] > 0,
              "schedule is a week-by-week table (independent marker / date count)"),
        Field("weekly_any", lambda r: len(schedule_body(r["schedule"])) >= 150, lambda s: s["n_weeks"] > 0,
              "schedule has >= 150 characters"),
        Field("syllabus", lambda r: _has(r["syllabus"]), lambda s: bool(s["syllabus"])),
        Field("objective", lambda r: _has(r["objective"]), lambda s: bool(s["objective"])),
        Field("schedule_clean", lambda r: _has(schedule_body(r["schedule"])), lambda s: bool(s["schedule_clean"])),
        Field("evaluation", lambda r: _has(r["evaluation"]), lambda s: bool(s["evaluation"])),
        Field("textbook_clean", lambda r: bool(TEXTBOOK_TAIL.sub("", nfkc(r["textbook"])).strip(" |")),
              lambda s: bool(s["textbook_clean"])),
        Field("syllabus_lang", lambda r: len(nfkc(r["syllabus"]) + nfkc(r["objective"])) >= 10,
              lambda s: bool(s["syllabus_lang"])),
    ]
    for t, pat in NOTE_SIGNALS.items():
        f.append(Field(f"note:{t}", lambda r, p=pat: bool(re.search(p, nfkc(r["note"]))),
                       lambda s, t=t: t in s["restr_types"], f"note matches /{pat}/; got = a {t} row"))
        f.append(Field(f"note~{t}", lambda r, p=pat: bool(re.search(p, nfkc(r["note"]))),
                       lambda s, p=pat: any(re.search(p, sp or "") for sp in s["restr_spans"]),
                       "got = some restriction row (any type) whose span holds the matched words"))
    return f


def load(src: sqlite3.Connection, out: sqlite3.Connection) -> tuple[dict, dict]:
    raw: dict[str, dict] = {}
    label = dict(out.execute("SELECT key, value FROM meta")).get("semester", "114-2")
    year, term = label.split("-")
    src.row_factory = sqlite3.Row
    for r in src.execute("SELECT * FROM COURSE WHERE y = ? AND s = ? ORDER BY id, dp1, dp2, dp3", (year, term)):
        d = raw.setdefault(r["id"], dict(r) | {"dp1s": set()})
        d["dp1s"].add(r["dp1"])
    src.row_factory = None
    out.row_factory = sqlite3.Row
    st = {r["course_id"]: dict(r) for r in out.execute("SELECT * FROM course")}
    out.row_factory = None
    count = lambda sql: Counter(dict(out.execute(sql).fetchall()))
    mains = count("SELECT course_id, COUNT(*) FROM session WHERE source='time' GROUP BY 1")
    notes = count("SELECT course_id, COUNT(*) FROM session WHERE source='note' GROUP BY 1")
    tg = count("SELECT course_id, COUNT(*) FROM target_class GROUP BY 1")
    rs = count("SELECT course_id, COUNT(*) FROM restriction GROUP BY 1")
    ch = count("SELECT course_id, COUNT(*) FROM change GROUP BY 1")
    ev = count("SELECT course_id, COUNT(*) FROM eval_item GROUP BY 1")
    types: dict[str, set] = defaultdict(set)
    spans: dict[str, list] = defaultdict(list)
    for cid, t, sp in out.execute("SELECT course_id, type, span FROM restriction"):
        types[cid].add(t)
        spans[cid].append(sp)
    ai = dict(out.execute("SELECT course_id, detail FROM ai_policy").fetchall())
    for cid, s in st.items():
        s.update(n_main_sessions=mains[cid], n_note_sessions=notes[cid], n_targets=tg[cid], n_restr=rs[cid],
                 n_change=ch[cid], n_eval=ev[cid], restr_types=types[cid], restr_spans=spans[cid],
                 ai_detail=ai.get(cid))
    return raw, st


def group_of(s: dict) -> tuple[str, str]:
    return s["ge_category"] or "非通識", s["college"] or "非學院單位"


def coverage(raw: dict, st: dict) -> dict:
    """{field: {"all": (hit, n), "ge": {cat: (hit, n)}, "college": {...}, "misses": [course ids]}}."""
    out = {}
    for f in fields():
        tot = [0, 0]
        ge: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        col: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        misses = []
        for cid, r in raw.items():
            if not f.raw(r):
                continue
            s = st[cid]
            hit = int(f.got(s))
            g, c = group_of(s)
            for bucket in (tot, ge[g], col[c]):
                bucket[0] += hit
                bucket[1] += 1
            if s["ge_core"]:
                ge["核心通識"][0] += hit
                ge["核心通識"][1] += 1
            if not hit:
                misses.append(cid)
        out[f.name] = {"all": tuple(tot), "ge": {k: tuple(v) for k, v in ge.items()},
                       "college": {k: tuple(v) for k, v in col.items()}, "misses": misses, "note": f.note}
    return out
