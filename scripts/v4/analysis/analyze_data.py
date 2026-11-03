"""v4 data analysis for the NCCU course DB and the student transcript export.

Read-only. Produces the numbers behind docs/v4/03_data_analysis.md.

    python3 scripts/v4/analysis/analyze_data.py                    # human-readable report
    python3 scripts/v4/analysis/analyze_data.py --json out.json    # also dump all stats
    python3 scripts/v4/analysis/analyze_data.py --transcript PATH  # structure-only transcript stats

The transcript section never prints names, student numbers, grades or other
identifying values: only key names, types, counts and match rates.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = Path("/Users/pengqize/Documents/code/NCCU-AI-SYSTEM/NCCUCourse/1142.db")
DEFAULT_UNIT_JSON = DEFAULT_DB.parent / "data" / "unit.json"
DEFAULT_TRANSCRIPT = Path(
    "/Users/pengqize/Documents/code/NCCU-AI-SYSTEM/scheduling_test/data/exportStudentData.json"
)

# ── period table ─────────────────────────────────────────────────────────────
# Verified two ways: (1) COURSE.timeEn (official English time string, e.g.
# 五D56 -> fri13-16) for all 650 distinct time strings, (2) a department
# timetable PDF (1 = 08:10-09:00 ... G = 20:10-21:00). Each period is 50 min,
# starting at hh:10 and ending at (hh+1):00.
PERIODS = ["A", "B", "1", "2", "3", "4", "C", "D", "5", "6", "7", "8", "E", "F", "G", "H"]
PERIOD_START = {p: 6 + i for i, p in enumerate(PERIODS)}  # hour of hh:10 start
WEEKDAYS = "一二三四五六日"
EN_DAY = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}
TBA = "未定或彈性"

EMPTY_MARKERS = {"＠備註:無", "＠異動資訊:無", "N/A", "＠Note:N/A", "＠Information of alteration:N/A"}

COLLEGE_DP1 = {"100", "200", "300", "400", "500", "600", "700", "800", "900", "Z23", "ZA0", "ZC0"}

TEXT_FIELDS = ("syllabus", "objective", "schedule", "evaluation", "textbook",
               "teaching_approach", "ai_policy")

# ── note (備註) restriction patterns ─────────────────────────────────────────
NOTE_PATTERNS = {
    "限特定系所/學院修讀": r"(僅限|限|只限|只開放|僅開放)[^，。；;]{0,20}(系|所|學院|學程|專班|ETP|IMBA)[^，。；;]{0,4}(學生)?(修讀|修習|選修|選讀|修課)?",
    "指定學院修讀(通識/外文分院)": r"學院修讀|^＠備註:[^，。；]{0,20}學院$",
    "某系所優先": r"優先",
    "年級限制(限大X以上/限碩博)": r"限\s*(大|碩|博)\s*[一二三四]|限\s*[一二三四1-4]\s*年級|大[一二三四]\s*以上|[一二三四1-4]\s*年級以上|限(碩|博)(士)?(班)?(生|學生)|限(大學部|學士班)",
    "不得重複修習/已修者勿選": r"不得重複|請勿重複|勿重複|重複修(習|課)|已修(過|習|畢)?[^，。；]{0,12}(者)?(不得|勿|請勿|不可)",
    "先修/擋修": r"先修|擋修|預修|須(先)?修(過|畢)|需(先)?修(過|畢)|修畢[^，。；]{0,10}(始|方)可",
    "英語授課": r"英語授課|英文授課|全英|EMI|English[- ]?taught|taught in English",
    "第一堂課必到": r"第[一1]\s*(堂|次|週|周)[^，。；]{0,8}(務必|必須|須|一定要|請務必)[^，。；]{0,4}(出席|到課|到)|第[一1]\s*(堂|次)課[^，。；]{0,6}未(到|出席)",
    "加簽規則": r"加簽",
    "不計入畢業學分": r"不(計入|列入|採計|予以?採計|算入)[^，。；]{0,8}畢業學分|學分不予以?採計",
    "合開(學碩/碩博/跨系)": r"合開",
    "灌檔(系統自動加選)": r"灌檔",
    "上課地點/教室說明": r"上課地點|集合地點|教室[：:]|請洽",
    "密集/集中授課": r"密集|集中授課|彈性授課|週末上課|隔週",
    "收費課程": r"收費",
    "外籍生/交換生專班": r"交換生|外籍(學位)?生|國際生",
    "雙主修/輔系相關": r"雙主修|輔系",
    "通過/不通過制": r"通過\s*/\s*不通過|pass\s*/\s*fail",
    "人數上限": r"限\s*\d+\s*(人|名)|人數上限|名額",
    "期末會考/統一考試": r"會考|統一考試",
    "額外實習/實驗課時段": r"實習課|實驗課|習作課|TA課",
}

ONLINE_PATTERNS = r"遠距|線上|同步|非同步|視訊|Teams|Google\s*Meet|Webex|Zoom|磨課師|MOOC|online|Online"

EVAL_ASPECTS = {
    "期中考": r"期中|midterm|mid-term",
    "期末考": r"期末考|期末測驗|final\s*exam|final examination",
    "小考": r"小考|quiz",
    "作業": r"作業|homework|assignment",
    "報告/論文": r"報告|report|paper|essay|論文",
    "專題/期末專案": r"專題|project|專案",
    "簡報/口頭": r"簡報|presentation|口頭|口報",
    "出席/參與": r"出席|出缺席|點名|參與|attendance|participation",
    "分組/小組": r"分組|小組|團體|group|team",
    "討論": r"討論|discussion",
}

RATE_KEYWORDS = ["作業", "報告", "考試", "期中", "期末", "點名", "分組", "給分", "甜", "涼",
                 "硬", "推", "不推", "重", "輕鬆", "收穫", "認真", "簡報", "小考", "出席",
                 "loading", "好過", "難", "無聊", "有趣", "實用"]


# ── helpers ──────────────────────────────────────────────────────────────────
def connect(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def quantiles(vals: list[int]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    pick = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return {"n": len(s), "p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9), "max": s[-1],
            "mean": round(statistics.mean(s), 1)}


def is_filled(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return bool(s) and s not in EMPTY_MARKERS


def english_ratio(text: str) -> float:
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = len(re.findall(r"[一-鿿]", text))
    return latin / (latin + cjk) if latin + cjk else 0.0


def load_semester(con, y: str, s: str) -> list[dict]:
    """One dict per course id; dp triples collected into `dps`."""
    out: dict[str, dict] = {}
    for r in con.execute("SELECT * FROM COURSE WHERE y=? AND s=? ORDER BY id, dp1, dp2, dp3", (y, s)):
        d = out.get(r["id"])
        if d is None:
            d = dict(r)
            d["dps"] = []
            out[r["id"]] = d
        d["dps"].append((r["dp1"], r["dp2"], r["dp3"]))
    return list(out.values())


# ── time ─────────────────────────────────────────────────────────────────────
def parse_time(time_str: str) -> list[tuple[int, int, int]]:
    """'一23四56' -> [(1, 9, 11), (4, 14, 16)] as (weekday, start_hour, end_hour).

    start_hour is the hh of the hh:10 start; end_hour is the (hh+1):00 end.
    Periods are contiguous only in PERIODS order, so 三23456 splits at 4|5
    (C and D sit between them), matching the official timeEn.
    """
    if not time_str or time_str.strip() == TBA:
        return []
    out, day, run = [], None, []

    def flush():
        if day is not None and run:
            out.append((day, PERIOD_START[run[0]], PERIOD_START[run[-1]] + 1))

    for ch in time_str:
        if ch in WEEKDAYS:
            flush()
            day, run = WEEKDAYS.index(ch) + 1, []
        elif ch in PERIOD_START:
            if run and PERIODS.index(ch) != PERIODS.index(run[-1]) + 1:
                flush()
                run = []
            run.append(ch)
    flush()
    return out


def parse_time_en(time_en: str) -> list[tuple[int, int, int]]:
    out = []
    for part in (time_en or "").split(","):
        m = re.match(r"([a-z]{3})(.*)", part.strip())
        if not m or m.group(1) not in EN_DAY:
            continue
        for a, b in re.findall(r"(\d\d)-(\d\d)", m.group(2)):
            out.append((EN_DAY[m.group(1)], int(a), int(b)))
    return sorted(out)


def analyze_time(con, courses) -> dict:
    sym = collections.Counter(ch for c in courses for ch in c["time"])
    all_sym = collections.Counter()
    distinct = con.execute("SELECT DISTINCT time, timeEn FROM COURSE").fetchall()
    mism = []
    for r in distinct:
        for ch in r["time"] or "":
            all_sym[ch] += 1
        if r["time"] == TBA:
            continue
        if sorted(parse_time(r["time"])) != parse_time_en(r["timeEn"]):
            mism.append((r["time"], r["timeEn"]))
    sess = [parse_time(c["time"]) for c in courses]
    hours = [sum(e - s for _, s, e in x) for x in sess]
    return {
        "symbols_1142": dict(sym.most_common()),
        "distinct_time_strings_all": len(distinct),
        "parser_vs_timeEn_mismatch": mism,
        "tba": sum(1 for c in courses if c["time"] == TBA),
        "sessions_per_course": dict(collections.Counter(len(x) for x in sess)),
        "weekly_hours": dict(collections.Counter(hours)),
        "has_weekend": sum(1 for x in sess if any(d >= 6 for d, _, _ in x)),
        "has_evening_E_to_H": sum(1 for x in sess if any(e > 18 for _, _, e in x)),
        "has_lunch_C": sum(1 for c in courses if "C" in c["time"]),
        "has_early_AB": sum(1 for c in courses if re.search("[AB]", c["time"])),
        "split_same_day": sum(1 for x in sess if len({d for d, _, _ in x}) < len(x)),
    }


# ── unit / department ───────────────────────────────────────────────────────
def load_unit_tree(path: Path) -> dict:
    tree = json.loads(path.read_text())
    names = {}
    for l1 in tree:
        if l1["utCodL1"] == "0":
            continue
        names[(l1["utCodL1"],)] = l1["utL1Text"]
        for l2 in l1["utL2"]:
            if l2["utCodL2"] == "0":
                continue
            names[(l1["utCodL1"], l2["utCodL2"])] = l2["utL2Text"]
            for l3 in l2["utL3"]:
                if l3["utCodL3"] != "0":
                    names[(l1["utCodL1"], l2["utCodL2"], l3["utCodL3"])] = l3["utL3Text"]
    return names


GRADE_CH = "一二三四五六"
CLASS_TOKEN = re.compile(r"(碩專|博專|碩|博|專)?([一二三四五六1-6])([甲乙丙丁A-Z])?")


def parse_unit(unit: str) -> dict:
    """Class-list string -> degree levels and years, e.g. 法律三甲法律三乙 -> 學士 {3}."""
    degrees, years = set(), set()
    for m in CLASS_TOKEN.finditer(unit):
        lvl = {"碩": "碩士", "博": "博士", "專": "在職專班", "碩專": "在職專班", "博專": "博士"}.get(m.group(1) or "", "學士")
        y = m.group(2)
        years.add((lvl, GRADE_CH.index(y) + 1 if y in GRADE_CH else int(y)))
        degrees.add(lvl)
    if "博士" in unit:
        degrees.add("博士")
    if "碩士" in unit:
        degrees.add("碩士")
    if "學士" in unit and not years:
        degrees.add("學士")
    return {"is_class_list": bool(years), "degrees": sorted(degrees), "years": sorted(years)}


def analyze_units(courses, unit_names) -> dict:
    units = collections.Counter(c["unit"] for c in courses)
    parsed = {u: parse_unit(u) for u in units}
    class_list = [u for u, p in parsed.items() if p["is_class_list"]]
    dp1 = collections.Counter(c["dps"][0][0] for c in courses)
    college_rows = [c for c in courses if c["dps"][0][0] in COLLEGE_DP1]
    prefix_eq = sum(1 for c in college_rows if any(c["subNum"][:3] == dp[2] for dp in c["dps"]))
    # home department via subNum prefix: majority dp3/dp1 seen for that prefix in college rows
    pre_map = collections.defaultdict(collections.Counter)
    for c in college_rows:
        for dp in c["dps"]:
            if dp[0] in COLLEGE_DP1:
                pre_map[c["subNum"][:3]][(dp[0], dp[2])] += 1
    resolvable = sum(1 for c in courses if c["subNum"][:3] in pre_map or c["dps"][0][0] in COLLEGE_DP1)
    dp_named = sum(1 for c in courses if all(dp in unit_names for dp in c["dps"]))
    target_grade = sum(1 for c in courses if parsed[c["unit"]]["is_class_list"])
    return {
        "n_units": len(units),
        "n_units_class_list": len(class_list),
        "n_units_plain_name": len(units) - len(class_list),
        "courses_with_class_list_unit": target_grade,
        "rows_per_course": dict(collections.Counter(len(c["dps"]) for c in courses)),
        "dp1_first": dict(dp1.most_common()),
        "college_courses": len(college_rows),
        "subNum_prefix_equals_dp3_in_college_rows": prefix_eq,
        "home_dept_resolvable_by_prefix_or_dp": resolvable,
        "all_dp_triples_named_in_unit_json": dp_named,
        "examples_plain": [u for u, _ in units.most_common() if not parsed[u]["is_class_list"]][:25],
        "examples_class": class_list[:15],
    }


# ── note / info / classroom ─────────────────────────────────────────────────
def analyze_note(courses) -> dict:
    notes = [c["note"] for c in courses]
    filled = [n for n in notes if is_filled(n)]
    counts = {k: sum(1 for n in filled if re.search(p, n, re.I)) for k, p in NOTE_PATTERNS.items()}
    any_hit = sum(1 for n in filled if any(re.search(p, n, re.I) for p in NOTE_PATTERNS.values()))
    info = [c["info"] for c in courses]
    info_cnt = {k: sum(1 for x in info if k in x) for k in ("停開", "教室異動", "時間異動", "教師異動")}
    return {"note_filled": len(filled), "note_distinct": len(set(filled)), "patterns": counts,
            "note_filled_matching_any_pattern": any_hit, "info": info_cnt}


def analyze_location(courses) -> dict:
    cls = collections.Counter()
    online = 0
    for c in courses:
        room = c["classroom"] or ""
        if not room:
            cls["空白"] += 1
        elif "洽" in room:
            cls["教室洽系所"] += 1
        elif re.search(r"\d{6}$", room):
            cls["大樓名+6碼教室代號"] += 1
        else:
            cls["其他文字(講堂/廳/電腦室)"] += 1
        blob = " ".join([room, c["note"] or "", c["unit"]])
        if re.search(ONLINE_PATTERNS, blob):
            online += 1
    syl_online = sum(1 for c in courses if re.search(r"遠距(教學|課程|授課)|線上(授課|上課|課程)|非同步", " ".join(c[f] or "" for f in TEXT_FIELDS)))
    return {"classroom": dict(cls), "online_signal_room_note_unit": online, "online_signal_syllabus": syl_online,
            "building_prefixes": dict(collections.Counter(re.sub(r"\d.*$", "", c["classroom"] or "") for c in courses).most_common(15))}


# ── syllabus sections ───────────────────────────────────────────────────────
TA_PAIR = re.compile(r"(\d{1,3})?%\|([^|]+)\|([^|]*)")


def parse_teaching_approach(ta: str) -> dict:
    out = {}
    for m in TA_PAIR.finditer(ta or ""):
        if m.group(1) is not None:
            out[m.group(3).strip() or m.group(2).strip()] = int(m.group(1))
    return out


PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*[%％]")


def parse_eval_breakdown(ev: str) -> list[tuple[str, float]]:
    """Assign each percentage to the aspect keyword nearest before it (60-char window)."""
    out = []
    for m in PCT.finditer(ev or ""):
        window = ev[max(0, m.start() - 60):m.start()]
        best, pos = "其他", -1
        for name, pat in EVAL_ASPECTS.items():
            for k in re.finditer(pat, window, re.I):
                if k.start() > pos:
                    best, pos = name, k.start()
        out.append((best, float(m.group(1))))
    return out


def analyze_sections(courses, weekly_fn=None) -> dict:
    res = {}
    for f in TEXT_FIELDS:
        vals = [c[f] or "" for c in courses]
        filled = [v for v in vals if v.strip()]
        res[f] = {
            "filled": len(filled),
            "len": quantiles([len(v) for v in filled]),
            "pipes_p50": quantiles([v.count("|") for v in filled]).get("p50"),
            "mostly_english(>0.7)": sum(1 for v in filled if english_ratio(v) > 0.7),
            "mixed(0.2-0.7)": sum(1 for v in filled if 0.2 <= english_ratio(v) <= 0.7),
            "distinct": len(set(filled)),
        }
    sch = [c["schedule"] or "" for c in courses]
    res["schedule_prefix_boilerplate"] = sum(1 for v in sch if v.startswith("每周課程進度與作業要求|Course Schedule & Requirements|"))
    res["schedule_dates>=8"] = sum(1 for v in sch if len(re.findall(r"(?<!\d)\d{1,2}/\d{1,2}(?!\d)", v)) >= 8)
    res["schedule_week_markers>=8"] = sum(1 for v in sch if len(re.findall(r"第\s*\|?\s*\d+\s*\|?\s*[週周]|W\d{1,2}|Week\s*\d+", v, re.I)) >= 8)
    res["schedule_has_workload_column"] = sum(1 for v in sch if re.search(r"學習投入時間|Outside-of-class|workload", v, re.I))
    if weekly_fn:
        res["loader_weekly_topics>=8"] = sum(1 for c in courses if len(weekly_fn(c["syllabus"] or "", c["schedule"] or "")) >= 8)
        res["loader_weekly_topics==0"] = sum(1 for c in courses if not weekly_fn(c["syllabus"] or "", c["schedule"] or ""))
    tb = [c["textbook"] or "" for c in courses]
    res["textbook_library_boilerplate"] = sum(1 for v in tb if "已申請之圖書館指定參考書目" in v)
    res["textbook_self_made_only"] = sum(1 for v in tb if re.search(r"自編|講義|投影片|slides", v, re.I))
    res["textbook_isbn"] = sum(1 for v in tb if re.search(r"ISBN|97[89]\d{10}", v))
    # teaching approach
    ta = [parse_teaching_approach(c["teaching_approach"]) for c in courses]
    ok = [t for t in ta if t and 95 <= sum(t.values()) <= 105]
    res["teaching_approach_parsed_sum100"] = len(ok)
    res["teaching_approach_lecture>=70"] = sum(1 for t in ok if t.get("Lecture", 0) >= 70)
    res["teaching_approach_group>=20"] = sum(1 for t in ok if t.get("Group activity", 0) >= 20)
    # ai policy
    ai = collections.Counter()
    for c in courses:
        a = c["ai_policy"] or ""
        k = ("未填" if not a.strip() else "無涉及AI" if "無涉及" in a else "完全開放" if "完全開放" in a
             else "有條件開放" if "有條件" in a else "禁止" if "禁止" in a else "其他")
        ai[k] += 1
    res["ai_policy"] = dict(ai)
    # evaluation
    ev = [parse_eval_breakdown(c["evaluation"]) for c in courses]
    sums = [sum(p for _, p in e) for e in ev]
    res["eval_has_pct"] = sum(1 for e in ev if e)
    res["eval_pct_sum_95_105"] = sum(1 for s in sums if 95 <= s <= 105)
    res["eval_pct_sum_gt105"] = sum(1 for s in sums if s > 105)
    labelled = [e for e, s in zip(ev, sums) if 95 <= s <= 105]
    res["eval_all_pct_labelled(no 其他)"] = sum(1 for e in labelled if all(k != "其他" for k, _ in e))
    asp = collections.Counter()
    for c in courses:
        txt = c["evaluation"] or ""
        for name, pat in EVAL_ASPECTS.items():
            if re.search(pat, txt, re.I):
                asp[name] += 1
    res["eval_aspect_mentions"] = dict(asp.most_common())
    res["eval_has_any_exam"] = sum(1 for c in courses if re.search(EVAL_ASPECTS["期中考"] + "|" + EVAL_ASPECTS["期末考"] + "|考試|exam", c["evaluation"] or "", re.I))
    res["eval_no_exam_explicit"] = sum(1 for c in courses if re.search(r"無(期中|期末)?考試|不考試|沒有考試|no (midterm|final|exam)", " ".join([c["evaluation"] or "", c["schedule"] or ""]), re.I))
    res["eval_attendance_rule"] = sum(1 for c in courses if re.search(r"缺(席|課)\s*(超過|達|滿|逾)?\s*\d+\s*次|扣\s*\d+\s*分|點名", c["evaluation"] or ""))
    return res


# ── derived-field coverage ──────────────────────────────────────────────────
PREREQ_TEXT = re.compile(r"先修(科目|課程)?|擋修|預修|prerequisite|pre-requisite|修過[^，。；]{0,15}(者|為佳|尤佳)", re.I)


def analyze_derived(courses, unit_names) -> dict:
    n = len(courses)
    note_or = lambda c: (c["note"] or "") if is_filled(c["note"]) else ""
    cov = {}
    cov["sessions(可排時間)"] = sum(1 for c in courses if parse_time(c["time"]))
    college_prefixes = {c["subNum"][:3] for c in courses if c["dps"][0][0] in COLLEGE_DP1}
    cov["department_code(學院列 dp3，或 subNum 前綴可對回學院列)"] = sum(
        1 for c in courses if any(dp[0] in COLLEGE_DP1 for dp in c["dps"]) or c["subNum"][:3] in college_prefixes)
    cov["college(dp1 學院碼)"] = sum(1 for c in courses if any(dp[0] in COLLEGE_DP1 for dp in c["dps"]))
    cov["degree_level(dp2=1/2/3)"] = sum(1 for c in courses if any(dp[1] in ("1", "2", "3") for dp in c["dps"]))
    cov["target_grade(unit 班級字串)"] = sum(1 for c in courses if parse_unit(c["unit"])["is_class_list"])
    cov["target_grade(unit 或 note 年級限制)"] = sum(1 for c in courses if parse_unit(c["unit"])["is_class_list"] or re.search(NOTE_PATTERNS["年級限制(限大X以上/限碩博)"], note_or(c)))
    cov["restrictions_list(note 命中任一限制類)"] = sum(1 for c in courses if any(re.search(NOTE_PATTERNS[k], note_or(c), re.I) for k in ("限特定系所/學院修讀", "指定學院修讀(通識/外文分院)", "年級限制(限大X以上/限碩博)", "不得重複修習/已修者勿選", "先修/擋修", "外籍生/交換生專班", "雙主修/輔系相關")))
    cov["prerequisite_mentioned(note)"] = sum(1 for c in courses if re.search(NOTE_PATTERNS["先修/擋修"], note_or(c)))
    cov["prerequisite_mentioned(note+大綱)"] = sum(1 for c in courses if PREREQ_TEXT.search(note_or(c) + " ".join(c[f] or "" for f in ("syllabus", "objective", "schedule"))))
    cov["evaluation_breakdown(%加總≈100)"] = sum(1 for c in courses if 95 <= sum(p for _, p in parse_eval_breakdown(c["evaluation"])) <= 105)
    cov["teaching_approach(%加總≈100)"] = sum(1 for c in courses if 95 <= sum(parse_teaching_approach(c["teaching_approach"]).values()) <= 105)
    cov["english_taught(lang=英文)"] = sum(1 for c in courses if c["lang"] == "英文")
    cov["english_taught(lang 或 note 英語授課)"] = sum(1 for c in courses if c["lang"] == "英文" or re.search(NOTE_PATTERNS["英語授課"], note_or(c), re.I))
    cov["syllabus_mostly_english"] = sum(1 for c in courses if english_ratio(" ".join(c[f] or "" for f in ("syllabus", "objective"))) > 0.7)
    cov["online_flag(有訊號)"] = sum(1 for c in courses if re.search(ONLINE_PATTERNS, " ".join([c["classroom"] or "", note_or(c), c["unit"]])))
    cov["cancelled(info 停開)"] = sum(1 for c in courses if "停開" in (c["info"] or ""))
    cov["workload_signal(作業/報告/考試 任一出現在評量)"] = sum(1 for c in courses if re.search(r"作業|報告|考|homework|assignment|report|exam", c["evaluation"] or "", re.I))
    cov["general_ed_category(lmtKind)"] = sum(1 for c in courses if c["lmtKind"])
    return {"n": n, "coverage": {k: (v, pct(v, n)) for k, v in cov.items()}}


# ── cross-semester ──────────────────────────────────────────────────────────
def analyze_linkage(con, courses) -> dict:
    past = collections.defaultdict(set)
    name_teacher = collections.defaultdict(set)
    names = collections.defaultdict(set)
    for r in con.execute("SELECT DISTINCT id, y, s, subNum, name, teacher FROM COURSE WHERE NOT (y='114' AND s='2')"):
        past[r["subNum"]].add(r["id"])
        name_teacher[(r["name"], r["teacher"])].add(r["id"])
        names[r["name"]].add(r["id"])
    rate_ids = collections.Counter(r[0] for r in con.execute("SELECT courseId FROM RATE"))
    course_ids = {r[0] for r in con.execute("SELECT DISTINCT id FROM COURSE")}
    by_sub = sum(1 for c in courses if past.get(c["subNum"]))
    by_nt = sum(1 for c in courses if name_teacher.get((c["name"], c["teacher"])))
    by_name = sum(1 for c in courses if names.get(c["name"]))
    rate_sub = [sum(rate_ids[i] for i in past.get(c["subNum"], ())) for c in courses]
    rate_nt = [sum(rate_ids[i] for i in name_teacher.get((c["name"], c["teacher"]), ())) for c in courses]
    rate_any = [max(a, b) for a, b in zip(rate_sub, rate_nt)]
    # does subNum keep the same name/teacher across semesters?
    sub_stats = con.execute("""SELECT COUNT(*), SUM(nn>1), SUM(nt>1) FROM (
        SELECT subNum, COUNT(DISTINCT name) nn, COUNT(DISTINCT teacher) nt FROM COURSE GROUP BY subNum HAVING COUNT(DISTINCT y||s)>1)""").fetchone()
    # same course name+teacher appears under several subNums across semesters
    nt_multi_sub = con.execute("""SELECT COUNT(*) FROM (SELECT name, teacher FROM COURSE
        GROUP BY name, teacher HAVING COUNT(DISTINCT subNum)>1)""").fetchone()[0]
    return {
        "linked_by_subNum": by_sub, "linked_by_name_teacher": by_nt, "linked_by_name": by_name,
        "with_rate_via_subNum": sum(1 for x in rate_sub if x), "with_rate_via_name_teacher": sum(1 for x in rate_nt if x),
        "with_rate_any": sum(1 for x in rate_any if x), "rate_per_linked_course": quantiles([x for x in rate_any if x]),
        "rate_courseIds_in_COURSE": pct(sum(1 for i in rate_ids if i in course_ids), len(rate_ids)),
        "multi_sem_subNums": sub_stats[0], "subNum_name_changed": sub_stats[1], "subNum_teacher_changed": sub_stats[2],
        "name_teacher_with_multiple_subNums": nt_multi_sub,
    }


def analyze_rate(con) -> dict:
    contents = [r[0] or "" for r in con.execute("SELECT content FROM RATE")]
    n = len(contents)
    kw = {k: pct(sum(1 for c in contents if k in c), n) for k in RATE_KEYWORDS}
    trivial = sum(1 for c in contents if len(c.strip()) <= 4)
    dup = collections.Counter(c.strip() for c in contents)
    per_course = [r[0] for r in con.execute("SELECT COUNT(*) FROM RATE GROUP BY courseId")]
    by_sem = {r[0]: r[1] for r in con.execute("SELECT substr(courseId,1,4), COUNT(*) FROM RATE GROUP BY 1")}
    return {"n": n, "len": quantiles([len(c) for c in contents]), "keyword_pct": kw,
            "trivial_len<=4": trivial, "top_duplicates": dup.most_common(15),
            "per_course": quantiles(per_course), "by_semester": by_sem,
            "contentEn_filled": con.execute("SELECT COUNT(*) FROM RATE WHERE contentEn!=''").fetchone()[0]}


def analyze_result(con, courses) -> dict:
    by_sem = {r[0]: r[1] for r in con.execute("SELECT yearsem, COUNT(*) FROM RESULT GROUP BY 1")}
    rows = con.execute("SELECT courseId, studentLimit, studentCount, lastEnroll FROM RESULT WHERE yearsem='1142'").fetchall()
    ids = {c["id"] for c in courses}
    lim = [r for r in rows if r[1] and r[1] > 0]
    ratio = [r[2] / r[1] for r in lim]
    old = con.execute("SELECT name FROM RESULT WHERE yearsem!='1142'").fetchall()
    old_kind = collections.Counter("體育" if "體育" in r[0] else "其他" for r in old)
    return {"by_semester": by_sem, "1142_rows": len(rows), "1142_match_course_ids": sum(1 for r in rows if r[0] in ids),
            "1142_limit_gt0": len(lim), "1142_limit_0": len(rows) - len(lim),
            "fill_ratio": quantiles([int(100 * x) for x in ratio]),
            "full(>=100%)": sum(1 for x in ratio if x >= 1), "over(>100%)": sum(1 for x in ratio if x > 1),
            "under_50%": sum(1 for x in ratio if x < 0.5), "count_0": sum(1 for r in rows if not r[2]),
            "lastEnroll_negative": sum(1 for r in rows if (r[3] or 0) < 0),
            "older_sem_rows_PE_share": dict(old_kind)}


# ── transcript (structure only) ─────────────────────────────────────────────
def describe_type(v) -> str:
    if isinstance(v, str):
        if v == "":
            return "str(empty)"
        if re.fullmatch(r"\d+", v):
            return f"str(digits,len{len(v)})"
        if re.fullmatch(r"[\d.]+", v):
            return "str(numeric)"
        return "str"
    return type(v).__name__


def transcript_schema(v, path="$", out=None) -> list[str]:
    out = [] if out is None else out
    if isinstance(v, dict):
        out.append(f"{path}: dict({len(v)})")
        for k, x in v.items():
            transcript_schema(x, f"{path}.{k}", out)
    elif isinstance(v, list):
        out.append(f"{path}: list({len(v)})")
        if v and isinstance(v[0], dict):
            merged: dict = {}
            for e in v:
                for k, x in e.items():
                    merged.setdefault(k, x if isinstance(x, (dict, list)) else describe_type(x))
            for k, x in merged.items():
                if isinstance(x, (dict, list)):
                    transcript_schema(x, f"{path}[].{k}", out)
                else:
                    out.append(f"{path}[].{k}: {x}")
    else:
        out.append(f"{path}: {describe_type(v)}")
    return out


def analyze_transcript(con, path: Path) -> dict:
    data = json.loads(path.read_text())
    study = next(x["課業學習"] for x in data if "課業學習" in x)
    recs = [r for g in study["gradeRecordList"] for r in g["GradeRecords"]]
    matched = name_eq = credit_eq = 0
    kind_vs_label = collections.Counter()
    for r in recs:
        row = con.execute("SELECT name, point, kind FROM COURSE WHERE y=? AND s=? AND subNum=? LIMIT 1",
                          (r["academicYear"], r["semester"], r["courseCode"])).fetchone()
        if row:
            matched += 1
            name_eq += row["name"].replace(" ", "") == r["courseName"].replace(" ", "")
            credit_eq += float(row["point"]) == float(r["credit"])
            kind_vs_label[(row["kind"], r["requiredOrElectiveCourse"])] += 1
    numeric = sum(1 for r in recs if re.fullmatch(r"[\d.]+", r["score"] or ""))
    return {
        "top_level_sections": [list(x.keys())[0] for x in data],
        "n_grade_records": len(recs),
        "n_semesters": len({r["academicYearSemester"] for r in recs}),
        "matched_to_COURSE_by_y_s_subNum": matched, "name_equal": name_eq, "credit_equal": credit_eq,
        "kind_vs_label": {f"kind={k}/{l}": v for (k, l), v in sorted(kind_vs_label.items())},
        "score_numeric": numeric, "score_non_numeric": len(recs) - numeric,
        "schema": transcript_schema(data),
    }


# ── main ────────────────────────────────────────────────────────────────────
def field_inventory(courses) -> dict:
    keys = [k for k in courses[0].keys() if k != "dps"]
    out = {}
    for k in keys:
        vals = [c[k] for c in courses]
        filled = [v for v in vals if is_filled(v)]
        out[k] = {"fill_pct": pct(len(filled), len(vals)), "distinct": len(set(map(str, filled))),
                  "len_p50": quantiles([len(str(v)) for v in filled]).get("p50")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--unit-json", type=Path, default=DEFAULT_UNIT_JSON)
    ap.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    ap.add_argument("--year", default="114")
    ap.add_argument("--sem", default="2")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from loader.courses import _extract_weekly_topics as weekly_fn  # type: ignore
    except Exception:  # loader is optional
        weekly_fn = None

    con = connect(args.db)
    courses = load_semester(con, args.year, args.sem)
    unit_names = load_unit_tree(args.unit_json) if args.unit_json.exists() else {}
    stats = {
        "semester": f"{args.year}-{args.sem}",
        "rows": con.execute("SELECT COUNT(*) FROM COURSE WHERE y=? AND s=?", (args.year, args.sem)).fetchone()[0],
        "courses": len(courses),
        "inventory": field_inventory(courses),
        "codes": {
            k: dict(collections.Counter(str(c[k]) for c in courses).most_common(20))
            for k in ("kind", "lmtKind", "core", "lang", "point", "smtQty", "tranTpe")
        },
        "time": analyze_time(con, courses),
        "units": analyze_units(courses, unit_names),
        "note": analyze_note(courses),
        "location": analyze_location(courses),
        "sections": analyze_sections(courses, weekly_fn),
        "derived": analyze_derived(courses, unit_names),
        "linkage": analyze_linkage(con, courses),
        "rate": analyze_rate(con),
        "result": analyze_result(con, courses),
    }
    if args.transcript.exists():
        stats["transcript"] = analyze_transcript(con, args.transcript)

    for section, val in stats.items():
        if section == "transcript":
            val = {k: v for k, v in val.items() if k != "schema"} | {"schema_lines": len(val["schema"])}
        print(f"\n=== {section} ===")
        print(json.dumps(val, ensure_ascii=False, indent=1, default=str)[:6000])
    if args.json:
        args.json.write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
