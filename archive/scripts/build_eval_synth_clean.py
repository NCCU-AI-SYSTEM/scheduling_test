"""
Build eval_synth_clean.jsonl (7954q) from eval_synth.jsonl (8253q)

Removes 299 constraint queries where the gold course does NOT satisfy
the constraints parsed from the query text.

Dimensions checked (constraint type only):
  - lang     : lang_include vs COURSE.lang
  - weekday  : weekday_include vs parsed COURSE.time
  - hour     : hour_min/hour_max vs parsed COURSE.time
  - unit     : unit_include vs COURSE.unit
  - point    : point_min/point_max vs COURSE.point

topic / colloquial are passed through unchanged (0 issues found).

Output: data/raw/eval_synth_clean.jsonl
  total : 7954  (topic=2751, colloquial=2751, constraint=2452)
  dropped: 299  (weekday=179, hour=178, unit=31, point=12, lang=6)
           note: multi-reason queries counted once per bad dimension
"""

import json
import sqlite3
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# lazy-import so this script can run without heavy deps loaded
try:
    from src.query_rewriters.structured import parse_constraints
except ImportError as e:
    print(f"[ERROR] cannot import parse_constraints: {e}", file=sys.stderr)
    sys.exit(1)

WEEKMAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7}
HOURMAP = {
    "A": 1, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "E": 11.5, "X": 12, "D": 13, "F": 14, "G": 15, "H": 16,
}


def parse_time(ts: str):
    slots = []
    for seg in str(ts or "").split():
        if not seg or seg[0] not in WEEKMAP:
            continue
        wd = WEEKMAP[seg[0]]
        hrs = [HOURMAP[ch] for ch in seg[1:] if ch in HOURMAP]
        slots.append((wd, hrs))
    return slots


def is_mismatch(query: str, row) -> list[str]:
    """Return list of mismatched dimension names, empty = OK."""
    c = parse_constraints(query)
    slots = parse_time(row["time"])
    gold_wds = {s[0] for s in slots}
    gold_hrs = {h for s in slots for h in s[1]}

    reasons = []

    if c.lang_include:
        if not any(l in str(row["lang"] or "") for l in c.lang_include):
            reasons.append("lang")

    if c.weekday_include and gold_wds:
        if not (gold_wds & c.weekday_include):
            reasons.append("weekday")

    if (c.hour_min is not None or c.hour_max is not None) and gold_hrs:
        ok = any(
            (c.hour_min is None or h >= c.hour_min)
            and (c.hour_max is None or h <= c.hour_max)
            for h in gold_hrs
        )
        if not ok:
            reasons.append("hour")

    if c.unit_include:
        if not any(kw in str(row["unit"] or "") for kw in c.unit_include):
            reasons.append("unit")

    if c.point_min is not None and c.point_max is not None:
        pt = float(row["point"] or 0)
        if not (c.point_min <= pt <= c.point_max):
            reasons.append("point")

    return reasons


def main():
    db_path  = ROOT / "data/1142.db"
    src_path = ROOT / "data/raw/eval_synth.jsonl"
    dst_path = ROOT / "data/raw/eval_synth_clean.jsonl"

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    rows = []
    with open(src_path) as f:
        for line in f:
            rows.append(json.loads(line))

    bad_qids = set()
    by_reason: Counter = Counter()

    for r in rows:
        if r["type"] != "constraint":
            continue
        gid = r.get("gold_course_id", "")
        db_row = con.execute(
            "SELECT lang, time, unit, point FROM COURSE WHERE id=?", (gid,)
        ).fetchone()
        if not db_row:
            bad_qids.add(r["qid"])
            by_reason["gold_not_in_db"] += 1
            continue
        reasons = is_mismatch(r["query"], db_row)
        if reasons:
            bad_qids.add(r["qid"])
            for rn in reasons:
                by_reason[rn] += 1

    clean = [r for r in rows if r["qid"] not in bad_qids]
    with open(dst_path, "w") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_orig  = len(rows)
    n_drop  = len(bad_qids)
    n_clean = len(clean)
    tc = Counter(r["type"] for r in clean)
    to = Counter(r["type"] for r in rows)

    print(f"原始: {n_orig}  刪除: {n_drop}  清理後: {n_clean}")
    print("各 type:")
    for t in ["topic", "colloquial", "constraint"]:
        print(f"  {t:12s}: {to[t]} → {tc[t]} (刪 {to[t]-tc[t]})")
    print("刪除原因:")
    for k, v in by_reason.most_common():
        print(f"  {k:10s}: {v}")
    print(f"輸出: {dst_path}")


if __name__ == "__main__":
    main()
