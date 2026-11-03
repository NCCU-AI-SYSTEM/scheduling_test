"""Build eval_constraint_clean.jsonl from eval_synth.jsonl.

Filters out constraint queries whose gold course has weekday/hour/kind
that contradicts the query (synth generation noise).

Usage:
    uv run scripts/build_eval_constraint_clean.py
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import unittest.mock as mock
sys.modules.setdefault("ollama", mock.MagicMock())

from src.query_rewriters.structured import parse_constraints  # noqa: E402

DB_PATH    = ROOT / "data" / "1142.db"
SYNTH_PATH = ROOT / "data" / "raw" / "eval_synth.jsonl"
OUT_PATH   = ROOT / "data" / "raw" / "eval_constraint_clean.jsonl"

WEEKMAP = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7}
HOURMAP = {
    "A": 1, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "E": 11.5, "X": 12,
    "D": 13, "F": 14, "G": 15, "H": 16,
}


def parse_time(ts: str):
    if not ts:
        return []
    slots = []
    for seg in ts.split():
        if not seg or seg[0] not in WEEKMAP:
            continue
        wd = WEEKMAP[seg[0]]
        hours = [HOURMAP[ch] for ch in seg[1:] if ch in HOURMAP]
        slots.append((wd, hours))
    return slots


def gold_consistent(c, gold_id: str, con: sqlite3.Connection) -> tuple[bool, str]:
    row = con.execute("SELECT kind, time FROM COURSE WHERE id=?", (gold_id,)).fetchone()
    if not row:
        return False, "not_in_db"
    slots      = parse_time(str(row["time"] or ""))
    gold_wds   = {s[0] for s in slots}
    gold_hours = {h for s in slots for h in s[1]}

    if c.weekday_include and gold_wds and not (gold_wds & c.weekday_include):
        return False, "weekday"
    if (c.hour_min is not None or c.hour_max is not None) and gold_hours:
        ok = any(
            (c.hour_min is None or h >= c.hour_min)
            and (c.hour_max is None or h <= c.hour_max)
            for h in gold_hours
        )
        if not ok:
            return False, "hour"
    if c.kind_include and row["kind"] not in c.kind_include:
        return False, "kind"
    return True, "ok"


def main():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row

    total = kept = dropped = 0
    drop_reasons: dict[str, int] = {}
    out_lines = []

    with open(SYNTH_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "constraint":
                continue
            total += 1

            q       = r["query"]
            gold_id = r.get("gold_course_id", "")
            c       = parse_constraints(q)

            has_any = bool(
                c.weekday_include
                or c.hour_min is not None
                or c.hour_max is not None
                or c.kind_include
            )

            if has_any and gold_id:
                ok, reason = gold_consistent(c, gold_id, con)
                if not ok:
                    dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue

            kept += 1
            out_lines.append(json.dumps({
                "qid":   r["qid"],
                "query": q,
                "qtype": "constraint",
                "gold":  [gold_id],
            }, ensure_ascii=False))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Total constraint: {total}")
    print(f"Kept (clean):     {kept}  ({kept / total * 100:.1f}%)")
    print(f"Dropped:          {dropped}  ({dropped / total * 100:.1f}%)")
    print(f"Drop breakdown:   {drop_reasons}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
