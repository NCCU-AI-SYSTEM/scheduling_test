"""Quick look into data/v4/course_struct.db: aliases, one course card, restriction split.

  python -m scripts.v4.analysis.peek_structured
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parents[3] / "data" / "v4" / "course_struct.db"


def main() -> None:
    con = sqlite3.connect(DB)
    cols = [r[1] for r in con.execute("PRAGMA table_info(alias)")]
    print("alias columns:", cols)
    for a in ("個經", "總經", "初會", "中會", "五管", "大英", "財管", "統計"):
        rows = con.execute("SELECT * FROM alias WHERE alias = ? LIMIT 3", (a,)).fetchall()
        print(" ", a, "->", rows)
    print("\nrestriction by source/hard:",
          con.execute("SELECT source, hard, count(*) FROM restriction GROUP BY 1, 2").fetchall()
          if "source" in [r[1] for r in con.execute("PRAGMA table_info(restriction)")] else
          [r[1] for r in con.execute("PRAGMA table_info(restriction)")])
    cid = con.execute("SELECT course_id FROM course WHERE name LIKE '自然語言處理%' LIMIT 1").fetchone()[0]
    print("\ncard:\n" + con.execute("SELECT card FROM course WHERE course_id = ?", (cid,)).fetchone()[0])
    print("\nrestrictions of a 會計 course with 擋修:")
    row = con.execute("SELECT course_id FROM course WHERE note LIKE '%擋修%' LIMIT 1").fetchone()
    if row:
        for r in con.execute("SELECT * FROM restriction WHERE course_id = ?", (row[0],)):
            print(" ", r)
        for r in con.execute("SELECT * FROM prereq WHERE course_id = ?", (row[0],)):
            print("  prereq", r)


if __name__ == "__main__":
    main()
