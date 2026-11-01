"""Which 1142.db does this checkout read, and does it carry the 2026/11/01 repair?

  python -m scripts.v4.check_source_db
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "1142.db"


def main() -> None:
    print("path", SRC, "->", os.path.realpath(SRC), "mtime", os.path.getmtime(os.path.realpath(SRC)))
    con = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
    bad = con.execute("SELECT count(*) FROM COURSE WHERE ai_policy LIKE 'Course Name:%' "
                      "OR evaluation LIKE 'Course Name:%'").fetchone()[0]
    row = con.execute("SELECT substr(evaluation, 1, 40) FROM COURSE WHERE id = '1142102023001'").fetchone()
    print("bad rows", bad, "| 1142102023001 evaluation:", row)


if __name__ == "__main__":
    main()
