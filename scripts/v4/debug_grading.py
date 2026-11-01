"""Show each step of the grading parser for one syllabus text (or a course name in 114-2).

  python -m scripts.v4.debug_grading 五人制足球
"""

from __future__ import annotations

import sqlite3
import sys

from src.v4.structured import grading
from src.v4.structured.text import merge_fragments, nfkc
from src.v4.catalog import DB


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    row = con.execute("SELECT name, evaluation FROM COURSE WHERE y='114' AND s='2' AND name LIKE ? LIMIT 1",
                      (f"%{sys.argv[1]}%",)).fetchone()
    name, raw = row
    text = grading.cn_percent(merge_fragments(nfkc(raw)))
    print(name, "|", text[:300])
    items = grading.raw_items(text)
    print("raw     ", items)
    print("totals  ", grading._drop_totals(items))
    print("collapse", grading._collapse_children(grading._drop_totals(items)))
    print("prefix  ", grading._prefix_100(grading._drop_totals(items)))
    g = grading.parse(raw)
    print("parse   ", [(i.category, i.label, i.pct) for i in g.items], g.pct_sum, g.reliable)


if __name__ == "__main__":
    main()
