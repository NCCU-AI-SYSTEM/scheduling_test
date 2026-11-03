"""How many 114-2 syllabi write the grading breakdown twice (Chinese + English)?

For each one, prints what the parser keeps now; compares with what it kept without the
bilingual rule so the change can be checked by eye.

  python -m scripts.v4.analysis.bilingual_grading_check 8
"""

from __future__ import annotations

import random
import sqlite3
import sys

from src.v4.catalog import DB
from src.v4.structured import grading
from src.v4.structured.text import merge_fragments, nfkc


def main() -> None:
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute("SELECT DISTINCT id, name, evaluation FROM COURSE WHERE y='114' AND s='2'").fetchall()
    hits = []
    for cid, name, raw in rows:
        text = grading.cn_percent(merge_fragments(nfkc(raw)))
        items = grading._drop_totals(grading.raw_items(text)) if text else []
        if grading._bilingual_half(items) is not None and not 95 <= sum(p for _, p in items) <= 105:
            old = grading._collapse_children(items)
            old = old if 95 <= sum(p for _, p in old) <= 105 else (grading._prefix_100(items) or items)
            hits.append((cid, name, old, grading.parse(raw).items))
    print(f"courses with a bilingual breakdown: {len(hits)} / {len(rows)}")
    changed = sum(1 for _, _, old, new in hits if [(l, p) for l, p in old] != [(i.label, i.pct) for i in new])
    print(f"kept items changed by the rule: {changed}")
    for cid, name, old, new in random.Random(3).sample(hits, min(n_show, len(hits))):
        print(f"\n{cid} {name}\n  before: {old}\n  after:  {[(i.label, i.pct) for i in new]}")


if __name__ == "__main__":
    main()
