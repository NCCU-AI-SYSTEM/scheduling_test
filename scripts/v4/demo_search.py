"""Search once and print the result table (Markdown).

  python -m scripts.v4.demo_search "週三下午的英文授課通識" --planner gemma4:12b
  STUDENT_ID=112703001 python -m scripts.v4.demo_search "想修自然語言處理"

The student number is read from the STUDENT_ID environment variable so it stays out of
shell history and logs; only the derived department, college, level and year are used,
and the number itself is never printed.
"""

from __future__ import annotations

import argparse
import os

from src.rerankers.models import load as load_reranker
from src.retrievers.encoders import SPECS, Encoder
from src.v4.access import from_student_id
from src.v4.aspects import build_aspects
from src.v4.catalog import load_catalog
from src.v4.offering import dept_registry, group_offerings
from src.v4.pipeline import Config, V4Search
from src.v4.planner import plan
from src.v4.present import row, to_markdown


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="+")
    p.add_argument("--planner", default="gemma4:12b")
    p.add_argument("--encoder", default="e5-large-instruct")
    p.add_argument("--reranker", default="qwen3-rerank-0.6b")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()
    cat = load_catalog()
    depts = dept_registry()
    offs = group_offerings(cat, depts)
    search = V4Search(cat, Encoder(SPECS[a.encoder]), load_reranker(a.reranker), build_aspects(cat, None),
                      offerings=offs)
    ident = None
    if os.environ.get("STUDENT_ID"):
        ident = from_student_id(os.environ["STUDENT_ID"], depts)
        print(f"身分：{ident.college}學院 {ident.level} {ident.year} 年級（系所代碼 {ident.dept_code}）")
    for q in a.query:
        pl = plan(q, a.planner)
        cfg = Config(merge_offerings=True, named_skips_conditions=True)
        hits = search.search(pl, q, a.k, cfg, identity=ident)
        cmap = {c.course_id: c for c in cat}
        print(f"\n### {q}\n規劃：need={pl.need!r} 課名={pl.named_course} 老師={pl.named_teacher} "
              f"must={pl.cond.must} must_not={pl.cond.must_not} prefs={pl.prefs}"
              f"{'　放寬：' + '、'.join(hits[0].relaxed) if hits and hits[0].relaxed else ''}\n")
        print(to_markdown([row(h, cmap) for h in hits]))
        print(f"\n（規劃 {pl.seconds:.2f}s，搜尋 {search.timing['total'] * 1000:.0f}ms）")


if __name__ == "__main__":
    main()
