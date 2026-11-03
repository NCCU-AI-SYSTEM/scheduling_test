"""Real questions against a restored release: does the search work from PostgreSQL alone?

  V4_PG_DSN=... V4_PG_DOC=v4.doc V4_PG_SESSION=v4.doc_session python -m scripts.v4.analysis.release_smoke
"""

from __future__ import annotations

import time

import numpy as np

from src.rerankers.models import load as load_reranker
from src.retrievers.encoders import SPECS, Encoder
from src.v4.pg_backend import DSN, PgRanker, cond_ids, connect
from src.v4.pipeline import Config, V4Search
from src.v4.planner import plan
from src.v4.struct_store import load_struct_pg

QUERIES = ["想學資料庫", "週三下午的英文授課通識", "不要早八的統計學", "晚上開的 2 學分選修", "資科系的機器學習",
           "不要期中考的通識", "個經哪個老師好", "週五沒課的人可以修什麼跟心理學有關的課"]


def main() -> None:
    t0 = time.time()
    b = load_struct_pg(DSN)
    print(f"semester {b.meta['semester']}: {len(b.catalog)} courses, {len(b.offerings)} classes, loaded in {time.time() - t0:.1f} s")
    ids = [c.course_id for c in b.catalog]
    conn = connect()
    search = V4Search(b.catalog, Encoder(SPECS["e5-large-instruct"]), load_reranker("qwen3-rerank-0.6b", device="cuda", max_len=512),
                      b.aspects, offerings=b.offerings, aliases=b.aliases, assess=b.assess,
                      pg=PgRanker(conn, ids, sparse="whitespace"), pg_conditions=lambda c: cond_ids(conn, c))
    cfg = Config(rerank=True, merge_offerings=True, named_skips_conditions=True)
    cat = {c.course_id: c for c in b.catalog}
    times = []
    for q in QUERIES:
        t = time.time()
        p = plan(q, "gemma4:12b")
        hits = search.search(p, q, 5, cfg)
        times.append(time.time() - t)
        print(f"\n{q}  ({times[-1] * 1000:.0f} ms)  must={p.cond.must} not={p.cond.must_not} course={p.named_course}")
        for h in hits:
            c = cat[h.course_id].course
            print(f"  {h.course_id}  {c.name}｜{c.teacher}｜{c.time_raw}｜{c.point} 學分｜{cat[h.course_id].meta.get('lang')}")
    print(f"\np50 {np.median(times[1:]) * 1000:.0f} ms over {len(times) - 1} queries")


if __name__ == "__main__":
    main()
