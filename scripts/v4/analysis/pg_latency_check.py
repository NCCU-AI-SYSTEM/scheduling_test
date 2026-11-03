"""Per-call latency of BM25 and dense in PostgreSQL, unmasked and masked, repeated to see warm-up.

  V4_PG_DSN=... V4_PG_DOC=v4.doc V4_PG_SESSION=v4.doc_session python -m scripts.v4.analysis.pg_latency_check
"""

from __future__ import annotations

import time

import numpy as np

from src.v4.pg_backend import DSN, PgRanker, connect
from src.v4.struct_store import load_struct_pg


def main() -> None:
    t0 = time.time()
    b = load_struct_pg(DSN, embedding_models=("e5-large-instruct",))
    print(f"load_struct_pg {time.time() - t0:.2f} s, {len(b.catalog)} courses, "
          f"{b.embeddings['e5-large-instruct'].shape}, offerings {len(b.offerings)}, semester {b.meta['semester']}")
    ids = [c.course_id for c in b.catalog]
    pg = PgRanker(connect(), ids, sparse="whitespace")
    full = np.ones(len(ids), dtype=bool)
    part = np.zeros(len(ids), dtype=bool)
    part[::5] = True
    qv = b.embeddings["e5-large-instruct"][0]
    for label, fn in (("bm25 all", lambda: pg.sparse("自然語言處理 機器學習", full)),
                      ("bm25 20%", lambda: pg.sparse("自然語言處理 機器學習", part)),
                      ("dense all", lambda: pg.dense(qv, full)),
                      ("dense 20%", lambda: pg.dense(qv, part))):
        ts = []
        for _ in range(30):
            t = time.time()
            fn()
            ts.append((time.time() - t) * 1000)
        print(f"{label:10s} first {ts[0]:7.1f} ms, p50 {np.median(ts):7.1f} ms, p95 {np.percentile(ts, 95):7.1f} ms")


if __name__ == "__main__":
    main()
