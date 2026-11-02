"""Where do PostgreSQL rankings differ from the in-memory ones? Dense and BM25 checked apart.

For R-set query texts: pgvector exact cosine vs numpy dot product (top 50), and pg_search
BM25 over the same char n-gram tokens vs rank_bm25 (top 50), both over all courses.

  python -m scripts.v4.pg_rank_parity
"""

from __future__ import annotations

import numpy as np

from scripts.v4.run_e2e import load_r
from src.retrievers.encoders import SPECS, Encoder
from src.retrievers.sparse import BM25, tokenize_all
from src.v4.pg_backend import PgRanker, connect
from src.v4.pipeline import V4Search, doc_text
from src.v4.struct_store import load_struct


def overlap(a: list[int], b: list[int], k: int) -> float:
    return len(set(a[:k]) & set(b[:k])) / max(1, len(set(a[:k]) | set(b[:k])))


def main() -> None:
    b = load_struct(embedding_models=("e5-large-instruct",))
    ids = [c.course_id for c in b.catalog]
    emb = b.embeddings["e5-large-instruct"]
    enc = Encoder(SPECS["e5-large-instruct"])
    bm25 = BM25(tokenize_all([doc_text(c) for c in b.catalog], "char12"))
    pg = PgRanker(connect(), ids, sparse="whitespace")
    mask = np.ones(len(ids), dtype=bool)
    qs = [r["query"] for r in load_r()][:100]
    d_same, d_ov, s_same1, s_ov10, s_ov50 = 0, [], 0, [], []
    for q in qs:
        qv = enc.encode_queries([q])[0]
        mem = V4Search._rank(emb @ qv, mask)
        db = pg.dense(qv, mask)
        d_same += mem[:10] == db[:10]
        d_ov.append(overlap(mem, db, 50))
        smem = V4Search._rank(bm25.scores(tokenize_all([q], "char12")[0]), mask)
        sdb = pg.sparse(q, mask)
        s_same1 += bool(smem) and bool(sdb) and smem[0] == sdb[0]
        s_ov10.append(overlap(smem, sdb, 10))
        s_ov50.append(overlap(smem, sdb, 50))
    n = len(qs)
    print(f"queries {n}")
    print(f"dense  pgvector vs numpy: identical top10 {d_same / n:.1%}, top50 overlap {np.mean(d_ov):.3f}")
    print(f"bm25   pg_search vs rank_bm25 (same tokens): same top1 {s_same1 / n:.1%}, "
          f"top10 overlap {np.mean(s_ov10):.3f}, top50 overlap {np.mean(s_ov50):.3f}")


if __name__ == "__main__":
    main()
