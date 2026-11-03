"""One query, dense side: numpy vs pgvector order with their scores, to see why they differ.

  python -m scripts.v4.analysis.pg_dense_debug
"""

from __future__ import annotations

import numpy as np

from scripts.v4.run_e2e import load_r
from src.retrievers.encoders import SPECS, Encoder
from src.v4.pg_backend import connect
from src.v4.pipeline import V4Search
from src.v4.struct_store import load_struct


def main() -> None:
    b = load_struct(embedding_models=("e5-large-instruct",))
    ids = [c.course_id for c in b.catalog]
    emb = b.embeddings["e5-large-instruct"]
    q = load_r()[3]["query"]
    qv = Encoder(SPECS["e5-large-instruct"]).encode_queries([q])[0]
    mask = np.ones(len(ids), dtype=bool)
    mem = V4Search._rank(emb @ qv, mask)[:10]
    vec = "[" + ",".join(f"{x:.7f}" for x in qv) + "]"
    cur = connect().cursor()
    cur.execute("SELECT course_id, 1 - (emb <=> %s::vector) FROM v4_doc ORDER BY emb <=> %s::vector LIMIT 10", (vec, vec))
    db = cur.fetchall()
    print("query:", q, "| norms doc", float(np.linalg.norm(emb[0])), "query", float(np.linalg.norm(qv)))
    print("numpy  :", [(ids[i], round(float(emb[i] @ qv), 5)) for i in mem])
    print("pg     :", [(cid, round(float(s), 5)) for cid, s in db])
    cur.execute("SELECT emb FROM v4_doc WHERE course_id = %s", (ids[mem[0]],))
    stored = np.array([float(x) for x in cur.fetchone()[0].strip("[]").split(",")])
    print("stored vs numpy max abs diff:", float(np.abs(stored - emb[mem[0]]).max()))


if __name__ == "__main__":
    main()
