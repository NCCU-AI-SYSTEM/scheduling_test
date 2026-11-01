"""Add the runtime tables to data/v4/course_struct.db so the search needs no other file.

  dept       department code -> college, degree level (student-number lookup); from 1142.db
  aspect     preference scores per course: syllabus-derived plus the review aspects that
             passed the manual audit (workload, difficulty, group_work, teaching, usefulness)
  embedding  document vectors per encoder, for the exact doc_text() the search indexes

Run after scripts.v4.build_structured (which rebuilds the file from scratch):

  python -m scripts.v4.build_runtime --encoder e5-large-instruct --aspects data/v4/aspects/gemma4_12b.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import time
from pathlib import Path

from src.v4.aspects import build_aspects
from src.v4.offering import dept_registry
from src.v4.pipeline import doc_text
from src.v4.struct_store import STRUCT_DB, load_struct

SCHEMA = """
DROP TABLE IF EXISTS dept;
DROP TABLE IF EXISTS aspect;
CREATE TABLE dept (code TEXT PRIMARY KEY, college TEXT, level TEXT);
CREATE TABLE aspect (course_id TEXT, aspect TEXT, value REAL, PRIMARY KEY (course_id, aspect));
CREATE TABLE IF NOT EXISTS embedding (course_id TEXT, model TEXT, dim INTEGER, vec BLOB,
  PRIMARY KEY (course_id, model));
"""


def text_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=STRUCT_DB)
    p.add_argument("--encoder", action="append", default=[], help="encoder name from src.retrievers.encoders.SPECS")
    p.add_argument("--aspects", type=Path, default=None, help="review aspect jsonl (scripts.v4.extract_aspects)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    t0 = time.time()
    bundle = load_struct(a.db)
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO dept VALUES (?, ?, ?)",
                    [(d.code, d.college, d.level) for d in dept_registry().values()])
    asp = build_aspects(bundle.catalog, a.aspects)
    con.executemany("INSERT INTO aspect VALUES (?, ?, ?)",
                    [(cid, k, float(v)) for cid, d in asp.items() for k, v in d.items() if v is not None])
    texts = [doc_text(c) for c in bundle.catalog]
    for name in a.encoder:
        from src.retrievers.encoders import SPECS, Encoder

        emb, _ = Encoder(SPECS[name], device=a.device).encode_docs(texts)
        con.execute("DELETE FROM embedding WHERE model = ?", (name,))
        con.executemany("INSERT INTO embedding VALUES (?, ?, ?, ?)",
                        [(c.course_id, name, emb.shape[1], emb[i].astype("float32").tobytes())
                         for i, c in enumerate(bundle.catalog)])
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"embedding_text_hash:{name}", text_hash(texts)))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('runtime_built_at', ?)", (time.strftime("%Y/%m/%d %H:%M"),))
    con.commit()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("dept", "aspect", "embedding")}
    con.close()
    print(f"runtime tables {counts} in {time.time() - t0:.1f}s; size {a.db.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
