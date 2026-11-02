"""Add the runtime tables to data/v4/course_struct.db so the search needs no other file.

  dept       department code -> college, degree level (student-number lookup); from 1142.db
  unit       department names <-> codes per level: unit.json (course-system tree) merged with
             nccu_units.tsv (school unit list); identity resolution reads it at runtime
  aspect     preference scores per course: syllabus-derived plus the review aspects that
             passed the manual audit (workload, difficulty, group_work, teaching, usefulness)
  embedding  document vectors per encoder, for the exact doc_text() the search indexes
  search_doc, search_session
             what the search ranks and filters on, already derived (document text, char
             n-grams, condition fields, session hours), so CourseDataPrep's prep.load_v4 can
             build the PostgreSQL tables from this file without importing v4 code

Run after scripts.v4.build_structured (which rebuilds the file from scratch):

  python -m scripts.v4.build_runtime --encoder e5-large-instruct --aspects data/v4/aspects/gemma4_12b.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from src.retrievers.sparse import _char12
from src.v4.aspects import build_aspects
from src.v4.identity import UnitTable
from src.v4.offering import dept_registry
from src.v4.pipeline import doc_text
from src.v4.semester import source_db
from src.v4.struct_store import STRUCT_DB, load_struct

SCHEMA = """
DROP TABLE IF EXISTS dept;
DROP TABLE IF EXISTS aspect;
DROP TABLE IF EXISTS unit;
CREATE TABLE dept (code TEXT PRIMARY KEY, college TEXT, level TEXT);
CREATE TABLE unit (code TEXT, name TEXT, college TEXT, level TEXT, source TEXT,
  PRIMARY KEY (code, level, source));
CREATE TABLE aspect (course_id TEXT, aspect TEXT, value REAL, PRIMARY KEY (course_id, aspect));
CREATE TABLE IF NOT EXISTS embedding (course_id TEXT, model TEXT, dim INTEGER, vec BLOB,
  PRIMARY KEY (course_id, model));
DROP TABLE IF EXISTS search_doc;
DROP TABLE IF EXISTS search_session;
CREATE TABLE search_doc (course_id TEXT PRIMARY KEY, pos INTEGER, name TEXT, lang TEXT, cats TEXT,
  lmt_kind TEXT, point REAL, unit TEXT, text TEXT, toks TEXT);
CREATE TABLE search_session (course_id TEXT, weekday INTEGER, start_h INTEGER, end_h INTEGER);
"""


def text_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def search_rows(catalog, texts: list[str]) -> tuple[list[tuple], list[tuple]]:
    """One search_doc row per course in catalogue order (pos = index into the embedding order) and
    its sessions. Empty condition fields become '' / 0.0, as src/eval/conditions treats them."""
    docs, sessions = [], []
    for i, (c, t) in enumerate(zip(catalog, texts)):
        m = c.meta
        docs.append((c.course_id, i, c.course.name, m.get("lang") or "", json.dumps(sorted(c.categories), ensure_ascii=False),
                     m.get("lmt_kind") or "", float(m.get("point") or 0), m.get("unit") or "", t, " ".join(_char12(t))))
        sessions += [(c.course_id, s["weekday"], s["start"], s["end"]) for s in m.get("sessions", [])]
    return docs, sessions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=STRUCT_DB)
    p.add_argument("--encoder", action="append", default=[], help="encoder name from src.retrievers.encoders.SPECS")
    p.add_argument("--aspects", type=Path, default=None, help="review aspect jsonl (scripts.v4.extract_aspects)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--src", type=Path, default=source_db(), help="raw scrape, for the department registry")
    a = p.parse_args()
    t0 = time.time()
    bundle = load_struct(a.db)
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO dept VALUES (?, ?, ?)",
                    [(d.code, d.college, d.level) for d in dept_registry(a.src).values()])
    con.executemany("INSERT OR IGNORE INTO unit VALUES (?, ?, ?, ?, ?)", UnitTable.load().rows())
    asp = build_aspects(bundle.catalog, a.aspects)
    con.executemany("INSERT INTO aspect VALUES (?, ?, ?)",
                    [(cid, k, float(v)) for cid, d in asp.items() for k, v in d.items() if v is not None])
    texts = [doc_text(c) for c in bundle.catalog]
    docs, sessions = search_rows(bundle.catalog, texts)
    con.executemany("INSERT INTO search_doc VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", docs)
    con.executemany("INSERT INTO search_session VALUES (?, ?, ?, ?)", sessions)
    for name in a.encoder:
        from src.retrievers.encoders import SPECS, Encoder

        emb, _ = Encoder(SPECS[name], device=a.device).encode_docs(texts)
        con.execute("DELETE FROM embedding WHERE model = ?", (name,))
        con.executemany("INSERT INTO embedding VALUES (?, ?, ?, ?)",
                        [(c.course_id, name, emb.shape[1], emb[i].astype("float32").tobytes())
                         for i, c in enumerate(bundle.catalog)])
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"embedding_text_hash:{name}", text_hash(texts)))
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (f"embedding_hf_id:{name}", SPECS[name].hf_id))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('runtime_built_at', ?)", (time.strftime("%Y/%m/%d %H:%M"),))
    con.commit()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("dept", "unit", "aspect", "embedding", "search_doc", "search_session")}
    con.close()
    print(f"runtime tables {counts} in {time.time() - t0:.1f}s; size {a.db.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
