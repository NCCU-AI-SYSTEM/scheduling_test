"""PostgreSQL (ParadeDB: pg_search + pgvector) backend for the v4 search, for comparison
with the in-memory SQLite path.

What moves into the database and what stays in Python:
  in SQL     hard conditions (WHERE over v4_doc and v4_session), BM25 (pg_search), dense
             nearest neighbours (pgvector, exact cosine, no index: 2,655 rows)
  in Python  planner, relaxation order, identity / transcript / assessment filters, RRF,
             reranker, preferences, class merge, result table (unchanged, they need no DB)

BM25 variants, one bm25 index over three columns so each can be queried on its own:
  toks    char unigrams + bigrams made in Python (same tokens as the SQLite path),
          indexed with pdb.whitespace: isolates Tantivy's BM25 from tokenisation
  doc_ng  raw text, pdb.ngram(1,2): no Python tokenizer at query time
  doc_jb  raw text, pdb.jieba: what the online system uses today

  V4_PG_DSN=postgresql://postgres:postgres@127.0.0.1:5434/v4

Ranking and conditions also run against a CourseDataPrep release (prep.load_v4 builds the same
columns): V4_PG_DOC=v4.doc V4_PG_SESSION=v4.doc_session, whitespace tokens only.
"""

from __future__ import annotations

import os

import numpy as np

from src.parsers.base import ConditionResult
from src.retrievers.sparse import _char12

DSN = os.environ.get("V4_PG_DSN", "postgresql://postgres:postgres@127.0.0.1:5434/v4")
SPARSE_FIELDS = {"whitespace": "toks", "ngram": "doc_ng", "jieba": "doc_jb"}
DOC = os.environ.get("V4_PG_DOC", "v4_doc")
SES = os.environ.get("V4_PG_SESSION", "v4_session")

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS v4_session;
DROP TABLE IF EXISTS v4_doc;
CREATE TABLE v4_doc (
  course_id TEXT PRIMARY KEY, pos INT NOT NULL, name TEXT, lang TEXT, cats TEXT[], lmt_kind TEXT,
  point REAL, unit TEXT, doc TEXT, toks TEXT, doc_ng TEXT, doc_jb TEXT, emb vector(%(dim)s));
CREATE TABLE v4_session (course_id TEXT REFERENCES v4_doc, weekday INT, start_h INT, end_h INT);
CREATE INDEX v4_session_cid ON v4_session (course_id);
"""
BM25_INDEX = ("CREATE INDEX v4_doc_bm25 ON v4_doc USING bm25 (course_id, (toks::pdb.whitespace), "
              "(doc_ng::pdb.ngram(1,2)), (doc_jb::pdb.jieba)) WITH (key_field='course_id')")


def connect(dsn: str = DSN):
    import psycopg2

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def load(conn, catalog, texts: list[str], emb: np.ndarray) -> None:
    """One row per course in catalogue order; pos is the index the Python search uses."""
    from psycopg2.extras import execute_values

    cur = conn.cursor()
    cur.execute(SCHEMA % {"dim": emb.shape[1]})
    rows = []
    for i, (c, t) in enumerate(zip(catalog, texts)):
        m = c.meta
        rows.append((c.course_id, i, c.course.name, m.get("lang"), sorted(c.categories), m.get("lmt_kind") or "",
                     float(m.get("point") or 0), m.get("unit") or "", t, " ".join(_char12(t)), t, t,
                     "[" + ",".join(f"{x:.7f}" for x in emb[i]) + "]"))
    execute_values(cur, "INSERT INTO v4_doc VALUES %s", rows, page_size=200)
    ses = [(c.course_id, s["weekday"], s["start"], s["end"]) for c in catalog for s in c.meta.get("sessions", [])]
    execute_values(cur, "INSERT INTO v4_session VALUES %s", ses, page_size=1000)
    cur.execute(BM25_INDEX)
    cur.execute("ANALYZE v4_doc; ANALYZE v4_session;")


# ── conditions as SQL (mirrors src/eval/conditions._dim_match) ──────────────
_SES = "EXISTS (SELECT 1 FROM " + SES + " s WHERE s.course_id = d.course_id AND {})"
_KINDS = ("必修", "選修", "群修", "通識", "體育")


def _clause(dim: str, value) -> tuple[str, list] | None:
    if value is None or (isinstance(value, list) and not value):
        return None
    if dim == "course_lang":
        return "d.lang = ANY(%s)", [list(value)]
    if dim == "weekday":
        return _SES.format("s.weekday = ANY(%s)"), [[int(x) for x in value]]
    if dim == "hour_range":
        return _SES.format("s.start_h < %s AND s.end_h > %s"), [int(value[1]), int(value[0])]
    if dim == "point":
        return "d.point = %s", [float(value)]
    if dim == "kind":
        wanted = [k for k in value if k in _KINDS]
        return ("d.cats && %s::text[]", [wanted]) if wanted else None
    if dim == "lmt_kind":
        return "d.lmt_kind = ANY(%s)", [list(value)]
    if dim == "unit":
        return "EXISTS (SELECT 1 FROM unnest(%s::text[]) u WHERE u <> '' AND position(u IN d.unit) > 0)", [list(value)]
    raise ValueError(f"unknown dimension {dim}")


def where(cond: ConditionResult) -> tuple[str, list]:
    parts, params = [], []
    for side, clauses in (("must", cond.must), ("not", cond.must_not)):
        for dim, value in clauses.items():
            c = _clause(dim, value)
            if c is None:
                continue
            parts.append(c[0] if side == "must" else f"NOT ({c[0]})")
            params += c[1]
    return (" AND ".join(parts) or "TRUE"), params


def cond_ids(conn, cond: ConditionResult) -> set[str]:
    sql, params = where(cond)
    cur = conn.cursor()
    cur.execute(f"SELECT course_id FROM {DOC} d WHERE {sql}", params)
    return {r[0] for r in cur.fetchall()}


class PgRanker:
    """Drop-in for V4Search's in-memory ranking: (text or vector, mask) -> ranked positions."""

    def __init__(self, conn, ids: list[str], sparse: str = "whitespace", k: int = 50):
        self.conn, self.ids, self.k = conn, ids, k
        self.field = SPARSE_FIELDS[sparse]
        self.sparse_kind = sparse
        self.pos = {cid: i for i, cid in enumerate(ids)}

    def _allowed(self, mask: np.ndarray) -> list[str] | None:
        return None if mask.all() else [self.ids[i] for i in np.flatnonzero(mask)]

    def sparse(self, text: str, mask: np.ndarray) -> list[int]:
        q = " ".join(_char12(text)) if self.sparse_kind == "whitespace" else text
        if not q.strip():
            return []
        allowed = self._allowed(mask)
        sql = (f"SELECT course_id FROM {DOC} WHERE {self.field} ||| %s"
               + (" AND course_id = ANY(%s)" if allowed is not None else "")
               + " ORDER BY pdb.score(course_id) DESC LIMIT %s")
        cur = self.conn.cursor()
        cur.execute(sql, [q] + ([allowed] if allowed is not None else []) + [self.k])
        return [self.pos[r[0]] for r in cur.fetchall()]

    def dense(self, qv: np.ndarray, mask: np.ndarray) -> list[int]:
        allowed = self._allowed(mask)
        vec = "[" + ",".join(f"{x:.7f}" for x in qv) + "]"
        sql = (f"SELECT course_id FROM {DOC}" + (" WHERE course_id = ANY(%s)" if allowed is not None else "")
               # <#> is negative inner product: the in-memory path ranks by dot product, and
               # fp16-normalised vectors have norms of 1.0000x, so cosine (<=>) reorders near-ties
               + " ORDER BY emb <#> %s::vector LIMIT %s")
        cur = self.conn.cursor()
        cur.execute(sql, ([allowed] if allowed is not None else []) + [vec, self.k])
        return [self.pos[r[0]] for r in cur.fetchall()]
