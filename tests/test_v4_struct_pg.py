"""load_struct_pg: the runtime bundle from a release's v4 schema must equal the one from SQLite.

The parity test needs a database holding the same release as the local course_struct.db:
  V4_PG_PARITY_DSN=postgresql://postgres:postgres@127.0.0.1:5433/academic pytest tests/test_v4_struct_pg.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

from src.v4.struct_store import STRUCT_DB, _pg_sql, _vector, load_struct, load_struct_pg

MODEL = "e5-large-instruct"


def test_pg_sql_translates_sqlite_dialect():
    assert _pg_sql("SELECT a FROM t WHERE b = ? AND c IN (?,?)") == "SELECT a FROM t WHERE b = %s AND c IN (%s,%s)"
    assert _pg_sql("SELECT offering_id, * FROM channel ORDER BY offering_id, rowid") == (
        "SELECT offering_id, * FROM channel ORDER BY offering_id, ctid")


def test_vector_parses_pgvector_text():
    v = _vector("[0.5,-1,0.25]")
    assert v.dtype == np.float32 and v.tolist() == [0.5, -1.0, 0.25]


def test_load_struct_pg_refuses_a_database_without_v4(monkeypatch):
    class Cur:
        def execute(self, *_a):
            pass

        def fetchone(self):
            return (None,)

        def close(self):
            pass

    class Conn:
        autocommit = False

        def cursor(self, **_k):
            return Cur()

        def close(self):
            pass

    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", lambda _dsn: Conn())
    with pytest.raises(FileNotFoundError, match="prep.load_v4"):
        load_struct_pg("postgresql://unused")


@pytest.mark.skipif(not os.environ.get("V4_PG_PARITY_DSN") or not Path(STRUCT_DB).exists(),
                    reason="needs V4_PG_PARITY_DSN and data/v4/course_struct.db of the same release")
def test_bundle_from_postgres_equals_sqlite():
    a = load_struct(embedding_models=(MODEL,))
    b = load_struct_pg(os.environ["V4_PG_PARITY_DSN"], embedding_models=(MODEL,))
    assert a.meta["semester"] == b.meta["semester"]
    assert [c.course_id for c in a.catalog] == [c.course_id for c in b.catalog]
    diffs = [x.course_id for x, y in zip(a.catalog, b.catalog) if repr(x) != repr(y)]
    assert not diffs, f"{len(diffs)} catalog courses differ, first {diffs[:3]}"
    assert repr(a.offerings) == repr(b.offerings)      # channel order decides the primary channel
    assert a.depts == b.depts
    assert a.aliases == b.aliases
    assert a.assess == b.assess
    assert a.aspects == b.aspects
    assert repr(a.units.rows()) == repr(b.units.rows())
    assert np.array_equal(a.embeddings[MODEL], b.embeddings[MODEL])
