"""Load course_struct.db into PostgreSQL (ParadeDB) for the SQLite-vs-PostgreSQL comparison.

  docker run -d --name v4-paradedb -p 127.0.0.1:5434:5432 -e POSTGRES_PASSWORD=postgres \
      -e POSTGRES_DB=v4 --shm-size=1g paradedb/paradedb:latest
  V4_PG_DSN=postgresql://postgres:postgres@127.0.0.1:5434/v4 python -m scripts.v4.pg_setup
"""

from __future__ import annotations

import time

from src.v4.pg_backend import connect, load
from src.v4.pipeline import doc_text
from src.v4.struct_store import load_struct


def main() -> None:
    b = load_struct(embedding_models=("e5-large-instruct",))
    texts = [doc_text(c) for c in b.catalog]
    conn = connect()
    t0 = time.time()
    load(conn, b.catalog, texts, b.embeddings["e5-large-instruct"])
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM v4_doc")
    n = cur.fetchone()[0]
    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('v4_doc')), "
                "pg_size_pretty(pg_database_size(current_database())), "
                "(SELECT extversion FROM pg_extension WHERE extname = 'pg_search'), "
                "(SELECT extversion FROM pg_extension WHERE extname = 'vector'), version()")
    size, dbsize, pgs, vec, ver = cur.fetchone()
    print(f"loaded {n} courses in {time.time() - t0:.1f}s; v4_doc {size}, database {dbsize}; "
          f"pg_search {pgs}, pgvector {vec}; {ver.split(',')[0]}")


if __name__ == "__main__":
    main()
