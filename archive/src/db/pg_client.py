"""PostgreSQL connection helper.

Reads connection params from environment variables (with defaults).

Env vars:
    PG_HOST     (default: localhost)
    PG_PORT     (default: 5433)   -- 5433 因本機 5432 被 SSH tunnel 佔用
    PG_USER     (default: postgres)
    PG_PASSWORD (default: nccu)
    PG_DBNAME   (default: nccu)
"""

from __future__ import annotations

import os

import psycopg2
import psycopg2.extensions


def get_connection() -> psycopg2.extensions.connection:
    """Return a psycopg2 connection using env vars or defaults."""
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5433")),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "nccu"),
        dbname=os.getenv("PG_DBNAME", "nccu"),
    )
