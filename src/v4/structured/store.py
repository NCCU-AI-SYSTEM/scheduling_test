"""SQLite schema and writer for data/v4/course_struct.db (rebuilt from scratch each run)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE course (
  course_id TEXT PRIMARY KEY, sub_num TEXT, course_group TEXT,
  name TEXT, name_en TEXT, name_base TEXT, teachers TEXT, teacher_raw TEXT,
  credits REAL, kind_code INTEGER, kind_label TEXT,
  ge_category TEXT, ge_detail TEXT, ge_core INTEGER,
  semester_span TEXT, enroll_mode TEXT, cancelled INTEGER,
  lang TEXT, english_taught INTEGER, syllabus_lang TEXT,
  dept_code TEXT, dept_name TEXT, college TEXT, level TEXT,
  unit_raw TEXT, unit_is_class INTEGER, is_pe INTEGER,
  time_raw TEXT, time_tba INTEGER, classroom_raw TEXT, building TEXT, room_id TEXT,
  ai_policy TEXT,
  teach_lecture INTEGER, teach_discussion INTEGER, teach_group INTEGER, teach_elearning INTEGER,
  teach_other INTEGER, teach_other_label TEXT, teach_sum INTEGER,
  eval_pct_sum REAL, eval_reliable INTEGER, eval_method TEXT,
  n_weeks INTEGER, weekly_method TEXT,
  enroll_limit INTEGER, enroll_count INTEGER,
  url TEXT, note_clean TEXT, info_clean TEXT,
  syllabus TEXT, objective TEXT, schedule_clean TEXT, evaluation TEXT, textbook_clean TEXT,
  card TEXT
);
CREATE TABLE course_path (course_id TEXT, dp1 TEXT, dp2 TEXT, dp3 TEXT, label TEXT);
CREATE TABLE session (course_id TEXT, seq INTEGER, weekday INTEGER, start TEXT, end TEXT, periods TEXT,
  kind TEXT, source TEXT, tentative INTEGER, in_time_field INTEGER);
CREATE TABLE target_class (course_id TEXT, dept_abbr TEXT, level TEXT, year INTEGER, section TEXT, raw TEXT);
CREATE TABLE restriction (id INTEGER PRIMARY KEY, course_id TEXT, type TEXT, value TEXT, polarity TEXT,
  hard INTEGER, source TEXT, span TEXT, method TEXT);
CREATE TABLE prereq (restriction_id INTEGER, course_id TEXT, mention TEXT, kind TEXT, resolved_base TEXT,
  resolved_names TEXT, method TEXT, in_1142 INTEGER);
CREATE TABLE change (course_id TEXT, type TEXT, date TEXT, raw TEXT);
CREATE TABLE eval_item (course_id TEXT, seq INTEGER, category TEXT, label TEXT, pct REAL, method TEXT);
CREATE TABLE ai_policy (course_id TEXT PRIMARY KEY, category TEXT, label_zh TEXT, detail TEXT, detail_source TEXT);
CREATE TABLE weekly (course_id TEXT, week INTEGER, week_end INTEGER, date TEXT, topic TEXT, detail TEXT,
  hours_in REAL, hours_out REAL, method TEXT);
CREATE TABLE alias (alias TEXT, canonical TEXT, members TEXT, kind TEXT, source TEXT, evidence INTEGER,
  confidence REAL, in_1142 INTEGER, note TEXT);
CREATE TABLE offering (offering_id INTEGER PRIMARY KEY, name_norm TEXT, teacher TEXT, time_raw TEXT,
  primary_course_id TEXT, n_channels INTEGER);
CREATE TABLE channel (offering_id INTEGER, course_id TEXT, sub_num TEXT, unit TEXT, dept_code TEXT, college TEXT,
  level TEXT, targets TEXT, enroll_mode TEXT, audience_level TEXT, colleges_only TEXT, status_only TEXT,
  special_class TEXT, url TEXT);
CREATE INDEX ix_session ON session(course_id);
CREATE INDEX ix_target ON target_class(course_id);
CREATE INDEX ix_restr ON restriction(course_id, type);
CREATE INDEX ix_eval ON eval_item(course_id);
CREATE INDEX ix_weekly ON weekly(course_id);
CREATE INDEX ix_alias ON alias(alias);
CREATE INDEX ix_channel ON channel(course_id);
"""


def _jsonish(v):
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        return json.dumps(sorted(v) if isinstance(v, (set, frozenset)) else v, ensure_ascii=False)
    if isinstance(v, bool):
        return int(v)
    return v


def write(path: Path, tables: dict[str, list[dict]], meta: dict[str, str]) -> None:
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
        for table, rows in tables.items():
            if not rows:
                continue
            cols = list(rows[0])
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
            con.executemany(sql, [tuple(_jsonish(r.get(c)) for c in cols) for r in rows])
        con.commit()
    finally:
        con.close()
    tmp.replace(path)
