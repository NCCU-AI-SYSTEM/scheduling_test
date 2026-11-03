"""Migrate courses from SQLite (data/1142.db) to PostgreSQL.

Usage:
    uv run python scripts/migrate_sqlite_to_pg.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 確保 src 可以被 import（在專案根目錄執行時自動生效）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2.extras

from src.db.pg_client import get_connection
from src.loader.courses import load_courses

BATCH_SIZE = 200


def main() -> None:
    print("載入課程資料（y=114, s=2）...")
    courses = load_courses(year="114", semester="2")
    print(f"  共讀取 {len(courses)} 門課程")

    conn = get_connection()
    conn.autocommit = False
    cur = conn.cursor()

    success = 0
    failure = 0
    batch: list[tuple] = []

    INSERT_SQL = """
        INSERT INTO courses (
            course_id, year, semester, name, name_en, teacher,
            kind, lmt_kind, lang, point, unit, time_raw,
            sessions, weekdays,
            has_morning, has_noon, has_afternoon, has_evening,
            embedding,
            info, note, objective, syllabus
        ) VALUES %s
        ON CONFLICT (course_id) DO NOTHING
    """

    def flush_batch(b: list[tuple]) -> tuple[int, int]:
        if not b:
            return 0, 0
        try:
            psycopg2.extras.execute_values(cur, INSERT_SQL, b, page_size=BATCH_SIZE)
            conn.commit()
            return len(b), 0
        except Exception as e:
            conn.rollback()
            print(f"  [batch error] {e}", file=sys.stderr)
            return 0, len(b)

    for course in courses:
        try:
            sessions = course.sessions  # list[Session]
            sessions_json = json.dumps(
                [
                    {
                        "weekday": s.weekday,
                        "start_hour": s.start_hour,
                        "end_hour": s.end_hour,
                    }
                    for s in sessions
                ],
                ensure_ascii=False,
            )
            weekdays = list(set(s.weekday for s in sessions))
            has_morning   = any(8  <= s.start_hour < 12 for s in sessions)
            has_noon      = any(12 <= s.start_hour < 13 for s in sessions)
            has_afternoon = any(13 <= s.start_hour < 18 for s in sessions)
            has_evening   = any(s.start_hour >= 18       for s in sessions)

            row = (
                course.course_id,
                course.year,
                course.semester,
                course.name,
                course.name_en,
                course.teacher,
                course.kind,
                course.lmt_kind,
                course.lang,
                course.point,
                course.unit,
                course.time_raw,
                sessions_json,          # JSONB
                weekdays,               # INT[]
                has_morning,
                has_noon,
                has_afternoon,
                has_evening,
                None,                   # embedding → NULL
                course.info,
                course.note,
                course.objective,
                course.syllabus,
            )
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                ok, fail = flush_batch(batch)
                success += ok
                failure += fail
                batch = []

        except Exception as e:
            print(f"  [row error] course_id={course.course_id}: {e}", file=sys.stderr)
            failure += 1

    # 最後一批
    ok, fail = flush_batch(batch)
    success += ok
    failure += fail

    cur.close()
    conn.close()

    print(f"\n移植完成：成功 {success} 筆，失敗 {failure} 筆")


if __name__ == "__main__":
    main()
