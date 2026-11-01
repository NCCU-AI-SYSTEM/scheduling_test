"""UMBRELA-style graded relevance judge for course search, via local Ollama.

Judges only the *topical* need of a query (the part left after structured
conditions are stripped). Hard conditions are checked against the DB by
`src.eval.conditions`, never by the LLM — LLMs are unreliable on negation and
cheap to replace with an exact check here.

Grades (UMBRELA, Upadhyay et al. 2024):
  0 無關  1 沾邊但不是學生要找的  2 相關，會列入考慮  3 完全切題

Results are cached in SQLite keyed by (model, prompt version, need, course_id)
so re-running an eval build never re-judges a pair.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DB = ROOT / "data" / "processed" / "judge_cache.db"
PROMPT_VERSION = "umbrela-zh-v1"
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "qwen3:14b")

_PROMPT = """你是政治大學的選課助理評審。學生在課程搜尋系統輸入了一段需求，請判斷下面這門課與「學生想學的內容」有多相關。

只評估主題/內容是否符合，不要考慮上課時間、語言、學分、系所等條件（那些另外檢查）。

評分標準：
0 = 無關：課程內容與需求無關。
1 = 沾邊：有一點關聯，但學生大概不會想選這門課來滿足此需求。
2 = 相關：課程有相當部分符合需求，學生會把它列入考慮。
3 = 完全切題：課程正是學生要找的。

學生需求：{need}

課程資料：
{doc}

請只輸出一個數字（0、1、2 或 3），不要輸出其他文字。"""

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS judge ("
        " model TEXT, version TEXT, need TEXT, course_id TEXT, grade INTEGER, raw TEXT,"
        " PRIMARY KEY (model, version, need, course_id))"
    )
    return conn


_db = None


def _get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = _conn()
    return _db


def cached(need: str, course_id: str, model: str = DEFAULT_MODEL) -> int | None:
    with _lock:
        row = _get_db().execute(
            "SELECT grade FROM judge WHERE model=? AND version=? AND need=? AND course_id=?",
            (model, PROMPT_VERSION, need, course_id),
        ).fetchone()
    return None if row is None else row[0]


def _parse_grade(text: str) -> int | None:
    m = re.search(r"[0-3]", text)
    return int(m.group()) if m else None


def judge(need: str, course_id: str, doc_text: str, model: str = DEFAULT_MODEL) -> int:
    """Return 0-3. Unparseable output is stored as -1 and treated as 0 by callers."""
    hit = cached(need, course_id, model)
    if hit is not None:
        return hit
    import ollama

    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": _PROMPT.format(need=need, doc=doc_text)}],
        think=False,
        options={"temperature": 0, "num_predict": 4, "num_ctx": 4096},
        keep_alive="30m",
    )
    raw = resp["message"]["content"]
    grade = _parse_grade(raw)
    grade = -1 if grade is None else grade
    with _lock:
        _get_db().execute(
            "INSERT OR REPLACE INTO judge VALUES (?,?,?,?,?,?)",
            (model, PROMPT_VERSION, need, course_id, grade, raw),
        )
        _get_db().commit()
    return grade
