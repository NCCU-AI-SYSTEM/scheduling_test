"""Generate per-course metadata via gemma4:e4b on home_mac.

Output schema (cached in SQLite `course_meta_v1`):
  summary_100   : 100-char Chinese summary of the course
  keywords      : list[str], 6-10 学科关键词 / 课程关键词
  topic_tags    : list[str], coarse topic tags from a controlled vocab
  level         : "入門" | "進階" | "研究所" | "未知"
  prereq_inferred : str, suggested prerequisites or ""

CLI:
    uv run python -m src.llm.meta_gen --year 114 --semester 2 [--limit N] [--force] [--workers 2]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.llm.ollama_client import DEFAULT_MODEL, chat_json, health_check
from src.loader import Course, load_courses

ROOT = Path(__file__).resolve().parents[2]
META_DB = ROOT / "data" / "processed" / "course_meta.db"

CONTROLLED_TAGS = [
    "AI", "資料科學", "機器學習", "程式設計", "數學", "統計",
    "語言學習", "文學", "歷史", "哲學", "宗教",
    "經濟", "金融", "會計", "商業管理", "行銷", "創業",
    "法律", "政治", "公共行政", "國際關係", "新聞傳播",
    "心理", "社會", "教育", "民族",
    "藝術", "音樂", "設計", "影視",
    "體育", "通識", "其他",
]

SYSTEM_PROMPT = (
    "你是政治大學課程資訊整理助手。"
    "輸出嚴格 JSON，不附加文字。"
    "所有文字使用繁體中文。"
)

USER_TEMPLATE = """請根據以下課程資訊產出檢索用 metadata。

[課名] {name}
[英文名] {name_en}
[教師] {teacher}
[學分] {point}　[語言] {lang}　[必選修代碼] {kind}　[類別] {lmt_kind}
[開課單位] {unit}

[課程目標]
{objective}

[課程進度/週次]
{schedule}

請輸出 JSON：
{{
  "summary_100": "100 字以內的繁中摘要，避免抄課名",
  "keywords": ["6 到 10 個檢索關鍵詞，含學科術語與口語別稱"],
  "topic_tags": ["從 {tags} 中挑 1~3 個最相關的"],
  "level": "入門 | 進階 | 研究所 | 未知",
  "prereq_inferred": "可能需要的先備課，沒有就空字串"
}}"""


def init_db(path: Path = META_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS course_meta_v1 (
          course_id TEXT PRIMARY KEY,
          summary_100 TEXT,
          keywords_json TEXT,
          topic_tags_json TEXT,
          level TEXT,
          prereq_inferred TEXT,
          model TEXT,
          generated_at TEXT,
          raw_json TEXT
        )
        """
    )
    con.commit()
    return con


def existing_ids(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT course_id FROM course_meta_v1")}


def build_messages(c: Course) -> list[dict]:
    user = USER_TEMPLATE.format(
        name=c.name,
        name_en=c.name_en,
        teacher=c.teacher,
        point=c.point,
        lang=c.lang,
        kind=c.kind,
        lmt_kind=c.lmt_kind or "—",
        unit=c.unit,
        objective=(c.objective or "（無）")[:600],
        schedule=(c.schedule or c.syllabus or "（無）")[:600],
        tags="、".join(CONTROLLED_TAGS),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_one(c: Course, model: str) -> dict:
    msgs = build_messages(c)
    obj = chat_json(msgs, model=model, temperature=0.2, num_ctx=4096, max_retries=2)
    return obj


def upsert(con: sqlite3.Connection, course_id: str, model: str, obj: dict) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO course_meta_v1
          (course_id, summary_100, keywords_json, topic_tags_json, level,
           prereq_inferred, model, generated_at, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
        """,
        (
            course_id,
            (obj.get("summary_100") or "")[:300],
            json.dumps(obj.get("keywords") or [], ensure_ascii=False),
            json.dumps(obj.get("topic_tags") or [], ensure_ascii=False),
            obj.get("level") or "未知",
            obj.get("prereq_inferred") or "",
            model,
            json.dumps(obj, ensure_ascii=False),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="114")
    p.add_argument("--semester", default="2")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0, help="0 = all remaining")
    p.add_argument("--force", action="store_true", help="regenerate even if cached")
    p.add_argument("--commit-every", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)

    health_check(model=args.model)
    print(f"[meta] model={args.model}  workers={args.workers}  db={META_DB}")

    courses = load_courses(year=args.year, semester=args.semester)
    con = init_db(META_DB)
    done = set() if args.force else existing_ids(con)
    todo = [c for c in courses if c.course_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[meta] total={len(courses)}  cached={len(done)}  todo={len(todo)}")

    fail = 0
    db_lock = threading.Lock()
    counter = {"n": 0}
    t0 = time.time()

    def work(c: Course) -> tuple[Course, dict | None, str | None]:
        try:
            obj = generate_one(c, args.model)
            return c, obj, None
        except Exception as e:  # noqa: BLE001
            return c, None, str(e)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(work, c) for c in todo]
        with tqdm(total=len(todo), desc="meta_gen") as pbar:
            for fut in as_completed(futures):
                c, obj, err = fut.result()
                if err is not None:
                    fail += 1
                    tqdm.write(f"[fail] {c.course_id} {c.name[:20]}: {err}")
                else:
                    with db_lock:
                        upsert(con, c.course_id, args.model, obj)
                        counter["n"] += 1
                        if counter["n"] % args.commit_every == 0:
                            con.commit()
                pbar.update(1)
    con.commit()
    con.close()

    dt = time.time() - t0
    rate = (len(todo) - fail) / dt if dt > 0 else 0
    print(
        f"[meta] done {len(todo)-fail}/{len(todo)}  fail={fail}  "
        f"elapsed={dt:.1f}s  rate={rate:.2f} req/s"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
