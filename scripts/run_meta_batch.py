"""Build + submit OpenAI batch for course metadata generation (W2).

Generates summary, keywords, topic_tags, level, prereq for each course.
Results saved to course_meta_v1 SQLite (same schema as meta_gen.py).

Usage:
    uv run python scripts/run_meta_batch.py --build    # build jsonl
    uv run python scripts/run_meta_batch.py --submit
    uv run python scripts/run_meta_batch.py --status
    uv run python scripts/run_meta_batch.py --fetch
    uv run python scripts/run_meta_batch.py --merge    # write to course_meta.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

BATCH_DIR = ROOT / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PATH = BATCH_DIR / "meta_eval.jsonl"
RESULT_PATH = BATCH_DIR / "meta_eval.results.jsonl"
STATE_PATH = BATCH_DIR / "state" / "meta.json"
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
META_DB = ROOT / "data" / "processed" / "course_meta.db"

MODEL = "gpt-4o-mini"

CONTROLLED_TAGS = [
    "AI", "資料科學", "機器學習", "程式設計", "數學", "統計",
    "語言學習", "文學", "歷史", "哲學", "宗教",
    "經濟", "金融", "會計", "商業管理", "行銷", "創業",
    "法律", "政治", "公共行政", "國際關係", "新聞傳播",
    "心理", "社會", "教育", "民族",
    "藝術", "音樂", "設計", "影視",
    "體育", "通識", "其他",
]

SYSTEM = (
    "你是政治大學課程資訊整理助手。"
    "輸出嚴格 JSON，不附加文字。"
    "所有文字使用繁體中文。"
)

USER_TMPL = """請根據以下課程資訊產出檢索用 metadata。

[課名] {name}
[教師] {teacher}
[學分] {point}　[語言] {lang}
[開課單位] {unit}

[課程目標]
{objective}

請輸出 JSON：
{{
  "summary_100": "100 字以內的繁中摘要，避免抄課名",
  "keywords": ["6 到 10 個檢索關鍵詞"],
  "topic_tags": ["從以下選 1~3 個：{tags}"],
  "level": "入門 | 進階 | 研究所 | 未知",
  "prereq_inferred": "可能需要的先備課，沒有就空字串"
}}"""


def _save_state(obj: dict) -> None:
    STATE_PATH.write_text(json.dumps(obj, indent=2))


def _load_state() -> dict:
    return json.loads(STATE_PATH.read_text())


def _existing_ids(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT course_id FROM course_meta_v1")}


def _init_db() -> sqlite3.Connection:
    META_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(META_DB)
    con.execute("""
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
    """)
    con.commit()
    return con


def do_build(force: bool = False) -> None:
    from src.loader import load_courses
    courses = load_courses()
    con = _init_db()
    done = set() if force else _existing_ids(con)
    con.close()
    todo = [c for c in courses if c.course_id not in done]
    print(f"[meta-batch] total={len(courses)} cached={len(done)} todo={len(todo)}")

    tags_str = "、".join(CONTROLLED_TAGS)
    with INPUT_PATH.open("w", encoding="utf-8") as f:
        for c in todo:
            obj = (c.objective or "（無）")[:600]
            user = USER_TMPL.format(
                name=c.name, teacher=c.teacher,
                point=c.point, lang=c.lang or "—",
                unit=c.unit, objective=obj, tags=tags_str,
            )
            req = {
                "custom_id": c.course_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    kb = INPUT_PATH.stat().st_size / 1024
    print(f"[meta-batch] wrote {INPUT_PATH}  ({kb:.0f} KB, {len(todo)} reqs)")


def do_submit() -> None:
    from openai import OpenAI
    client = OpenAI()
    print(f"[meta-batch] uploading {INPUT_PATH}")
    f = client.files.create(file=INPUT_PATH.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "nccu-course-meta"},
    )
    print(f"[meta-batch] batch_id={batch.id}  status={batch.status}")
    _save_state({"file_id": f.id, "batch_id": batch.id, "status": batch.status})


def do_status() -> dict:
    from openai import OpenAI
    client = OpenAI()
    state = _load_state()
    b = client.batches.retrieve(state["batch_id"])
    rc = b.request_counts
    print(f"[meta-batch] {b.id}  status={b.status}  req={rc.completed}/{rc.total}  failed={rc.failed}")
    state["status"] = b.status
    state["output_file_id"] = b.output_file_id
    _save_state(state)
    return state


def do_fetch() -> None:
    from openai import OpenAI
    client = OpenAI()
    state = _load_state()
    if state.get("status") != "completed":
        state = do_status()
    if state["status"] != "completed":
        raise RuntimeError(f"Not completed: {state['status']}")
    content = client.files.content(state["output_file_id"])
    RESULT_PATH.write_bytes(content.read())
    print(f"[meta-batch] wrote {RESULT_PATH}  ({RESULT_PATH.stat().st_size/1024:.1f} KB)")


def do_merge() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError("Run --fetch first.")
    con = _init_db()
    ok = fail = 0
    with RESULT_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            course_id = rec["custom_id"]
            try:
                text = rec["response"]["body"]["choices"][0]["message"]["content"]
                obj = json.loads(text)
            except (KeyError, IndexError, json.JSONDecodeError):
                fail += 1
                continue
            con.execute("""
                INSERT OR REPLACE INTO course_meta_v1
                  (course_id, summary_100, keywords_json, topic_tags_json, level,
                   prereq_inferred, model, generated_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
            """, (
                course_id,
                (obj.get("summary_100") or "")[:300],
                json.dumps(obj.get("keywords") or [], ensure_ascii=False),
                json.dumps(obj.get("topic_tags") or [], ensure_ascii=False),
                obj.get("level") or "未知",
                obj.get("prereq_inferred") or "",
                MODEL,
                json.dumps(obj, ensure_ascii=False),
            ))
            ok += 1
    con.commit()
    con.close()
    total = ok + fail
    print(f"[meta-batch] merged ok={ok} fail={fail} total={total}")
    cur = sqlite3.connect(META_DB).execute("SELECT COUNT(*) FROM course_meta_v1").fetchone()[0]
    print(f"[meta-batch] course_meta_v1 total rows: {cur}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--merge", action="store_true")
    p.add_argument("--force", action="store_true", help="re-generate even if cached")
    args = p.parse_args()
    if args.build:
        do_build(force=args.force)
    elif args.submit:
        do_submit()
    elif args.status:
        do_status()
    elif args.fetch:
        do_fetch()
    elif args.merge:
        do_merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
