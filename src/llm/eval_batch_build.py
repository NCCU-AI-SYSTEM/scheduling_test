"""Build batch input files for OpenAI Batch API and Anthropic Message Batches API.

Outputs:
  batches/openai_eval.jsonl       (model: gpt-4.1-mini)
  batches/anthropic_eval.jsonl    (model: claude-opus-4)

Each line corresponds to one course. The model is asked to produce 3 query
personas (topic / constraint / colloquial), so we get 3 (query, gold_courseId)
pairs per call. With both providers running in parallel we get 6 per course.

Usage:
    uv run python -m src.llm.eval_batch_build --year 114 --semester 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.loader import Course, load_courses

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL = "gpt-4.1-mini"
ANTHROPIC_MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = (
    "你是政治大學選課學生，會用日常口吻搜尋自己想修的課。"
    "依使用者提供的課程資訊，產生三個不同類型的查詢，皆使用繁體中文。"
    "輸出嚴格 JSON，無額外文字。"
)

USER_TEMPLATE = """這是一門政大課程，請站在不知道課名的學生角度，產生三個查詢。

[課程資訊]
課名：{name}
教師：{teacher}
單位：{unit}
語言：{lang}　學分：{point}
課程目標：{objective}
週次主題：{schedule}

請輸出 JSON：
{{
  "topic": "用主題描述想找這類課（不可直接抄課名，例：『想學 X』『關於 X 的課』）",
  "constraint": "帶生活/時間/語言/院系約束的詢問（例：『週二下午的XX』、『英文授課的XX』）",
  "colloquial": "口語、不正式、可能有錯別字的同義詢問"
}}

每個 query 至少 6 個字、不超過 30 字。不要直接出現課名原字串。"""


def _course_payload(c: Course) -> tuple[str, str]:
    objective = (c.objective or "（無）")[:500]
    schedule = (c.schedule or c.syllabus or "（無）")[:300]
    return (
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(
            name=c.name,
            teacher=c.teacher,
            unit=c.unit,
            lang=c.lang or "—",
            point=c.point,
            objective=objective,
            schedule=schedule,
        ),
    )


def write_openai_jsonl(courses: list[Course], path: Path) -> None:
    """OpenAI Batch API format. /v1/chat/completions, custom_id = course_id."""
    with path.open("w", encoding="utf-8") as f:
        for c in courses:
            sys_p, user_p = _course_payload(c)
            req = {
                "custom_id": c.course_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ],
                    "temperature": 0.8,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def write_anthropic_jsonl(courses: list[Course], path: Path) -> None:
    """Anthropic Message Batches format.
    https://docs.anthropic.com/en/api/creating-message-batches
    """
    with path.open("w", encoding="utf-8") as f:
        for c in courses:
            sys_p, user_p = _course_payload(c)
            req = {
                "custom_id": c.course_id,
                "params": {
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 400,
                    "temperature": 0.8,
                    "system": sys_p,
                    "messages": [{"role": "user", "content": user_p}],
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="114")
    p.add_argument("--semester", default="2")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default=str(BATCH_DIR))
    args = p.parse_args(argv)

    courses = load_courses(year=args.year, semester=args.semester)
    if args.limit:
        courses = courses[: args.limit]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    openai_path = out / "openai_eval.jsonl"
    anthropic_path = out / "anthropic_eval.jsonl"

    write_openai_jsonl(courses, openai_path)
    write_anthropic_jsonl(courses, anthropic_path)

    print(f"[batch] {openai_path}  ({openai_path.stat().st_size/1024:.1f} KB, {len(courses)} reqs)")
    print(
        f"[batch] {anthropic_path}  "
        f"({anthropic_path.stat().st_size/1024:.1f} KB, {len(courses)} reqs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
