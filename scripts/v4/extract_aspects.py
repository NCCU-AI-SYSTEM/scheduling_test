"""Turn each course's past student reviews into aspect scores (1-5 or null).

Reviews are linked by subNum or (course name, teacher) across 101-1..114-1. The LLM
reads at most 40 non-trivial comments per course and returns per-aspect scores with
a short evidence quote, so a later answer can show why a course was called 涼 or 甜.

  python -m scripts.v4.extract_aspects --model qwen3.5:9b --min-reviews 3
  python -m scripts.v4.extract_aspects --model gpt-oss:120b-cloud --sample 60   # reference subset
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.llm.ollama_http import chat
from src.v4.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "v4" / "aspects"
ASPECTS = ["workload", "grading", "difficulty", "attendance", "group_work", "teaching", "usefulness"]

PROMPT = """以下是政大某門課（同一老師）歷屆學生的期末評鑑短評。請依評論內容替這門課的各面向打分（1–5 的整數），評論沒有提到的面向填 null，不要猜。
- workload：作業、報告、考試等負擔（1 很輕鬆、5 很重）
- grading：給分寬鬆程度（1 很嚴、5 很甜）
- difficulty：內容難度（1 很簡單、5 很難）
- attendance：點名或出席要求（1 不點名、5 很嚴格）
- group_work：分組作業的比重（1 幾乎沒有、5 很多）
- teaching：學生對老師教學的整體評價（1 很差、5 很好）
- usefulness：學生覺得實用、有收穫的程度（1 沒收穫、5 收穫很多）
另外寫 summary：30 字以內的中文摘要，只根據評論。

課程：{name}（{teacher}）
評論：
{reviews}"""

SCHEMA = {"type": "object", "properties": {
    **{a: {"type": ["integer", "null"]} for a in ASPECTS}, "summary": {"type": "string"}},
    "required": ASPECTS + ["summary"]}


def _clean(v):
    return v if isinstance(v, int) and 1 <= v <= 5 else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--min-reviews", type=int, default=3)
    p.add_argument("--max-reviews", type=int, default=40)
    p.add_argument("--sample", type=int, default=0, help="random subset of courses (reference runs)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    cat = [c for c in load_catalog() if len(c.reviews) >= a.min_reviews]
    # one call per (course name, teacher): sections of the same family share reviews
    fam: dict[tuple, list] = {}
    for c in cat:
        fam.setdefault(c.family, []).append(c)
    items = list(fam.items())
    if a.sample:
        random.Random(a.seed).shuffle(items)
        items = items[:a.sample]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{re.sub(r'[^A-Za-z0-9.]+', '_', a.model)}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["family"] for x in out_path.open(encoding="utf-8")}

    def one(item):
        key, courses = item
        c = courses[0]
        revs = c.reviews[-a.max_reviews:]  # most recent offerings last in past_ids order
        res = chat(a.model, [{"role": "user", "content": PROMPT.format(
            name=c.course.name, teacher=c.course.teacher,
            reviews="\n".join(f"- {r[:150]}" for r in revs))}], fmt=SCHEMA, timeout=300)
        try:
            d = json.loads(res.content)
        except json.JSONDecodeError:
            d = {}
        return {"family": "|".join(key), "course_ids": [x.course_id for x in courses], "n_reviews": len(c.reviews),
                **{asp: _clean(d.get(asp)) for asp in ASPECTS}, "summary": (d.get("summary") or "")[:80],
                "seconds": round(res.seconds, 2), "error": res.error}

    todo = [it for it in items if "|".join(it[0]) not in done]
    with ThreadPoolExecutor(a.workers) as ex, out_path.open("a", encoding="utf-8") as f:
        for row in tqdm(ex.map(one, todo), total=len(todo)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
