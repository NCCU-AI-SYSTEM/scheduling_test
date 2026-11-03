"""Generate short student-style questions per course for question-to-question retrieval.

HyPE (Vake et al., 2025) indexes hypothetical questions instead of (or beside) the
document; Doc2Query-- (Gospodinov et al., 2023) shows filtering off-topic generations
matters. Real student queries are short (median 16 characters in the forum bank),
while syllabi run to thousands, so matching question-to-question narrows the gap
without any query-time LLM call.

  python -m scripts.v4.analysis.gen_hype --model gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.llm.ollama_http import chat
from src.v4.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "v4" / "hype"

PROMPT = """下面是一門大學課程的資料。請想像 8 位不同的政大學生，在還不知道課名的情況下，會怎麼向選課助理描述自己想找的課，每人寫一句。
要求：
- 口語、簡短（6 到 18 個字），像「想學怎麼做市場調查」「有沒有教 Python 爬蟲的課」
- 不要寫出課名、老師名、課號
- 涵蓋不同角度：想學的主題、想練的技能、未來用途、課程形式（例如實作、討論）
- 只根據課程資料，不要編造課程沒有的內容
輸出 JSON：{"questions": ["...", ...]}

課名：{name}
開課單位：{unit}
課程目標：{objective}
週次主題：{topics}"""

SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
          "required": ["questions"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    cat = load_catalog(with_reviews=False)
    if a.limit:
        cat = cat[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{re.sub(r'[^A-Za-z0-9.]+', '_', a.model)}.jsonl"
    done = {json.loads(x)["course_id"] for x in path.open(encoding="utf-8")} if path.exists() else set()

    def one(c):
        x = c.course
        text = (PROMPT.replace("{name}", x.name).replace("{unit}", x.unit)
                .replace("{objective}", (x.objective or x.syllabus or "")[:700])
                .replace("{topics}", "、".join(c.topics)[:400]))  # str.format trips on the JSON example
        res = chat(a.model, [{"role": "user", "content": text}], fmt=SCHEMA, temperature=0.7, timeout=300)
        try:
            qs = json.loads(res.content).get("questions", [])
        except json.JSONDecodeError:
            qs = []
        name = re.sub(r"[（(].*?[)）]", "", x.name)
        qs = [q.strip() for q in qs if isinstance(q, str) and 4 <= len(q.strip()) <= 30 and name not in q][:10]
        return {"course_id": c.course_id, "questions": qs, "seconds": round(res.seconds, 2), "error": res.error}

    todo = [c for c in cat if c.course_id not in done]
    with ThreadPoolExecutor(a.workers) as ex, path.open("a", encoding="utf-8") as f:
        for row in tqdm(ex.map(one, todo), total=len(todo)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print("wrote", path)


if __name__ == "__main__":
    main()
