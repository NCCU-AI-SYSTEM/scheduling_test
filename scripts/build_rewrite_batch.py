"""Build OpenAI batch input for HyDE and Q2D query rewrites.

Each request: given a query, produce a hypothetical course description (HyDE)
or an expanded passage (Q2D). Results are cached to disk in the same format
as src/query_rewriters/llm.py so run_experiment.py picks them up for free.

Usage:
    uv run python scripts/build_rewrite_batch.py
    # -> batches/rewrite_eval.jsonl  (1000 reqs = 500 HyDE + 500 Q2D)
    # -> batches/rewrite_eval_queries.json  (qid -> query mapping for merge)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"

HYDE_SYSTEM = (
    "你是政治大學課程資訊助教。"
    "依使用者的查詢，撰寫一段 80 至 150 字的繁體中文課程簡介，"
    "假裝這就是符合查詢的課程，"
    "內容要含主題、學科術語、可能的週次主題與學習目標。"
    "只輸出該段文字，不加標題、不加引號。"
)

Q2D_SYSTEM = (
    "你是政治大學課程資訊助教。"
    "依使用者的查詢，產生一段 100 字左右的擴展段落，"
    "段落必須包含原查詢字串，加入 3-5 個相關學科術語、口語別稱、可能的課程內容。"
    "只輸出段落本文。"
)


def _req(custom_id: str, system: str, query: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"查詢：{query}"},
            ],
            "temperature": 0.4,
            "max_tokens": 300,
        },
    }


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.eval import from_jsonl
    import random
    queries_all = from_jsonl()
    rng = random.Random(42)
    rng.shuffle(queries_all)
    queries = queries_all[:500]
    print(f"[batch] {len(queries)} queries")

    # save qid -> query mapping for merge step
    qmap = {q.qid: q.query for q in queries}
    (BATCH_DIR / "rewrite_eval_queries.json").write_text(
        json.dumps(qmap, ensure_ascii=False, indent=2)
    )

    out_path = BATCH_DIR / "rewrite_eval.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(_req(f"hyde|{q.qid}", HYDE_SYSTEM, q.query), ensure_ascii=False) + "\n")
            f.write(json.dumps(_req(f"q2d|{q.qid}", Q2D_SYSTEM, q.query), ensure_ascii=False) + "\n")

    print(f"[batch] wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB, {len(queries)*2} reqs)")


if __name__ == "__main__":
    main()
