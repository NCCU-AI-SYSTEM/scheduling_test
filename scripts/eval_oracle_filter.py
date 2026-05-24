"""
Oracle Filter 實驗（Phase 4）

用 eval_conditions_v1.jsonl 的 ground truth conditions（must/must_not）
直接對 PostgreSQL 做 SQL filter，驗證：

實驗 A：gold_in_pool rate
  gold_course_id 有沒有在 filtered 結果裡？
  目標：100%（oracle 條件不應誤殺 gold）

實驗 B：pool_size 分析
  filter 後候選池縮小多少？按維度組合分析

使用：
  uv run python scripts/eval_oracle_filter.py
  uv run python scripts/eval_oracle_filter.py --limit 100  # 只跑前 100 筆
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.pg_client import get_connection
from src.db.filter import conditions_to_sql
from src.parsers.base import ConditionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_PATH = ROOT / "data/raw/eval_conditions_v1.jsonl"


def row_to_condition(row: dict) -> ConditionResult:
    """把 eval jsonl 的 must/must_not 轉成 ConditionResult。"""
    return ConditionResult(
        must=row.get("must", {}),
        must_not=row.get("must_not", {}),
        should=row.get("should", {}),
    )


def run_filter(con, where: str, params: list) -> tuple[list[str], int]:
    """執行 SQL filter，回傳 (course_ids, pool_size)。"""
    sql = f"SELECT course_id FROM courses WHERE {where}"
    cur = con.cursor()
    cur.execute(sql, params)
    ids = [r[0] for r in cur.fetchall()]
    return ids, len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 筆（0=全部）")
    parser.add_argument("--eval",  default=str(EVAL_PATH))
    args = parser.parse_args()

    # 載入 eval 資料
    rows = []
    with open(args.eval) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if args.limit:
        rows = rows[: args.limit]
    log.info(f"Loaded {len(rows)} eval rows")

    con = get_connection()
    total_courses = con.cursor()
    total_courses.execute("SELECT COUNT(*) FROM courses")
    n_total = total_courses.fetchone()[0]
    log.info(f"Total courses in DB: {n_total}")

    # ── 實驗 A + B ────────────────────────────────────────────────────────────
    results = []
    n_gold_in_pool = 0
    n_gold_missing = 0
    pool_sizes     = []
    miss_examples  = []

    # 按維度組合統計
    dim_stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "gold_in_pool": 0, "pool_sizes": []
    })

    for i, row in enumerate(rows):
        cond    = row_to_condition(row)
        gold_id = row["gold_course_id"]

        # 建 SQL
        where, params = conditions_to_sql(cond)

        # 執行 filter
        pool_ids, pool_size = run_filter(con, where, params)
        gold_in_pool = gold_id in pool_ids

        # 記錄結果
        pool_sizes.append(pool_size)
        if gold_in_pool:
            n_gold_in_pool += 1
        else:
            n_gold_missing += 1
            if len(miss_examples) < 10:
                miss_examples.append({
                    "qid":      row["qid"],
                    "query_zh": row["queries"]["zh"],
                    "gold_id":  gold_id,
                    "must":     row["must"],
                    "must_not": row["must_not"],
                    "pool_size": pool_size,
                })

        # 維度組合 key
        must_dims    = "+".join(sorted(row["must"].keys()))    or "(none)"
        must_not_dims= "+".join(sorted(row["must_not"].keys()))or ""
        dim_key = f"must=[{must_dims}]" + (f" must_not=[{must_not_dims}]" if must_not_dims else "")
        dim_stats[dim_key]["total"]       += 1
        dim_stats[dim_key]["pool_sizes"].append(pool_size)
        if gold_in_pool:
            dim_stats[dim_key]["gold_in_pool"] += 1

        if (i + 1) % 200 == 0:
            log.info(f"Progress: {i+1}/{len(rows)}, "
                     f"gold_in_pool={n_gold_in_pool}, miss={n_gold_missing}")

    con.close()

    # ── 輸出報告 ──────────────────────────────────────────────────────────────
    n_total_eval   = len(rows)
    gold_in_rate   = n_gold_in_pool / n_total_eval * 100
    avg_pool       = sum(pool_sizes) / len(pool_sizes)
    avg_pool_pct   = avg_pool / n_total * 100
    median_pool    = sorted(pool_sizes)[len(pool_sizes) // 2]

    print("\n" + "=" * 60)
    print("Oracle Filter 實驗結果")
    print("=" * 60)
    print(f"\n【實驗 A：gold_in_pool rate】")
    print(f"  總筆數      : {n_total_eval}")
    print(f"  gold 在 pool : {n_gold_in_pool} ({gold_in_rate:.2f}%)")
    print(f"  gold 不在 pool: {n_gold_missing} ({100-gold_in_rate:.2f}%)")

    print(f"\n【實驗 B：pool size】")
    print(f"  全庫課程數  : {n_total}")
    print(f"  平均 pool   : {avg_pool:.1f} ({avg_pool_pct:.1f}% 全庫)")
    print(f"  中位數 pool : {median_pool}")
    print(f"  最小 pool   : {min(pool_sizes)}")
    print(f"  最大 pool   : {max(pool_sizes)}")

    print(f"\n【按維度組合分析（top 15）】")
    sorted_dims = sorted(dim_stats.items(), key=lambda x: -x[1]["total"])
    for dim_key, stat in sorted_dims[:15]:
        t  = stat["total"]
        gp = stat["gold_in_pool"]
        ap = sum(stat["pool_sizes"]) / t
        print(f"  {dim_key}")
        print(f"    n={t}  gold_in_pool={gp/t*100:.1f}%  avg_pool={ap:.0f}")

    if miss_examples:
        print(f"\n【gold 不在 pool 的範例（前 {len(miss_examples)} 筆）】")
        for ex in miss_examples:
            print(f"  qid: {ex['qid']}")
            print(f"  query: {ex['query_zh']}")
            print(f"  must={ex['must']}  must_not={ex['must_not']}")
            print(f"  pool_size={ex['pool_size']}")
            print()

    print("=" * 60)

    # 儲存結果
    out = ROOT / "results/tables/oracle_filter_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_eval":          n_total_eval,
        "n_gold_in_pool":  n_gold_in_pool,
        "gold_in_rate":    round(gold_in_rate, 4),
        "n_gold_missing":  n_gold_missing,
        "avg_pool_size":   round(avg_pool, 1),
        "median_pool_size": median_pool,
        "min_pool_size":   min(pool_sizes),
        "max_pool_size":   max(pool_sizes),
        "total_db_courses": n_total,
        "miss_examples":   miss_examples,
    }
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info(f"Saved to {out}")


if __name__ == "__main__":
    main()
