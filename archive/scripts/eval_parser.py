"""
Parser 實驗（Phase 4 後半）

對 eval_conditions_v1.jsonl 的每筆 queries.zh，
用 P0~P4 各 parser 解析，比較與 ground truth 的差距。

指標：
  - Slot F1（每個維度分開）
  - Negation F1（must_not 整體）
  - Exact Match

使用：
  uv run python scripts/eval_parser.py
  uv run python scripts/eval_parser.py --limit 200
  uv run python scripts/eval_parser.py --parsers p0,p1,p2,p3,p4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parsers.base import ConditionResult
from src.parsers import rule_v0, regex_v1
from src.eval.parser_metrics import slot_f1, negation_f1, exact_match, aggregate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_PATH = ROOT / "data/raw/eval_conditions_v1.jsonl"
DIMS = ["course_lang", "weekday", "hour_range", "point", "kind", "lmt_kind", "unit"]


def load_parser(name: str, model: str | None = None):
    """
    動態載入 parser，回傳 (parse_fn, label)。
    支援 p2@gpt-4.1-mini 格式指定 model。
    """
    if "@" in name:
        name, model = name.split("@", 1)

    if name == "p0":
        return rule_v0.parse, "P0 Rule-based"
    if name == "p1":
        return regex_v1.parse, "P1 Regex+否定詞"

    from src.parsers import llm_zeroshot, llm_fewshot, llm_structured

    if name == "p2":
        label = f"P2 zero-shot [{model or 'default'}]"
        fn = lambda q, _m=model: llm_zeroshot.parse(q, model=_m)
        return fn, label
    if name == "p3":
        label = f"P3 few-shot [{model or 'default'}]"
        fn = lambda q, _m=model: llm_fewshot.parse(q, model=_m)
        return fn, label
    if name == "p4":
        label = f"P4 structured [{model or 'default'}]"
        fn = lambda q, _m=model: llm_structured.parse(q, model=_m)
        return fn, label
    raise ValueError(f"Unknown parser: {name}")


def eval_one(pred: ConditionResult, gold_row: dict) -> dict:
    """比較一筆 pred 和 gold，回傳各項指標。"""
    gold = ConditionResult(
        must=gold_row.get("must", {}),
        must_not=gold_row.get("must_not", {}),
        should=gold_row.get("should", {}),
    )

    f1s  = slot_f1(pred, gold)
    nf1  = negation_f1(pred, gold)
    em   = exact_match(pred, gold)

    return {
        "slot_f1":      f1s,
        "negation_f1":  nf1,
        "exact_match":  em,
        "has_negation": gold_row.get("has_negation", False),
        "n_must":       gold_row.get("n_must", 0),
        "n_must_not":   gold_row.get("n_must_not", 0),
        "complexity":   gold_row.get("complexity", ""),
    }


def summarise(results: list[dict], parser_name: str) -> dict:
    """聚合所有結果，輸出摘要。"""
    n = len(results)
    if n == 0:
        return {}

    # Overall exact match
    em_rate = sum(r["exact_match"] for r in results) / n

    # Slot F1 per dimension（must + must_not 合計）
    dim_p = defaultdict(list)
    dim_r = defaultdict(list)
    dim_f = defaultdict(list)
    for r in results:
        for side in ("must", "must_not"):
            for dim, metrics in r["slot_f1"].get(side, {}).items():
                dim_p[dim].append(metrics["p"])
                dim_r[dim].append(metrics["r"])
                dim_f[dim].append(metrics["f1"])

    dim_summary = {}
    for dim in DIMS:
        fs = dim_f.get(dim, [])
        if fs:
            dim_summary[dim] = {
                "p":  round(sum(dim_p[dim]) / len(dim_p[dim]), 4),
                "r":  round(sum(dim_r[dim]) / len(dim_r[dim]), 4),
                "f1": round(sum(fs) / len(fs), 4),
                "n":  len(fs),
            }

    # Negation F1（只看有 must_not 的筆）
    neg_rows = [r for r in results if r["has_negation"]]
    neg_f1_score = (
        sum(r["negation_f1"]["f1"] for r in neg_rows) / len(neg_rows)
        if neg_rows else 0.0
    )

    # Exact match by complexity
    em_by_complexity = defaultdict(list)
    for r in results:
        em_by_complexity[r["complexity"]].append(r["exact_match"])
    em_complexity = {
        k: round(sum(v)/len(v), 4)
        for k, v in em_by_complexity.items()
    }

    # Overall slot F1（所有 dim 平均）
    all_f1s = [v["f1"] for v in dim_summary.values()]
    overall_slot_f1 = round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0.0

    return {
        "parser":           parser_name,
        "n":                n,
        "exact_match":      round(em_rate, 4),
        "overall_slot_f1":  overall_slot_f1,
        "negation_f1":      round(neg_f1_score, 4),
        "negation_n":       len(neg_rows),
        "dim_f1":           dim_summary,
        "exact_match_by_complexity": em_complexity,
    }


def print_summary(s: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  {s['parser']}")
    print(f"{'─'*55}")
    print(f"  n             : {s['n']}")
    print(f"  Exact Match   : {s['exact_match']*100:.1f}%")
    print(f"  Overall Slot F1: {s['overall_slot_f1']*100:.1f}%")
    print(f"  Negation F1   : {s['negation_f1']*100:.1f}%  (n={s['negation_n']})")
    print(f"\n  Slot F1 by dimension:")
    for dim in DIMS:
        if dim in s["dim_f1"]:
            d = s["dim_f1"][dim]
            print(f"    {dim:15s}: F1={d['f1']*100:.1f}%  P={d['p']*100:.1f}%  R={d['r']*100:.1f}%  (n={d['n']})")
    print(f"\n  Exact Match by complexity:")
    for k, v in sorted(s["exact_match_by_complexity"].items()):
        print(f"    {k:10s}: {v*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=0)
    parser.add_argument("--parsers", default="p0,p1", help="e.g. p0,p1,p3@gpt-4.1-mini")
    parser.add_argument("--eval",    default=str(EVAL_PATH))
    parser.add_argument("--workers", type=int, default=10, help="LLM 並行數（預設 10）")
    args = parser.parse_args()

    parser_names = [p.strip() for p in args.parsers.split(",")]

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
    log.info(f"Loaded {len(rows)} eval rows, parsers={parser_names}")

    all_summaries = []

    for pname in parser_names:
        try:
            parse_fn, label = load_parser(pname)
        except Exception as e:
            log.error(f"Cannot load {pname}: {e}")
            continue

        log.info(f"Running {label} ...")
        results_map: dict[int, dict] = {}
        t0 = time.time()
        base_name = pname.split("@")[0]
        is_llm = base_name not in ("p0", "p1")

        def _run_one(idx_row):
            idx, row = idx_row
            query = row["queries"]["zh"]
            try:
                pred = parse_fn(query)
            except Exception as e:
                log.warning(f"Parse error [{pname}] qid={row['qid']}: {e}")
                pred = ConditionResult()
            return idx, eval_one(pred, row)

        if is_llm and args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_run_one, (i, row)): i
                           for i, row in enumerate(rows)}
                done = 0
                for fut in as_completed(futures):
                    idx, result = fut.result()
                    results_map[idx] = result
                    done += 1
                    if done % 200 == 0:
                        log.info(f"  {pname}: {done}/{len(rows)}")
        else:
            for i, row in enumerate(rows):
                _, result = _run_one((i, row))
                results_map[i] = result
                if (i + 1) % 200 == 0:
                    log.info(f"  {pname}: {i+1}/{len(rows)}")

        results = [results_map[i] for i in range(len(rows))]

        elapsed = time.time() - t0
        log.info(f"  Done in {elapsed:.1f}s ({elapsed/len(rows)*1000:.1f}ms/query)")

        summary = summarise(results, label)
        summary["elapsed_s"] = round(elapsed, 1)
        summary["ms_per_query"] = round(elapsed / len(rows) * 1000, 1)
        all_summaries.append(summary)
        print_summary(summary)

    # 比較表
    if len(all_summaries) > 1:
        print(f"\n{'='*55}")
        print("比較表")
        print(f"{'='*55}")
        header = f"{'Parser':30s} {'ExactMatch':>12} {'SlotF1':>8} {'NegF1':>8}"
        print(header)
        print("-" * 55)
        for s in all_summaries:
            print(f"{s['parser']:30s} {s['exact_match']*100:>11.1f}% "
                  f"{s['overall_slot_f1']*100:>7.1f}% "
                  f"{s['negation_f1']*100:>7.1f}%")

    # 儲存（append 模式：合併舊結果，同 parser name 的就覆蓋）
    out = ROOT / "results/tables/parser_eval_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except Exception:
            existing = []
    # 用 parser name 當 key merge
    merged = {s["parser"]: s for s in existing}
    for s in all_summaries:
        merged[s["parser"]] = s
    with open(out, "w") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)
    log.info(f"Saved to {out} ({len(merged)} parsers total)")


if __name__ == "__main__":
    main()
