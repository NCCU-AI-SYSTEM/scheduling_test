"""Test constraint-aware query expansion: parse_constraints → BM25 boost string.

Strategy: keep Dense query original (semantic), but feed BM25 with
"<original> + <expansion>" where expansion contains exact-match terms
that align with D-V2 doc text:
  - weekday → "星期X" (single weekday only; ranges skipped to avoid dilution)
  - hour    → "HH:00" at min/mid/max
  - lang    → "中文" / "英文" / etc.
  - unit    → "<keyword>系"
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import unittest.mock as mock
sys.modules.setdefault("ollama", mock.MagicMock())

from src.loader import load_courses
from src.doc_builders import build_d_v2
from src.retrievers import BM25Retriever, DenseRetriever
from src.query_rewriters.structured import parse_constraints

DB_PATH    = ROOT / "data" / "1142.db"
CLEAN_PATH = ROOT / "data" / "raw" / "eval_constraint_clean.jsonl"

WEEKDAY_ZH = ["", "一", "二", "三", "四", "五", "六", "日"]


def expand_for_bm25(c) -> str:
    parts: list[str] = []
    # weekday: only single weekday; ranges (e.g. 1..5) are too noisy
    if len(c.weekday_include) == 1:
        wd = next(iter(c.weekday_include))
        parts.append(f"星期{WEEKDAY_ZH[wd]}")
    # hour: min / mid / max (zero-padded to match doc format)
    if c.hour_min is not None and c.hour_max is not None:
        mid = (int(c.hour_min) + int(c.hour_max)) // 2
        seen = set()
        for h in (int(c.hour_min), mid, int(c.hour_max)):
            if h not in seen:
                parts.append(f"{h:02d}:00")
                seen.add(h)
    elif c.hour_min is not None:
        parts.append(f"{int(c.hour_min):02d}:00")
    # language
    for lang in sorted(c.lang_include):
        parts.append(lang)
    # unit
    for unit in sorted(c.unit_include):
        parts.append(f"{unit}系")
    return " ".join(parts)


def weighted_rrf(bm25_run, dense_run, w_bm25=1.0, w_dense=1.0, k=60, top_k=50):
    fused: dict[str, float] = defaultdict(float)
    for rank, (doc, _) in enumerate(bm25_run):
        fused[doc.course_id] += w_bm25 / (k + rank + 1)
    for rank, (doc, _) in enumerate(dense_run):
        fused[doc.course_id] += w_dense / (k + rank + 1)
    return [cid for cid, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]]


def main():
    print("Loading...")
    courses = load_courses(str(DB_PATH))
    docs    = build_d_v2(courses)
    bm25    = BM25Retriever.from_docs(docs, k=50)
    print("Building dense...")
    dense   = DenseRetriever.from_docs(docs, model_name="BAAI/bge-m3", k=50)

    queries = []
    with open(CLEAN_PATH) as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Queries: {len(queries)}")

    # Pre-compute: BM25(orig), BM25(expanded), Dense(orig)
    print("Running 3-way retrieval per query...")
    runs = []
    expansion_count = 0
    t0 = time.time()
    for i, r in enumerate(queries):
        q     = r["query"]
        c     = parse_constraints(q)
        exp   = expand_for_bm25(c)
        if exp:
            expansion_count += 1
        q_exp = f"{q} {exp}".strip()

        bm_orig = bm25.search(q,     k=50)
        bm_exp  = bm25.search(q_exp, k=50) if exp else bm_orig
        de_orig = dense.search(q,    k=50)
        runs.append((set(r["gold"]), bm_orig, bm_exp, de_orig))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(queries)}  ({(i+1)/(time.time()-t0):.1f} q/s)")
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"Queries with expansion: {expansion_count}/{len(queries)}")

    # Eval: BM25 raw, BM25 expanded, RRF (orig+orig), RRF (expanded+orig)
    def eval_lists(get_list):
        h10 = h20 = h50 = 0
        for golds, bm_o, bm_e, de_o in runs:
            lst = get_list(golds, bm_o, bm_e, de_o)
            top10 = set(lst[:10])
            top20 = set(lst[:20])
            top50 = set(lst[:50])
            if golds & top10: h10 += 1
            if golds & top20: h20 += 1
            if golds & top50: h50 += 1
        n = len(runs)
        return h10 / n, h20 / n, h50 / n

    print("\n=== Constraint subset (n={}) ===".format(len(runs)))

    print("\n--- Single retriever ---")
    r10, r20, r50 = eval_lists(lambda g, bo, be, do: [d.course_id for d, _ in bo])
    print(f"  BM25 (orig query)      R@10={r10:.4f}  R@20={r20:.4f}  R@50={r50:.4f}")
    r10, r20, r50 = eval_lists(lambda g, bo, be, do: [d.course_id for d, _ in be])
    print(f"  BM25 (expanded query)  R@10={r10:.4f}  R@20={r20:.4f}  R@50={r50:.4f}")
    r10, r20, r50 = eval_lists(lambda g, bo, be, do: [d.course_id for d, _ in do])
    print(f"  Dense (orig query)     R@10={r10:.4f}  R@20={r20:.4f}  R@50={r50:.4f}")

    print("\n--- RRF fusion (1.0:1.0) ---")
    r10, r20, r50 = eval_lists(lambda g, bo, be, do: weighted_rrf(bo, do, 1.0, 1.0))
    print(f"  RRF(BM25 orig + Dense orig)      R@10={r10:.4f}  R@20={r20:.4f}  R@50={r50:.4f}")
    r10, r20, r50 = eval_lists(lambda g, bo, be, do: weighted_rrf(be, do, 1.0, 1.0))
    print(f"  RRF(BM25 expanded + Dense orig)  R@10={r10:.4f}  R@20={r20:.4f}  R@50={r50:.4f}  ★")

    print("\n--- RRF fusion (BM25 expanded boosted) ---")
    for w in [1.5, 2.0]:
        r10, r20, r50 = eval_lists(lambda g, bo, be, do, w=w: weighted_rrf(be, do, w, 1.0))
        print(f"  RRF(BM25_exp×{w} + Dense)        R@10={r10:.4f}  R@20={r20:.4f}  R@50={r50:.4f}")

    out = ROOT / "results" / "tables" / "constraint_aware_expansion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": len(runs),
        "expansion_count": expansion_count,
    }
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
