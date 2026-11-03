"""Simulate RRF with weighted BM25 to find best weight ratio.

For each weight w in [0.5, 1.0, 1.5, 2.0, 3.0]:
  fused = sum(w / (k + rank_bm25 + 1) + 1 / (k + rank_dense + 1))
Compute R@20 and R@50 (top-K candidate-pool recall — what reranker sees).
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_courses
from src.doc_builders import build_d_v2
from src.retrievers import BM25Retriever, DenseRetriever

DB_PATH    = ROOT / "data" / "1142.db"
CLEAN_PATH = ROOT / "data" / "raw" / "eval_constraint_clean.jsonl"


def weighted_rrf(bm25_run, dense_run, w_bm25=1.0, w_dense=1.0, k=60, top_k=50):
    fused = defaultdict(float)
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

    # 預先 search 每個 query (Dense 部分慢，所以一次跑完)
    print("Running BM25 + Dense per query...")
    runs = []
    t0 = time.time()
    for i, r in enumerate(queries):
        q = r["query"]
        bm = bm25.search(q, k=50)
        de = dense.search(q, k=50)
        runs.append((set(r["gold"]), bm, de))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(queries)}  ({(i+1)/(time.time()-t0):.1f} q/s)")
    print(f"Done in {time.time()-t0:.1f}s")

    # 對不同 weight 做 fusion，算 R@10/20/50
    results = {}
    for w_bm25 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        hits10 = hits20 = hits50 = 0
        for golds, bm, de in runs:
            fused = weighted_rrf(bm, de, w_bm25=w_bm25, w_dense=1.0, top_k=50)
            top10 = set(fused[:10])
            top20 = set(fused[:20])
            top50 = set(fused[:50])
            if golds & top10: hits10 += 1
            if golds & top20: hits20 += 1
            if golds & top50: hits50 += 1
        n = len(runs)
        results[w_bm25] = {
            "R@10": hits10 / n,
            "R@20": hits20 / n,
            "R@50": hits50 / n,
        }
        print(f"  w_bm25={w_bm25:>4.1f}  R@10={hits10/n:.4f}  R@20={hits20/n:.4f}  R@50={hits50/n:.4f}")

    # 加碼：dense weight boost
    print("\n  --- BM25 fixed at 1.0, dense boosted ---")
    for w_dense in [1.5, 2.0, 3.0]:
        hits10 = hits20 = hits50 = 0
        for golds, bm, de in runs:
            fused = weighted_rrf(bm, de, w_bm25=1.0, w_dense=w_dense, top_k=50)
            top10 = set(fused[:10])
            top20 = set(fused[:20])
            top50 = set(fused[:50])
            if golds & top10: hits10 += 1
            if golds & top20: hits20 += 1
            if golds & top50: hits50 += 1
        n = len(runs)
        print(f"  w_dense={w_dense:>4.1f}  R@10={hits10/n:.4f}  R@20={hits20/n:.4f}  R@50={hits50/n:.4f}")

    out_path = ROOT / "results" / "tables" / "constraint_rrf_weight_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
