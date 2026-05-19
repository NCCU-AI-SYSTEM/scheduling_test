"""Estimate BM25 gold-rank distribution on clean constraint subset.

This tells us how many gold-misses are recoverable by boosting BM25 weight.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.loader import load_courses
from src.doc_builders import build_d_v2
from src.retrievers import BM25Retriever, DenseRetriever

DB_PATH    = ROOT / "data" / "1142.db"
CLEAN_PATH = ROOT / "data" / "raw" / "eval_constraint_clean.jsonl"


def main():
    print("Loading courses...")
    courses = load_courses(str(DB_PATH))
    print(f"Courses: {len(courses)}")

    print("Building D-V2 docs...")
    docs = build_d_v2(courses)
    print(f"Docs: {len(docs)}")

    print("Building BM25 index...")
    t0 = time.time()
    bm25 = BM25Retriever.from_docs(docs, k=50)
    print(f"BM25 ready in {time.time() - t0:.1f}s")

    print("Building Dense index...")
    t0 = time.time()
    dense = DenseRetriever.from_docs(docs, model_name="BAAI/bge-m3", k=50)
    print(f"Dense ready in {time.time() - t0:.1f}s")

    queries = []
    with open(CLEAN_PATH) as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"Clean queries: {len(queries)}")

    bm25_rank = Counter()
    dense_rank = Counter()
    bm25_only_top10 = 0     # in BM25 top-10 但不在 Dense top-10
    dense_only_top10 = 0
    both_top10 = 0
    both_top50 = 0

    for r in queries:
        q = r["query"]
        golds = set(r["gold"])

        bm25_results = bm25.search(q, k=50)
        dense_results = dense.search(q, k=50)

        # find gold rank in BM25
        bm25_r = None
        for rank, (doc, _) in enumerate(bm25_results):
            if doc.course_id in golds:
                bm25_r = rank
                break
        bm25_rank[bm25_r if bm25_r is not None else "miss"] += 1

        # find gold rank in Dense
        dense_r = None
        for rank, (doc, _) in enumerate(dense_results):
            if doc.course_id in golds:
                dense_r = rank
                break
        dense_rank[dense_r if dense_r is not None else "miss"] += 1

        in_bm10  = bm25_r is not None and bm25_r < 10
        in_de10  = dense_r is not None and dense_r < 10
        in_bm50  = bm25_r is not None
        in_de50  = dense_r is not None

        if in_bm10 and in_de10: both_top10 += 1
        elif in_bm10: bm25_only_top10 += 1
        elif in_de10: dense_only_top10 += 1

        if in_bm50 and in_de50: both_top50 += 1

    n = len(queries)

    def report(name, rank_counter):
        hit10 = sum(v for k, v in rank_counter.items() if k != "miss" and k < 10)
        hit20 = sum(v for k, v in rank_counter.items() if k != "miss" and k < 20)
        hit50 = sum(v for k, v in rank_counter.items() if k != "miss" and k < 50)
        miss  = rank_counter["miss"]
        print(f"  {name:8s}: R@10={hit10/n:.3f} R@20={hit20/n:.3f} R@50={hit50/n:.3f} miss={miss/n:.3f}")

    print(f"\n=== Per-retriever gold rank distribution (n={n}) ===")
    report("BM25",  bm25_rank)
    report("Dense", dense_rank)

    print(f"\n=== Gold@10 set overlap ===")
    print(f"  Both in top10:        {both_top10}/{n} = {both_top10/n:.3f}")
    print(f"  BM25 only in top10:   {bm25_only_top10}/{n} = {bm25_only_top10/n:.3f}")
    print(f"  Dense only in top10:  {dense_only_top10}/{n} = {dense_only_top10/n:.3f}")
    print(f"  Neither in top10:     {n - both_top10 - bm25_only_top10 - dense_only_top10}/{n}")
    print(f"\n=== Gold@50 union (upper-bound for fusion+rerank) ===")
    print(f"  At least one has gold in top50: {sum(1 for q in queries if True)}")  # placeholder
    union50 = sum(
        1 for k_b, k_d in zip(bm25_rank, dense_rank) if False  # noqa
    )
    # 簡單算 union50：bm25_top50 + dense_only_top50
    bm25_top50 = sum(v for k, v in bm25_rank.items() if k != "miss")
    dense_top50 = sum(v for k, v in dense_rank.items() if k != "miss")
    print(f"  BM25 top50:  {bm25_top50}/{n} = {bm25_top50/n:.3f}")
    print(f"  Dense top50: {dense_top50}/{n} = {dense_top50/n:.3f}")

    # save raw rank info
    out = {
        "n": n,
        "bm25_rank": {str(k): v for k, v in bm25_rank.items()},
        "dense_rank": {str(k): v for k, v in dense_rank.items()},
        "both_top10": both_top10,
        "bm25_only_top10": bm25_only_top10,
        "dense_only_top10": dense_only_top10,
    }
    out_path = ROOT / "results" / "tables" / "constraint_per_retriever_rank.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
