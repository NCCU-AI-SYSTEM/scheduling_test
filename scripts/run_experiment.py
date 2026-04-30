"""Run a single retrieval experiment.

Usage:
    uv run python scripts/run_experiment.py --doc d-base --retriever bm25 --eval objective_smoke --n 100
    uv run python scripts/run_experiment.py --doc d-obj  --retriever dense
    uv run python scripts/run_experiment.py --doc d-obj  --retriever rrf --n 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from src.doc_builders import BUILDERS
from src.eval import QueryEval, aggregate, from_jsonl, from_objective
from src.loader import load_courses
from src.retrievers import BM25Retriever, DenseRetriever, rrf_fuse

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "runs"
TABLES_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="114")
    p.add_argument("--semester", default="2")
    p.add_argument("--doc", choices=list(BUILDERS), default="d-base")
    p.add_argument("--retriever", choices=["bm25", "dense", "rrf"], default="bm25")
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument(
        "--eval",
        choices=["objective_smoke", "synth_jsonl"],
        default="objective_smoke",
    )
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--retrieve-k", type=int, default=50, help="depth before fusion")
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)

    tag = args.tag or f"{args.doc}__{args.retriever}__{args.eval}"
    print(f"[run] tag={tag}")

    courses = load_courses(year=args.year, semester=args.semester)
    print(f"[run] courses={len(courses)}")
    docs = BUILDERS[args.doc](courses)
    print(f"[run] docs={len(docs)} avg_len={sum(len(d.text) for d in docs)/len(docs):.0f}")

    bm25 = dense = None
    t0 = time.time()
    if args.retriever in ("bm25", "rrf"):
        bm25 = BM25Retriever.from_docs(docs, k=args.retrieve_k)
        print(f"[run] bm25 ready in {time.time()-t0:.1f}s")
    if args.retriever in ("dense", "rrf"):
        t1 = time.time()
        dense = DenseRetriever.from_docs(docs, model_name=args.dense_model, k=args.retrieve_k)
        print(f"[run] dense ready in {time.time()-t1:.1f}s (device={dense.model.device})")

    queries = from_objective(courses, n=args.n) if args.eval == "objective_smoke" else from_jsonl()
    print(f"[run] eval queries={len(queries)}")

    evals: list[QueryEval] = []
    dump: list[dict] = []
    t0 = time.time()
    for q in tqdm(queries, desc="retrieve"):
        if args.retriever == "bm25":
            hits = bm25.search(q.query, k=args.top_k)
        elif args.retriever == "dense":
            hits = dense.search(q.query, k=args.top_k)
        else:  # rrf
            r1 = bm25.search(q.query, k=args.retrieve_k)
            r2 = dense.search(q.query, k=args.retrieve_k)
            hits = rrf_fuse([r1, r2], top_k=args.top_k)
        ids = [d.course_id for d, _ in hits]
        evals.append(QueryEval(qid=q.qid, relevant=q.gold, retrieved=ids))
        dump.append(
            {"qid": q.qid, "query": q.query, "gold": sorted(q.gold), "retrieved": ids, "qtype": q.qtype}
        )
    elapsed = time.time() - t0

    metrics = aggregate(evals, ks=(5, 10, 20))
    metrics["doc_builder"] = args.doc
    metrics["retriever"] = args.retriever
    metrics["eval_set"] = args.eval
    metrics["latency_ms_per_query"] = round(elapsed / max(1, len(queries)) * 1000, 2)
    metrics["index_size"] = len(docs)

    runs_path = RESULTS_DIR / f"{tag}.jsonl"
    with runs_path.open("w", encoding="utf-8") as f:
        for r in dump:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics_path = TABLES_DIR / f"{tag}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[run] {runs_path}")
    print(f"[run] {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
