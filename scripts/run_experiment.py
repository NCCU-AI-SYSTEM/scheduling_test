"""Run a single retrieval experiment (W6 update: + structured filter, + LLM rewriters).

Usage:
    uv run python scripts/run_experiment.py --doc d-base --retriever bm25
    uv run python scripts/run_experiment.py --doc d-obj  --retriever bm25 --filter struct
    uv run python scripts/run_experiment.py --doc d-obj  --retriever bm25 --rewrite hyde
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from src.doc_builders import BUILDERS
from src.eval import QueryEval, aggregate, from_jsonl, from_objective
from src.filters import filter_hits
from src.loader import load_courses
from src.query_rewriters import hyde, multi_query, parse_constraints, q2d, step_back
from src.retrievers import BM25Retriever, rrf_fuse
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "runs"
TABLES_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _rewrite(query: str, mode: str) -> list[str]:
    """Return list of query strings to retrieve with (RRF-fused if >1)."""
    if mode == "raw":
        return [query]
    if mode == "hyde":
        return [hyde(query)]
    if mode == "q2d":
        return [q2d(query)]
    if mode == "stepback":
        return [query, step_back(query)]
    if mode == "multi":
        return [query, *multi_query(query)]
    raise ValueError(mode)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--year", default="114")
    p.add_argument("--semester", default="2")
    p.add_argument("--doc", choices=list(BUILDERS), default="d-base")
    p.add_argument("--retriever", choices=["bm25", "dense", "rrf"], default="bm25")
    p.add_argument("--reranker", choices=["none", "bge"], default="none")
    p.add_argument("--rewrite", choices=["raw", "hyde", "q2d", "stepback", "multi"], default="raw")
    p.add_argument("--filter", choices=["none", "struct"], default="none")
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument(
        "--eval", choices=["objective_smoke", "synth_jsonl"], default="objective_smoke"
    )
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--retrieve-k", type=int, default=50)
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)

    tag = args.tag or f"{args.doc}__{args.retriever}__{args.reranker}__{args.rewrite}__{args.filter}__{args.eval}"
    print(f"[run] tag={tag}")

    courses = load_courses(year=args.year, semester=args.semester)
    docs = BUILDERS[args.doc](courses)

    bm25 = dense = None
    t0 = time.time()
    if args.retriever in ("bm25", "rrf"):
        bm25 = BM25Retriever.from_docs(docs, k=args.retrieve_k)
    if args.retriever in ("dense", "rrf"):
        from src.retrievers import DenseRetriever  # lazy import: avoid HF on bm25 runs

        dense = DenseRetriever.from_docs(docs, model_name=args.dense_model, k=args.retrieve_k)
    print(f"[run] index ready in {time.time()-t0:.1f}s")

    queries = from_objective(courses, n=args.n) if args.eval == "objective_smoke" else from_jsonl()
    if args.n and args.eval == "synth_jsonl":
        import random
        rng = random.Random(42)
        rng.shuffle(queries)
        queries = queries[: args.n]
    print(f"[run] eval queries={len(queries)}")

    evals: list[QueryEval] = []
    dump: list[dict] = []
    t0 = time.time()

    # collect all hits first (needed for batch_rerank)
    all_hits: list[list[tuple]] = []
    all_queries_str: list[str] = []
    for q in tqdm(queries, desc="retrieve"):
        rewritten = _rewrite(q.query, args.rewrite)
        runs: list[list] = []
        for rq in rewritten:
            if args.retriever == "bm25":
                runs.append(bm25.search(rq, k=args.retrieve_k))
            elif args.retriever == "dense":
                runs.append(dense.search(rq, k=args.retrieve_k))
            else:
                runs.append(bm25.search(rq, k=args.retrieve_k))
                runs.append(dense.search(rq, k=args.retrieve_k))
        hits = runs[0] if len(runs) == 1 else rrf_fuse(runs, top_k=args.retrieve_k)
        if args.filter == "struct":
            constraints = parse_constraints(q.query)
            hits = filter_hits(hits, constraints)
        all_hits.append(hits)
        all_queries_str.append(q.query)

    # batch rerank if requested
    if args.reranker == "bge":
        from src.rerankers import batch_rerank
        print(f"[run] reranking {len(all_queries_str)} queries × {args.retrieve_k} pairs...")
        all_hits = batch_rerank(all_queries_str, all_hits, top_k=args.top_k)
        print(f"[run] rerank done in {time.time()-t0:.1f}s")

    for q, hits in zip(queries, all_hits):
        ids = [d.course_id for d, _ in hits[: args.top_k]]
        evals.append(QueryEval(qid=q.qid, relevant=q.gold, retrieved=ids))
        dump.append(
            {
                "qid": q.qid,
                "query": q.query,
                "gold": sorted(q.gold),
                "retrieved": ids,
                "qtype": q.qtype,
                "rewritten": _rewrite(q.query, args.rewrite) if args.rewrite != "raw" else None,
            }
        )
    elapsed = time.time() - t0

    metrics = aggregate(evals, ks=(5, 10, 20))
    metrics["doc_builder"] = args.doc
    metrics["retriever"] = args.retriever
    metrics["reranker"] = args.reranker
    metrics["rewrite"] = args.rewrite
    metrics["filter"] = args.filter
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
