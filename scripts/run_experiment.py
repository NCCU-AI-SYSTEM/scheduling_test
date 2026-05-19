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
from src.eval.datasets import EVAL_V2
from src.filters import filter_hits
from src.loader import load_courses
from src.query_rewriters import hyde, multi_query, parse_constraints, q2d, step_back
from src.retrievers import BM25Retriever, rrf_fuse
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "runs"
TABLES_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


WEEKDAY_ZH = ["", "一", "二", "三", "四", "五", "六", "日"]


def _expand_for_bm25(query: str) -> str:
    """Append constraint-aware exact-match terms to a query for BM25 retrieval.

    Aligns with D-V2 doc text format (星期X / HH:00 / lang / unit系).
    Single weekday only; ranges (week 1..5) skipped to avoid dilution.
    """
    c = parse_constraints(query)
    parts: list[str] = []
    if len(c.weekday_include) == 1:
        wd = next(iter(c.weekday_include))
        parts.append(f"星期{WEEKDAY_ZH[wd]}")
    if c.hour_min is not None and c.hour_max is not None:
        mid = (int(c.hour_min) + int(c.hour_max)) // 2
        seen: set[int] = set()
        for h in (int(c.hour_min), mid, int(c.hour_max)):
            if h not in seen:
                parts.append(f"{h:02d}:00")
                seen.add(h)
    elif c.hour_min is not None:
        parts.append(f"{int(c.hour_min):02d}:00")
    for lang in sorted(c.lang_include):
        parts.append(lang)
    for unit in sorted(c.unit_include):
        parts.append(f"{unit}系")
    if not parts:
        return query
    return f"{query} {' '.join(parts)}"


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
    p.add_argument(
        "--bm25-expand-constraint",
        action="store_true",
        help="Append constraint-aware exact-match terms to BM25 query (Dense unchanged).",
    )
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument(
        "--eval", choices=["objective_smoke", "synth_jsonl", "eval_v2"], default="objective_smoke"
    )
    p.add_argument("--n", type=int, default=200, help="query cap; 0 = all")
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

    queries = from_objective(courses, n=args.n) if args.eval == "objective_smoke" else \
              from_jsonl(EVAL_V2) if args.eval == "eval_v2" else from_jsonl()
    if args.n and args.eval in ("synth_jsonl", "eval_v2"):  # 0 = all
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
            # constraint-aware expansion: only the BM25 leg sees the boosted query
            rq_bm25 = _expand_for_bm25(rq) if args.bm25_expand_constraint else rq
            if args.retriever == "bm25":
                runs.append(bm25.search(rq_bm25, k=args.retrieve_k))
            elif args.retriever == "dense":
                runs.append(dense.search(rq, k=args.retrieve_k))
            else:
                runs.append(bm25.search(rq_bm25, k=args.retrieve_k))
                runs.append(dense.search(rq, k=args.retrieve_k))
        hits = runs[0] if len(runs) == 1 else rrf_fuse(runs, top_k=args.retrieve_k)
        all_hits.append(hits)
        all_queries_str.append(q.query)

    # batch rerank if requested
    if args.reranker == "bge":
        from src.rerankers import batch_rerank
        print(f"[run] reranking {len(all_queries_str)} queries × {args.retrieve_k} pairs...")
        all_hits = batch_rerank(all_queries_str, all_hits, top_k=args.top_k)
        print(f"[run] rerank done in {time.time()-t0:.1f}s")

    # smart struct filter: post-rerank, only when constraints detected
    if args.filter == "struct":
        filtered_hits = []
        for q, hits in zip(queries, all_hits):
            constraints = parse_constraints(q.query)
            if constraints.has_constraints():
                hits = filter_hits(hits, constraints)
            filtered_hits.append(hits)
        all_hits = filtered_hits

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
