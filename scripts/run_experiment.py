"""Run a single retrieval experiment.

Usage:
    uv run python scripts/run_experiment.py --doc d-base --eval objective_smoke --n 100
    uv run python scripts/run_experiment.py --doc d-obj  --eval synth_jsonl
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
from src.retrievers import BM25Retriever

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
    p.add_argument(
        "--eval",
        choices=["objective_smoke", "synth_jsonl"],
        default="objective_smoke",
    )
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tag", default=None, help="run tag (default: doc+eval)")
    args = p.parse_args(argv)

    tag = args.tag or f"{args.doc}__{args.eval}"
    print(f"[run] tag={tag}")

    print(f"[run] loading courses {args.year}-{args.semester}")
    courses = load_courses(year=args.year, semester=args.semester)
    print(f"[run] courses={len(courses)}")

    print(f"[run] building docs via {args.doc}")
    docs = BUILDERS[args.doc](courses)
    print(f"[run] docs={len(docs)}  (avg text len={sum(len(d.text) for d in docs)/len(docs):.0f})")

    print("[run] building BM25 + jieba index")
    t0 = time.time()
    retriever = BM25Retriever.from_docs(docs, k=args.top_k)
    print(f"[run] index built in {time.time()-t0:.1f}s")

    if args.eval == "objective_smoke":
        queries = from_objective(courses, n=args.n)
    else:
        queries = from_jsonl()
    print(f"[run] eval queries={len(queries)}")

    evals: list[QueryEval] = []
    retrieved_dump: list[dict] = []
    t0 = time.time()
    for q in tqdm(queries, desc="retrieve"):
        hits = retriever.search(q.query, k=args.top_k)
        ids = [doc.course_id for doc, _ in hits]
        evals.append(QueryEval(qid=q.qid, relevant=q.gold, retrieved=ids))
        retrieved_dump.append(
            {
                "qid": q.qid,
                "query": q.query,
                "gold": sorted(q.gold),
                "retrieved": ids,
                "qtype": q.qtype,
            }
        )
    elapsed = time.time() - t0

    metrics = aggregate(evals, ks=(5, 10, 20))
    metrics["doc_builder"] = args.doc
    metrics["eval_set"] = args.eval
    metrics["latency_ms_per_query"] = round(elapsed / max(1, len(queries)) * 1000, 2)
    metrics["index_size"] = len(docs)

    runs_path = RESULTS_DIR / f"{tag}.jsonl"
    with runs_path.open("w", encoding="utf-8") as f:
        for r in retrieved_dump:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics_path = TABLES_DIR / f"{tag}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[run] wrote {runs_path}")
    print(f"[run] wrote {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
