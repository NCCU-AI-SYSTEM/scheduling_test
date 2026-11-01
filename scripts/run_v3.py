"""Run retrieval configs on eval v3 and report graded / constraint-aware metrics.

Configs are composable flags rather than a fixed list:

  --mode        bm25 | dense | rrf             first stage
  --expand      BM25 leg gets constraint exact-match terms (E7)
  --rerank      bge-reranker-v2-m3 over retrieve-k candidates
  --cond        none | oracle | parser | regex  where hard conditions come from
  --apply       hard | soft                     pre-filter mask vs score multiplier
  --dense-query raw | need                      embed the full query or only its topic

  E7                       --mode rrf --expand --rerank
  E7 + oracle pre-filter   --mode rrf --expand --rerank --cond oracle --apply hard
  E7 + LLM parser, soft    --mode rrf --expand --rerank --cond parser --apply soft

Example:
  python -m scripts.run_v3 --name e7 --mode rrf --expand --rerank
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.eval.conditions import check, satisfies
from src.eval.v3_metrics import Scorer, load_eval, summarize
from src.parsers.base import ConditionResult

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3"


def _conditions(args, queries) -> tuple[list[ConditionResult | None], list[str]]:
    """Per-query conditions and topical remainder, from the chosen source."""
    if args.cond == "none":
        return [None] * len(queries), [q.query for q in queries]
    if args.cond == "oracle":
        return [q.cond if q.has_cond else None for q in queries], [q.need or q.query for q in queries]
    if args.cond == "regex":
        from src.parsers.regex_v1 import parse

        conds = [parse(q.query) for q in queries]
        return [c if c.has_any_condition() else None for c in conds], [q.query for q in queries]
    from src.parsers.ollama_parser import OllamaParser

    parser = OllamaParser(args.parser_model)
    raws = parser.parse_many([q.query for q in queries])
    conds = [OllamaParser.to_condition(r) for r in raws]
    needs = [(r.get("need") or "").strip() or q.query for r, q in zip(raws, queries)]
    return [c if c.has_any_condition() else None for c in conds], needs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--system", choices=["searcher", "prod"], default="searcher")
    p.add_argument("--doc", default="d-v2")
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument("--mode", choices=["bm25", "dense", "rrf"], default="rrf")
    p.add_argument("--expand", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--cond", choices=["none", "oracle", "parser", "regex"], default="none")
    p.add_argument("--apply", choices=["hard", "soft"], default="hard")
    p.add_argument("--dense-query", choices=["raw", "need"], default="raw")
    p.add_argument("--parser-model", default="gemma4:e4b")
    p.add_argument("--retrieve-k", type=int, default=50)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--sets", default="T,M,C")
    a = p.parse_args()

    from scripts.run_experiment import _expand_for_bm25
    from src.pipeline_v3 import Searcher

    queries, qrels = load_eval()
    queries = [q for q in queries if q.set in a.sets.split(",")]
    conds, needs = _conditions(a, queries)  # parse before the GPU models load

    t0 = time.time()
    if a.system == "prod":
        from src.prod_replica import ProdReplica

        s = ProdReplica.build()
    else:
        s = Searcher.build(a.doc, dense_model=None if a.mode == "bm25" else a.dense_model)
    meta = {cid: s.doc(cid).metadata for cid in s.ids}
    names = {c.course_id: c.name for c in s.courses}
    t_index = time.time() - t0

    raw = [q.query for q in queries]
    dense_q = needs if a.dense_query == "need" else raw
    bm25_q = [_expand_for_bm25(x) for x in raw] if a.expand else raw
    allowed = None
    if a.cond != "none" and a.apply == "hard":
        allowed = []
        for c in conds:
            ok = {cid for cid in s.ids if satisfies(meta[cid], c)} if c else None
            allowed.append(ok or None)  # empty pool -> fall back to unfiltered

    t0 = time.time()
    cands = s.candidates(dense_q, a.mode, a.retrieve_k, bm25_queries=bm25_q, allowed=allowed)
    t_first = time.time() - t0
    t0 = time.time()
    ranked = s.rerank(raw, cands, a.retrieve_k) if a.rerank else cands
    t_rerank = time.time() - t0

    if a.cond != "none" and a.apply == "soft":
        from src.scoring.soft_constraints import soft_factor

        ranked = [
            sorted(((cid, sc * soft_factor(check(meta[cid], c))) for cid, sc in r),
                   key=lambda x: -x[1]) if c else r
            for r, c in zip(ranked, conds)
        ]

    runs = {q.qid: [cid for cid, _ in r[: a.top_k]] for q, r in zip(queries, ranked)}
    scorer = Scorer(queries, qrels, meta, names)
    summary = summarize(scorer, runs)
    n = max(1, len(queries))
    summary["_config"] = vars(a) | {
        "ms_per_q_first_stage": round(t_first / n * 1000, 1),
        "ms_per_q_rerank": round(t_rerank / n * 1000, 1),
        "index_s": round(t_index, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (OUT / f"{a.name}.runs.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps({"qid": q.qid, "ranked": runs[q.qid]}, ensure_ascii=False) + "\n")
    for g in ("ALL", "T", "M", "C"):
        if g in summary:
            row = summary[g]
            print(f"{a.name:28s} {g:3s} n={row['n']:4d} nDCG={row['ndcg']:.3f} "
                  f"P={row['p_rel']:.3f} Succ={row['success']:.3f} "
                  f"P_cond={row.get('p_cond', float('nan')):.3f} sibR={row['sib_recall']:.3f} "
                  f"judged={row['judged']:.2f}")


if __name__ == "__main__":
    main()
