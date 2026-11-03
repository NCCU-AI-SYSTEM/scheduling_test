"""Rescore saved parse outputs with the corrected course-category semantics.

The benchmark started before the kind-code fix (3=群修, 4=通識, PE by name). Parsed
conditions are stored per row, so every model is rescored here with one rule set.
eval_conditions_v1 rows whose oracle uses `kind` were generated with the wrong code
table (a query saying 體育 has a 通識 gold course), so they are dropped.

  python -m scripts.v4.analysis.rescore_llm_bench
"""

from __future__ import annotations

import json
from pathlib import Path

from src.doc_builders import _course_metadata
from src.eval.conditions import satisfies
from src.loader import load_courses
from src.parsers.base import ConditionResult

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "v4" / "llm_bench"


def main() -> None:
    meta = {c.course_id: _course_metadata(c) for c in load_courses()}
    bench = {r["qid"]: r for r in (json.loads(x) for x in (ROOT / "data/v4/llm_bench/parse_set.jsonl").open(encoding="utf-8"))}
    bad = {q for q, r in bench.items() if r["src"] == "cond" and ("kind" in (r["must"] or {}) or "kind" in (r["must_not"] or {}))}
    pools: dict[str, set[str]] = {}
    for q, r in bench.items():
        if r["src"] == "cond" and q not in bad:
            oc = ConditionResult(must=r["must"], must_not=r["must_not"])
            pools[q] = {c for c, m in meta.items() if satisfies(m, oc)}
    rows_out = []
    for f in sorted(RES.glob("*.parse.jsonl")):
        rows = [json.loads(x) for x in f.open(encoding="utf-8")]
        kept, pj, exact, neg, cplx, topic_cond, secs = [], [], [], [], [], [], []
        for r in rows:
            if r["qid"] in bad:
                continue
            b = bench[r["qid"]]
            cond = ConditionResult(must=r["pred"]["must"], must_not=r["pred"]["must_not"])
            kept.append(satisfies(meta[b["gold"]], cond))
            secs.append(r["seconds"])
            if b["src"] == "cond":
                pred = {c for c, m in meta.items() if satisfies(m, cond)}
                gold = pools[r["qid"]]
                j = len(pred & gold) / len(pred | gold) if pred | gold else 1.0
                pj.append(j)
                exact.append(j >= 0.999)
                if b["stratum"].endswith("-neg"):
                    neg.append(j)
                if b["stratum"].startswith("complex"):
                    cplx.append(j)
            elif b["stratum"] in ("topic", "colloquial"):
                topic_cond.append(r["has_cond"])
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        secs.sort()
        rows_out.append({"model": f.name.removesuffix(".parse.jsonl"), "n": len(kept), "gold_kept": mean(kept),
                         "pool_j": mean(pj), "exact": mean(exact), "neg_j": mean(neg), "complex_j": mean(cplx),
                         "topic_has_cond": mean(topic_cond), "p50_s": secs[len(secs) // 2],
                         "p95_s": secs[int(len(secs) * 0.95)]})
    (RES / "parse_rescored.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2))
    print(f"excluded {len(bad)} corrupted rows")
    print(f"{'model':22s} {'n':>4s} {'kept':>6s} {'poolJ':>6s} {'exact':>6s} {'negJ':>6s} {'cplxJ':>6s} {'topicC':>6s} {'p50':>5s} {'p95':>5s}")
    for d in rows_out:
        print(f"{d['model']:22s} {d['n']:4d} {d['gold_kept']:6.3f} {d['pool_j']:6.3f} {d['exact']:6.3f} "
              f"{d['neg_j'] or 0:6.3f} {d['complex_j'] or 0:6.3f} {d['topic_has_cond']:6.3f} {d['p50_s']:5.2f} {d['p95_s']:5.2f}")


if __name__ == "__main__":
    main()
