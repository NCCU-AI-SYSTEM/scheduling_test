"""Build the query-parsing benchmark set for comparing small LLMs.

Two sources, both with a known gold course so parses can be scored by execution:
  cond   eval_conditions_v1: conditions were generated FROM the gold course, so the
         oracle conditions are exact. Stratified by complexity x negation.
  synth  eval_synth_clean topic / colloquial / constraint queries: no oracle
         conditions, but the gold course must survive whatever the parser extracts.

  python -m scripts.v4.build_llm_bench --per-stratum 20 --per-type 40
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "v4" / "llm_bench"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-stratum", type=int, default=20)
    p.add_argument("--per-type", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261018)
    a = p.parse_args()
    rng = random.Random(a.seed)

    rows = []
    cond = [json.loads(x) for x in (ROOT / "data/raw/eval_conditions_v1.jsonl").open(encoding="utf-8")]
    strata: dict[tuple, list] = {}
    for r in cond:
        strata.setdefault((r["complexity"], r["has_negation"]), []).append(r)
    for (cx, neg), items in sorted(strata.items()):
        for r in rng.sample(items, min(a.per_stratum, len(items))):
            rows.append({"qid": r["qid"], "src": "cond", "stratum": f"{cx}{'-neg' if neg else ''}",
                         "query": r["queries"]["zh"], "gold": r["gold_course_id"],
                         "must": r["must"], "must_not": r["must_not"]})
    synth = [json.loads(x) for x in (ROOT / "data/raw/eval_synth_clean.jsonl").open(encoding="utf-8")]
    by_type: dict[str, list] = {}
    for r in synth:
        by_type.setdefault(r["type"], []).append(r)
    for t, items in sorted(by_type.items()):
        for r in rng.sample(items, a.per_type):
            rows.append({"qid": r["qid"], "src": "synth", "stratum": t, "query": r["query"],
                         "gold": r["gold_course_id"], "must": None, "must_not": None})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "parse_set.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
