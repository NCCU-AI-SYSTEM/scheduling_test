"""Print a random sample of extracted aspect scores next to the raw reviews, for manual audit.

  python -m scripts.v4.analysis.audit_aspects data/v4/aspects/gemma4_12b.jsonl --n 15 --reviews 20
"""

from __future__ import annotations

import argparse
import json
import random

from src.v4.catalog import load_catalog

ASPECTS = ["workload", "grading", "difficulty", "attendance", "group_work", "teaching", "usefulness"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--reviews", type=int, default=20)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()
    rows = [json.loads(x) for x in open(a.path, encoding="utf-8")]
    cat = {c.course_id: c for c in load_catalog()}
    for i, r in enumerate(random.Random(a.seed).sample(rows, a.n)):
        c = cat.get(r["course_ids"][0])
        if c is None:
            continue
        scores = " ".join(f"{k}={r[k]}" for k in ASPECTS)
        print(f"### [{i}] {r['family']}  n_reviews={r['n_reviews']}\n{scores}\nsummary: {r['summary']}")
        for t in c.reviews[-a.reviews:]:
            print("  -", t[:120])
        print()


if __name__ == "__main__":
    main()
