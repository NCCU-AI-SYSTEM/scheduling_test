"""Convergent validity of review-derived aspects against independent syllabus signals.

No human labels exist for 涼/甜, so check whether the LLM's reading of past reviews
agrees with what the syllabus itself says, where the two measure related things:
  group_work (reviews)  vs  group-activity share in the teaching approach (syllabus)
  attendance (reviews)  vs  participation weight in the grading breakdown (syllabus)
  workload (reviews)    vs  number of graded items + report/project/homework weight
Also reports coverage (share of courses with a non-null score per aspect).

  python -m scripts.v4.analysis.validate_aspects data/v4/aspects/gemma4_12b.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.v4.catalog import load_catalog


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def main() -> None:
    path = Path(sys.argv[1])
    asp = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        for cid in r["course_ids"]:
            asp[cid] = r
    cat = {c.course_id: c for c in load_catalog()}
    n = len(asp)
    print(f"families with reviews scored: {sum(1 for _ in path.open())}, courses covered: {n}")
    for a in ("workload", "grading", "difficulty", "attendance", "group_work", "teaching", "usefulness"):
        print(f"  coverage {a:11s} {sum(v.get(a) is not None for v in asp.values()) / max(1, n):.3f}")
    pairs = {
        "group_work vs teaching.group%": ("group_work", lambda c: c.teaching.get("group") if c.teaching else None),
        "attendance vs participation%": ("attendance", lambda c: c.grading["by_cat"].get("participation", 0)
                                         if c.grading["reliable"] else None),
        "workload vs graded work%": ("workload", lambda c: sum(c.grading["by_cat"].get(k, 0) for k in
                                                              ("report", "project", "homework", "presentation"))
                                     if c.grading["reliable"] else None),
    }
    for label, (a, fn) in pairs.items():
        xs, ys = [], []
        for cid, v in asp.items():
            if cid in cat and v.get(a) is not None:
                s = fn(cat[cid])
                if s is not None:
                    xs.append(v[a])
                    ys.append(s)
        print(f"  {label:32s} n={len(xs):4d} spearman={spearman(xs, ys):.3f}")


if __name__ == "__main__":
    main()
