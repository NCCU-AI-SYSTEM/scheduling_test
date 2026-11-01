"""Per-course aspect scores (1-5) used for preference re-ranking.

Sources, in priority order:
  reviews    LLM-extracted from past student evaluations (scripts/v4/extract_aspects.py)
  syllabus   group_work from the teaching-approach group share and graded group items;
             attendance from the participation weight in the grading breakdown
  enrollment popularity from the 114-2 fill ratio (count / limit)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.v4.catalog import CatalogCourse


def _bucket(x: float, cuts: list[float]) -> int:
    return 1 + sum(x >= c for c in cuts)


def syllabus_aspects(c: CatalogCourse) -> dict[str, int]:
    out: dict[str, int] = {}
    if c.teaching:
        g = c.teaching.get("group", 0)
        group_items = c.grading["by_cat"].get("group", 0) if c.grading["reliable"] else 0
        out["group_work"] = _bucket(max(g, group_items), [1, 15, 25, 40])
    if c.grading["reliable"]:
        out["attendance"] = _bucket(c.grading["by_cat"].get("participation", 0), [1, 10, 20, 30])
    if c.enrollment:
        out["popularity"] = _bucket(c.enrollment["fill"], [0.3, 0.6, 0.9, 1.0])
    return out


# Manual audit of 12 families (docs/v4/LAB_NOTEBOOK.md, 2026/10/18): review-derived
# workload, difficulty, teaching and usefulness matched the comments; grading was
# unsupported in 3/3 non-null cases and attendance in 2/3, so those two are not taken
# from reviews (attendance still comes from the syllabus participation weight).
REVIEW_TRUSTED = ("workload", "difficulty", "group_work", "teaching", "usefulness")


def build_aspects(catalog: list[CatalogCourse], review_file: Path | None) -> dict[str, dict]:
    out = {c.course_id: syllabus_aspects(c) for c in catalog}
    if review_file and review_file.exists():
        for line in review_file.open(encoding="utf-8"):
            r = json.loads(line)
            for cid in r["course_ids"]:
                if cid in out:
                    for k in REVIEW_TRUSTED:
                        if r.get(k) is not None:
                            out[cid][k] = r[k]
    return out
