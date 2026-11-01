"""Assessment conditions on the real catalogue and through the planner.

  python -m scripts.v4.assessment_check --planner gemma4:12b
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.v4.assessment import CONDITION_ITEMS, profile_from, verdict
from src.v4.catalog import load_catalog
from src.v4.planner import plan

QUERIES = ["不要期中考的通識", "有小考的統計課", "沒有期末考，用報告取代的課", "週三下午不用考試的課",
           "想修不要上台報告的管理課", "要有期中期末考的經濟學", "不要分組的英文授課課程", "作業少一點的程式課"]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--planner", default="gemma4:12b")
    args = a.parse_args()
    cat = load_catalog(with_reviews=False)
    profs = [profile_from([lbl for lbl, _ in c.grading.get("items", [])], c.course.evaluation) for c in cat]
    print(f"courses {len(cat)}; syllabus states no item: {sum(not p.stated for p in profs)}")
    for item in CONDITION_ITEMS:
        no = Counter(verdict(p, (), (item,)) for p in profs)
        print(f"  不要{item}: ok {no['ok']}  unknown {no['unknown']}  violate {no['violate']}")
    for q in QUERIES:
        p = plan(q, args.planner)
        print(f"{q} -> must {p.assess_must} not {p.assess_not} | cond {p.cond.must} / {p.cond.must_not} "
              f"| prefs {p.prefs}")


if __name__ == "__main__":
    main()
