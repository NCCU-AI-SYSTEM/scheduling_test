"""Print courses whose 不要<item> verdict flips between the keyword scan and eval_item.

  python -m scripts.v4.analysis.sample_assess_flips 作業 violate ok 6
"""

from __future__ import annotations

import random
import sys

from src.v4.assessment import profile_from, verdict
from src.v4.catalog import load_catalog
from src.v4.struct_store import load_struct


def main() -> None:
    item, a, z, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    old = {c.course_id: c for c in load_catalog(with_reviews=False)}
    b = load_struct()
    hits = []
    for cid, c in old.items():
        p_old = profile_from([lbl for lbl, _ in c.grading.get("items", [])], c.course.evaluation)
        if verdict(p_old, (), (item,)) == a and verdict(b.assess[cid], (), (item,)) == z:
            hits.append(c)
    print(f"{len(hits)} courses {a}->{z} for 不要{item}")
    for c in random.Random(7).sample(hits, min(n, len(hits))):
        print("\n#", c.course.name, "|", sorted(b.assess[c.course_id].items))
        print("  ", (c.course.evaluation or "")[:260].replace("\n", " "))


if __name__ == "__main__":
    main()
