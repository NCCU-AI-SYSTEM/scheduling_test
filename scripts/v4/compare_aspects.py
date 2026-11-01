"""Do the runtime aspects (course_struct.db) agree with the aspects the scorer uses (1142.db)?

  python -m scripts.v4.compare_aspects data/v4/aspects/gemma4_12b.jsonl
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from src.v4.aspects import build_aspects
from src.v4.catalog import load_catalog
from src.v4.struct_store import load_struct


def main() -> None:
    review = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    ref = build_aspects(load_catalog(), review)
    run = load_struct().aspects
    stats: dict[str, Counter] = {}
    for cid, d in ref.items():
        r = run.get(cid, {})
        for k in set(d) | set(r):
            c = stats.setdefault(k, Counter())
            a, b = d.get(k), r.get(k)
            c["both" if a is not None and b is not None else "ref_only" if a is not None else "run_only"] += 1
            if a is not None and b is not None:
                c["equal"] += a == b
                c["absdiff"] += abs(a - b)
    for k, c in sorted(stats.items()):
        both = c["both"] or 1
        print(f"{k:12s} both {c['both']:5d} equal {c['equal'] / both:6.1%} mean|diff| {c['absdiff'] / both:.2f} "
              f"ref_only {c['ref_only']:5d} run_only {c['run_only']:5d}")


if __name__ == "__main__":
    main()
