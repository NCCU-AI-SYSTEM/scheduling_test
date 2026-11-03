"""How many 114-2 syllabi let us state the assessment format (期中考、小考、報告 ...)?

  python -m scripts.v4.analysis.format_coverage
"""

from __future__ import annotations

import re
from collections import Counter

from src.v4.catalog import load_catalog
from src.v4.present import FORMAT_KEYS, grading_summary, teaching_summary


def main() -> None:
    cat = load_catalog(with_reviews=False)
    g = [grading_summary(c) for c in cat]
    t = [teaching_summary(c) for c in cat]
    with_pct = sum(bool(re.search(r"\d+%", x)) for x in g)
    print(f"courses {len(cat)}")
    print(f"評分方式 stated: {sum(x != '大綱未註明' for x in g) / len(cat):.1%} "
          f"(with a percentage {with_pct / len(cat):.1%})")
    print(f"授課方式 stated: {sum(x != '大綱未註明' for x in t) / len(cat):.1%}")
    keys = Counter(k for x in g for k, _ in FORMAT_KEYS if k in x)
    print("assessment items mentioned:", {k: f"{v / len(cat):.0%}" for k, v in keys.most_common()})
    for c, x in list(zip(cat, g))[300:306]:
        print(" ", c.course.name, "|", x)


if __name__ == "__main__":
    main()
