"""Build data/v4/course_struct.db from the raw scrape for one semester.

  python -m scripts.v4.build_structured                        # 114-2 from data/1142.db, cached LLM answers only
  python -m scripts.v4.build_structured --llm cloud            # also ask gemma4:31b-cloud for cache misses
  python -m scripts.v4.build_structured --llm local            # ask local gemma4:12b instead
  python -m scripts.v4.build_structured --semester 115-1 --src data/1151.db --out data/v4/staging/x.db

The source database is opened read-only. See docs/v4/07_structured_data.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.v4.semester import current, parse, source_db
from src.v4.structured.build import OUT_DB, build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llm", choices=["off", "cloud", "local"], default="off")
    ap.add_argument("--semester", default=current().label, help="e.g. 115-1 (default V4_SEMESTER or 114-2)")
    ap.add_argument("--src", type=Path, default=source_db(), help="raw scrape (default V4_SOURCE_DB or data/1142.db)")
    ap.add_argument("--out", type=Path, default=OUT_DB)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build(args.llm, args.out, log=lambda m: print(m, flush=True), sem=parse(args.semester), src=args.src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
