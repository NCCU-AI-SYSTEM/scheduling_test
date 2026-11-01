"""Build data/v4/course_struct.db from data/1142.db (114-2).

  python -m scripts.v4.build_structured              # rules + cached LLM answers only
  python -m scripts.v4.build_structured --llm cloud  # also ask gemma4:31b-cloud for cache misses
  python -m scripts.v4.build_structured --llm local  # ask local gemma4:12b instead

The source database is opened read-only. See docs/v4/07_structured_data.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.v4.structured.build import OUT_DB, build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llm", choices=["off", "cloud", "local"], default="off")
    ap.add_argument("--out", type=Path, default=OUT_DB)
    args = ap.parse_args()
    build(args.llm, args.out, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
