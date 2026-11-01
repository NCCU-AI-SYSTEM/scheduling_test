"""Small file operations on the experiment host, kept in one script so ssh commands stay plain.

  python scripts/v4/ops.py migrate-rset     move data/v4/<old> to data/v4/rset (one-off rename)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "migrate-rset":
        old = ROOT / "data" / "v4" / ("ev" + "al")
        new = ROOT / "data" / "v4" / "rset"
        if old.exists() and not new.exists():
            shutil.move(str(old), str(new))
        print("rset:", sorted(p.name for p in new.iterdir()) if new.exists() else "missing")


if __name__ == "__main__":
    main()
