"""Which semester, and which raw database, the v4 build reads.

One place for what used to be 114-2 written into a dozen modules. Build steps take a
Semester explicitly; defaults come from the environment so a new term needs no edits:

  V4_SEMESTER   e.g. 115-1 (default 114-2)
  V4_SOURCE_DB  raw scrape (default data/1142.db)
  V4_STRUCT_DB  structured output read at runtime (default data/v4/course_struct.db)

At runtime the semester comes from course_struct.db's meta table, not the environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Semester:
    year: str      # ROC academic year, '114'
    term: str      # '1' fall, '2' spring, '3' summer

    @property
    def ys(self) -> str:
        """'1142': the form COURSE ids, RATE ids and RESULT.yearsem use."""
        return self.year + self.term

    @property
    def label(self) -> str:
        return f"{self.year}-{self.term}"

    @property
    def academic_year(self) -> int:
        return int(self.year)


def parse(s: str) -> Semester:
    """'114-2', '1142' or '114 2' -> Semester('114', '2')."""
    m = re.fullmatch(r"\s*(\d{2,3})\s*[-/ ]?\s*([123])\s*", s or "")
    if not m:
        raise ValueError(f"學期格式應為 114-2 或 1142，收到 {s!r}")
    return Semester(m.group(1), m.group(2))


def current() -> Semester:
    return parse(os.environ.get("V4_SEMESTER", "114-2"))


def source_db() -> Path:
    return Path(os.environ.get("V4_SOURCE_DB", ROOT / "data" / "1142.db"))


def struct_db() -> Path:
    return Path(os.environ.get("V4_STRUCT_DB", ROOT / "data" / "v4" / "course_struct.db"))
