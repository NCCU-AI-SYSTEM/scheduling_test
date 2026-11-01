"""Hand labels for the eligibility cases (60 courses x 3 profiles), 2026/10/18.

Why hand labels: gpt-oss:120b answered "no" for 176/180 cases, treating the course's
target class (開課班級, e.g. 國貿三四) as an exclusive restriction. At NCCU that field
says who the course is mainly for; only the note restricts other students.

Rules applied
  - target class alone is not a restriction
  - 僅限 / 限 X 系 / 限本系 X 生, student not in X              -> no
  - 限 N 年級以上, student below                                 -> no
  - 先修 / 擋修 / 須修過 X, X not in the student's taken courses  -> no
  - student already passed the same course                       -> no
  - graduate course with no explicit rule, 學程專班, add-permit-only limits,
    near-equivalent prerequisite names (程式設計一 vs 計算機程式設計) -> unsure
  - otherwise                                                    -> yes

  python -m scripts.v4.eligibility_gold
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "data" / "v4" / "eligibility" / "cases.jsonl"
N, Y, U = "no", "yes", "unsure"

LABELS: dict[int, str] = {}


def s(idx, ans: str) -> None:
    for i in idx:
        LABELS[i] = ans


s(range(0, 23), N); s([23], Y)
s([24, 26], U); s([25], N)
s([27, 29], N); s([28], U)
s(range(30, 52), N); s([52], Y); s([53], N)
s(range(54, 57), N); s([57, 58], Y); s([59], N)
s(range(60, 80), N); s([80], Y)
s(range(81, 96), N); s([96, 97, 98], Y); s([99, 101], U); s([100], Y)
s(range(102, 122), N); s([122], U)
s(range(123, 126), N); s([126], Y); s([127, 128], N)
s(range(129, 132), N); s([132, 133, 134], Y)
s(range(135, 147), N); s([147], Y); s([148, 149], N)
s(range(150, 153), N); s([153, 155], U); s([154], N)
s(range(156, 161), N); s([161], Y); s(range(162, 165), N)
s([165, 166], Y); s([167], N); s(range(168, 174), N); s([174, 175, 176], Y); s(range(177, 180), N)


def main() -> None:
    assert sorted(LABELS) == list(range(180)), len(LABELS)
    rows = [json.loads(x) for x in PATH.open(encoding="utf-8")]
    for i, r in enumerate(rows):
        if "gptoss" not in r:
            r["gptoss"] = r["ref"]
        r["ref"] = {"answer": LABELS[i], "reason": "hand label"}
    with PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(Counter(LABELS.values()))
    agree = sum(r["gptoss"]["answer"] == r["ref"]["answer"] for r in rows) / len(rows)
    print(f"gpt-oss agreement with hand labels: {agree:.3f}")


if __name__ == "__main__":
    main()
