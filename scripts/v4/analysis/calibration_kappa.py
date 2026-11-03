"""Agreement between the human calibration labels and the LLM judge (qwen3:14b, UMBRELA 0-3).

Reports exact agreement, Cohen's kappa (unweighted and quadratic-weighted, since the
grades are ordinal), and binary agreement at the relevance cut used by the metrics
(grade >= 2).

  python -m scripts.v4.analysis.calibration_kappa
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAL = ROOT / "data" / "v4" / "calibration"


def kappa(a: list[int], b: list[int], k: int, weighted: bool) -> float:
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    pa, pb = Counter(a), Counter(b)
    w = [[((i - j) ** 2) / ((k - 1) ** 2) if weighted else float(i != j) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] / n for i in range(k) for j in range(k))
    den = sum(w[i][j] * pa[i] * pb[j] / n ** 2 for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


def main() -> None:
    key = {json.loads(x)["id"]: json.loads(x) for x in (CAL / "calibration_key.jsonl").open(encoding="utf-8")}
    human, judge, notes = [], [], []
    for f in sorted((CAL / "labels").glob("i*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        human.append(int(d["grade"]))
        judge.append(int(key[f.stem]["judge"]))
        if d.get("note"):
            notes.append((f.stem, d["note"]))
    n = len(human)
    exact = sum(h == j for h, j in zip(human, judge)) / n
    hb = [int(h >= 2) for h in human]
    jb = [int(j >= 2) for j in judge]
    print(f"n={n}  exact={exact:.3f}  kappa={kappa(human, judge, 4, False):.3f}  "
          f"weighted_kappa={kappa(human, judge, 4, True):.3f}")
    print(f"binary (>=2): agree={sum(x == y for x, y in zip(hb, jb)) / n:.3f}  kappa={kappa(hb, jb, 2, False):.3f}")
    print("human grade dist:", dict(sorted(Counter(human).items())), " judge:", dict(sorted(Counter(judge).items())))
    conf = Counter(zip(judge, human))
    print("confusion judge->human:", {f"{j}->{h}": c for (j, h), c in sorted(conf.items())})
    by_set = Counter()
    for f, h, j in zip(sorted((CAL / "labels").glob("i*.json")), human, judge):
        s = key[f.stem]["set"]
        by_set[(s, "n")] += 1
        by_set[(s, "agree_bin")] += int((h >= 2) == (j >= 2))
    for s in sorted({k[0] for k in by_set}):
        print(f"set {s}: n={by_set[(s, 'n')]} binary agree={by_set[(s, 'agree_bin')] / by_set[(s, 'n')]:.3f}")
    print("notes:", notes)


if __name__ == "__main__":
    main()
