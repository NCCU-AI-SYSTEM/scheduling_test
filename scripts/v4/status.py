"""One-shot status of the v4 jobs on the experiment host (avoids fragile ssh one-liners)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def tail_marks(path: Path, n: int = 3) -> list[str]:
    if not path.exists():
        return []
    return [l.rstrip() for l in path.open(encoding="utf-8", errors="replace") if l.startswith("[")][-n:]


def main() -> None:
    procs = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout.splitlines()
    for p in procs:
        if "scripts.v4" in p and "status" not in p:
            print("RUN", p.strip()[:140])
    r = ROOT / "data/v4/rset/r_queries.jsonl"
    if r.exists():
        rows = [json.loads(x) for x in r.open(encoding="utf-8")]
        print("r_queries ref:", sum("ref" in x for x in rows), "/", len(rows))
    for log in ("llm_bench.log", "gpu_chain.log"):
        for line in tail_marks(ROOT / "logs" / log):
            print(log, line)
    c = ROOT / "data/v4/eligibility/cases.jsonl"
    print("eligibility cases:", sum(1 for _ in c.open()) if c.exists() else "building")


if __name__ == "__main__":
    main()
