"""Inspect why struct filter drops recall on objective_smoke."""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.query_rewriters import parse_constraints  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "results" / "runs" / "d-obj__bm25__raw__struct__objective_smoke.jsonl"

miss = []
with path.open() as f:
    for line in f:
        r = json.loads(line)
        if not set(r["gold"]) & set(r["retrieved"]):
            miss.append(r)
print(f"miss={len(miss)}/100")
for r in miss[:15]:
    c = parse_constraints(r["query"])
    d = dataclasses.asdict(c)
    nonempty = {
        k: v for k, v in d.items()
        if v and k not in ("raw", "semantic_residual")
    }
    print("Q:", r["query"][:80])
    print("  parsed:", nonempty)
