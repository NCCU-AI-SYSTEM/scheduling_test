"""CPU-only end-to-end cost: planner on CPU threads plus search on CPU, on R-set queries.

For each planner and thread count: planner latency p50/p95 (Ollama num_gpu=0), whether
the CPU plan equals the saved GPU plan field by field, and search latency (e5 on CPU,
no reranker). Run with nothing else using the CPU.

  python -m scripts.v4.analysis.cpu_e2e_bench --planners gemma4:e4b,gemma4:12b --threads 12,4 --n 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.v4.run_e2e import load_plans, load_r
from src.v4.deploy import build, preset
from src.v4.planner import plan

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "v4" / "cpu_e2e_bench.json"


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))]


def _same(a, b) -> bool:
    return (a.need, a.named_course, a.named_teacher, a.cond.must, a.cond.must_not) == \
           (b.need, b.named_course, b.named_teacher, b.cond.must, b.cond.must_not)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--planners", default="gemma4:e4b,gemma4:12b")
    p.add_argument("--threads", default="12,4")
    p.add_argument("--n", type=int, default=30)
    a = p.parse_args()
    rows = [r for r in load_r() if not {"out_of_scope", "unsupported"} & set(r["ref"]["intents"])][:a.n]
    out = []
    for t in (int(x) for x in a.threads.split(",")):
        search, b, cfg = build(preset("cpu", threads=t))
        for model in a.planners.split(","):
            gpu = load_plans(model)
            plan(rows[0]["query"], model, {"num_gpu": 0, "num_thread": t})   # load the model once
            plan_s, search_s, same = [], [], 0
            for r in rows:
                pl = plan(r["query"], model, {"num_gpu": 0, "num_thread": t})
                plan_s.append(pl.seconds)
                same += _same(pl, gpu[r["qid"]]) if r["qid"] in gpu else 0
                t0 = time.time()
                search.search(pl, r["query"], 10, cfg)
                search_s.append(time.time() - t0)
            res = {"planner": model, "threads": t, "n": len(rows),
                   "plan_s_p50": round(_pct(plan_s, 0.5), 2), "plan_s_p95": round(_pct(plan_s, 0.95), 2),
                   "search_ms_p50": round(_pct(search_s, 0.5) * 1000), "search_ms_p95": round(_pct(search_s, 0.95) * 1000),
                   "same_plan_as_gpu": round(same / len(rows), 3)}
            out.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("CPU E2E DONE")


if __name__ == "__main__":
    main()
