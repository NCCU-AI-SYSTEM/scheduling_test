"""Pre-cache HyDE / Q2D / step_back for eval_synth queries.

Calls LLM one by one; results saved to disk cache so run_experiment.py
picks them up for free on subsequent runs.

Usage:
    uv run python scripts/prebuild_rewrite_cache.py --method hyde --n 500
    uv run python scripts/prebuild_rewrite_cache.py --method q2d  --n 500
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.eval import from_jsonl
from src.query_rewriters import hyde, q2d, step_back


METHODS = {"hyde": hyde, "q2d": q2d, "stepback": step_back}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), required=True)
    p.add_argument("--n", type=int, default=500)
    args = p.parse_args()

    fn = METHODS[args.method]
    queries = from_jsonl()[:args.n]
    print(f"[cache] method={args.method}  n={len(queries)}")

    ok = fail = 0
    for q in tqdm(queries, desc=args.method):
        try:
            fn(q.query)  # result saved to disk cache automatically
            ok += 1
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"[fail] {q.query[:40]}: {e}")
            fail += 1

    print(f"[cache] done ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
