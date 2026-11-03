"""Does the planner give the same plans over an OpenAI-compatible endpoint as over native Ollama?

The backend reaches models only through OpenAI-compatible APIs. Runs the R set both ways with the
same model, and scores each against the hand-made plans.

  V4_OPENAI_EXTRA='{"reasoning_effort": "none"}' python -m scripts.v4.analysis.planner_endpoint_parity \
      --base-url http://localhost:11434/v1 --model gemma4:12b --model gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.v4.run_e2e import load_r
from src.v4.planner import plan

ROOT = Path(__file__).resolve().parents[3]


def _key(p) -> str:
    return json.dumps({"need": p.need, "course": p.named_course, "teacher": p.named_teacher, "must": p.cond.must,
                       "not": p.cond.must_not, "prefs": sorted(map(str, p.prefs)), "intents": sorted(p.intents)},
                      ensure_ascii=False, sort_keys=True, default=str)


def _cond(must: dict | None, must_not: dict | None) -> str:
    return json.dumps({"must": must or {}, "not": must_not or {}}, ensure_ascii=False, sort_keys=True)


def run(model: str, rows: list[dict], base_url: str | None) -> list:
    if base_url:
        os.environ["V4_OPENAI_BASE_URL"] = base_url
    else:
        os.environ.pop("V4_OPENAI_BASE_URL", None)
    return [plan(r["query"], model) for r in rows]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--base-url", required=True)
    a.add_argument("--model", action="append", required=True)
    args = a.parse_args()
    rows = load_r()
    out = {}
    for model in args.model:
        native = run(model, rows, None)
        openai = run(model, rows, args.base_url)
        same = sum(_key(x) == _key(y) for x, y in zip(native, openai))
        same_cond = sum(_cond(x.cond.must, x.cond.must_not) == _cond(y.cond.must, y.cond.must_not)
                        for x, y in zip(native, openai))
        gold = [_cond(r["ref"].get("must"), r["ref"].get("must_not")) for r in rows]
        n = len(rows)
        out[model] = {
            "queries": n, "identical_plans": same / n, "identical_conditions": same_cond / n,
            "failed_native": sum(not p.ok for p in native), "failed_openai": sum(not p.ok for p in openai),
            "gold_conditions_native": sum(_cond(p.cond.must, p.cond.must_not) == g for p, g in zip(native, gold)) / n,
            "gold_conditions_openai": sum(_cond(p.cond.must, p.cond.must_not) == g for p, g in zip(openai, gold)) / n,
            "p50_s_native": float(np.median([p.seconds for p in native[5:]])),
            "p50_s_openai": float(np.median([p.seconds for p in openai[5:]])),
        }
        print(model, json.dumps(out[model], ensure_ascii=False), flush=True)
    (ROOT / "results/v4/planner_endpoint_parity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
