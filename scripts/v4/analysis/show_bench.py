"""Print the LLM benchmark summary as two compact tables (latest run per model)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    latest: dict[tuple, dict] = {}
    for line in (ROOT / "results/v4/llm_bench/summary.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if not d.get("limit"):
            latest[(d["model"], d["task"])] = d
    print(f"{'model':20s} {'goldKept':>8s} {'poolJ':>6s} {'exact':>6s} {'negJ':>6s} {'cplxJ':>6s} "
          f"{'topicCond':>9s} {'p50s':>6s} {'p95s':>6s} {'tok/s':>6s}")
    for (m, t), d in latest.items():
        if t == "parse":
            print(f"{m:20s} {d['gold_kept_all']:8.3f} {d['pool_jaccard_cond']:6.3f} {d['exact_pool_cond']:6.3f} "
                  f"{d['neg_pool_jaccard']:6.3f} {d['complex_pool_jaccard']:6.3f} {d['topic_has_cond']:9.3f} "
                  f"{d['p50_s']:6.2f} {d['p95_s']:6.2f} {d['tok_s'] or 0:6.1f}")
    print(f"\n{'model':20s} {'pass':>6s} {'toolAcc':>7s} {'args':>6s} {'forbid':>6s} {'p50s':>6s} {'err':>4s}")
    for (m, t), d in latest.items():
        if t.startswith("tool"):
            print(f"{m[:13] + ('*' if t.endswith('strict') else ''):20s} {d['pass']:6.3f} {d['tool_acc']:7.3f} {d['arg_score']:6.3f} {d['forbid_rate']:6.3f} "
                  f"{d['p50_s'] or 0:6.2f} {d['errors']:4d}")


if __name__ == "__main__":
    main()
