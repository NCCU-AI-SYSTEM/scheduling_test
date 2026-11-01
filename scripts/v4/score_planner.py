"""Score planner outputs on the realistic R-set against the hand-annotated gold labels.

Fields scored:
  named      named course / teacher found when gold has one (substring match either way),
             and no invented name when gold has none
  cond       conditions scored by execution, on gold queries that have conditions:
             Jaccard between courses passing the planner's conditions and the gold's
  cond_fp    share of gold condition-free queries where the planner added a condition
  prefs      recall / precision of (aspect, direction) pairs
  oos        out_of_scope or unsupported detection: recall on gold-excluded queries
  latency    planner seconds p50 / p95

  python -m scripts.v4.score_planner gemma4:e4b qwen3.5:9b ...
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from src.eval.conditions import satisfies
from src.parsers.base import ConditionResult
from src.v4.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]


def _n(s) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", str(s or ""))


def _name_ok(gold, pred) -> bool:
    if not gold:
        return not pred
    return bool(pred) and (_n(gold) in _n(pred) or _n(pred) in _n(gold))


def main() -> None:
    gold = {json.loads(x)["qid"]: json.loads(x)["ref"] for x in (ROOT / "data/v4/rset/r_gold.jsonl").open(encoding="utf-8")}
    meta = {c.course_id: c.meta for c in load_catalog(with_reviews=False)}
    print(f"{'planner':20s} {'named':>6s} {'condJ':>6s} {'condFP':>6s} {'prefR':>6s} {'prefP':>6s} {'oosR':>6s} {'p50':>5s} {'p95':>5s}")
    for model in sys.argv[1:]:
        path = ROOT / "data/v4/plans" / f"{re.sub(r'[^A-Za-z0-9.]+', '_', model)}.jsonl"
        plans = {json.loads(x)["qid"]: json.loads(x) for x in path.open(encoding="utf-8")}
        named, condj, condfp, pr_hit, pr_n, pp_hit, pp_n, oos_hit, oos_n, secs = [], [], [], 0, 0, 0, 0, 0, 0, []
        for qid, g in gold.items():
            p = plans.get(qid)
            if p is None:
                continue
            secs.append(p["seconds"])
            excluded = bool({"out_of_scope", "unsupported"} & set(g["intents"]))
            if excluded:
                oos_n += 1
                oos_hit += bool({"out_of_scope", "unsupported"} & set(p.get("intents") or []))
                continue
            named.append(_name_ok(g["named_course"], p["named_course"]) and _name_ok(g["named_teacher"], p["named_teacher"]))
            gc = ConditionResult(must=g["must"], must_not=g["must_not"])
            pc = ConditionResult(must=p["must"], must_not=p["must_not"])
            if gc.must or gc.must_not:
                gp = {c for c, m in meta.items() if satisfies(m, gc)}
                pp = {c for c, m in meta.items() if satisfies(m, pc)}
                condj.append(len(gp & pp) / len(gp | pp) if gp | pp else 1.0)
            else:
                condfp.append(bool(pc.must or pc.must_not))
            gs = {(x["aspect"], x["want"]) for x in g["prefs"]}
            ps = {(x["aspect"], x["want"]) for x in p.get("prefs") or []}
            pr_hit += len(gs & ps)
            pr_n += len(gs)
            pp_hit += len(gs & ps)
            pp_n += len(ps)
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
        secs.sort()
        print(f"{model:20s} {mean(named):6.3f} {mean(condj):6.3f} {mean(condfp):6.3f} "
              f"{pr_hit / max(1, pr_n):6.3f} {pp_hit / max(1, pp_n):6.3f} {oos_hit / max(1, oos_n):6.3f} "
              f"{secs[len(secs) // 2]:5.2f} {secs[int(len(secs) * 0.95)]:5.2f}")


if __name__ == "__main__":
    main()
