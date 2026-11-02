"""Do hard conditions as SQL select exactly the courses the Python check selects?

Conditions tested: every gold plan in the R set, every planner output (gemma4:e4b, 12b),
and each relaxation step of those (the pipeline drops dimensions in RELAX_ORDER).

  python -m scripts.v4.pg_condition_parity
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.eval.conditions import check as check_cond
from src.parsers.base import ConditionResult
from src.v4.pg_backend import cond_ids, connect
from src.v4.pipeline import RELAX_ORDER
from src.v4.struct_store import load_struct

ROOT = Path(__file__).resolve().parents[2]


def conditions() -> list[ConditionResult]:
    out = []
    for x in (ROOT / "data/v4/rset/r_gold.jsonl").open(encoding="utf-8"):
        ref = json.loads(x)["ref"]
        out.append(ConditionResult(must=ref.get("must") or {}, must_not=ref.get("must_not") or {}))
    for model in ("gemma4_e4b", "gemma4_12b"):
        path = ROOT / f"data/v4/plans/{model}.jsonl"
        if path.exists():
            for x in path.open(encoding="utf-8"):
                p = json.loads(x)
                out.append(ConditionResult(must=p.get("must") or {}, must_not=p.get("must_not") or {}))
    steps = []
    for c in out:   # relaxation steps, as V4Search._cond_mask produces them
        must, must_not = dict(c.must), dict(c.must_not)
        for dim in RELAX_ORDER:
            if dim in must or dim in must_not:
                must.pop(dim, None)
                must_not.pop(dim, None)
                steps.append(ConditionResult(must=dict(must), must_not=dict(must_not)))
    return [c for c in out + steps if c.must or c.must_not]


def main() -> None:
    b = load_struct()
    conn = connect()
    conds = conditions()
    same, diffs, t_py, t_sql = 0, [], 0.0, 0.0
    for c in conds:
        t0 = time.time()
        py = {x.course_id for x in b.catalog if all(check_cond(x.meta, c).values())}
        t1 = time.time()
        sql = cond_ids(conn, c)
        t2 = time.time()
        t_py, t_sql = t_py + t1 - t0, t_sql + t2 - t1
        if py == sql:
            same += 1
        else:
            diffs.append({"must": c.must, "must_not": c.must_not, "python_only": sorted(py - sql)[:5],
                          "sql_only": sorted(sql - py)[:5], "n_py": len(py), "n_sql": len(sql)})
    n = len(conds)
    print(f"conditions {n}: identical {same} ({same / n:.1%}); "
          f"mean python {t_py / n * 1000:.1f} ms, sql {t_sql / n * 1000:.1f} ms")
    for d in diffs[:10]:
        print("  DIFF", json.dumps(d, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
