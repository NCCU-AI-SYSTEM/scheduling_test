"""Judge every unjudged (query, course) in any run's top-10, then rescore all runs.

Pooling is done once up front, so a system evaluated later can surface courses
no pooled system retrieved; those would silently count as irrelevant. This
keeps the pool "living": new top-10 courses get judged with the same judge
and prompt, appended to qrels.jsonl, and every run is rescored from its saved
ranking (no retrieval re-run).

  python -m scripts.rescore_v3 [--workers 4] [--no-judge]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from scripts.build_eval_v3 import judge_doc
from src.eval.judge import judge
from src.eval.v3_metrics import EVAL_DIR, Scorer, load_eval, summarize

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "v3"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--runs-dir", default=str(RUNS))
    a = p.parse_args()
    runs_dir = Path(a.runs_dir)

    from src.doc_builders import _course_metadata
    from src.loader import load_courses

    courses = load_courses()
    by_id = {c.course_id: c for c in courses}
    queries, qrels = load_eval()
    qmap = {q.qid: q for q in queries}
    runs = {f.name.removesuffix(".runs.jsonl"): {json.loads(l)["qid"]: json.loads(l)["ranked"]
                                                 for l in f.open(encoding="utf-8")}
            for f in sorted(runs_dir.glob("*.runs.jsonl"))}

    missing = sorted({(qid, cid) for run in runs.values() for qid, ranked in run.items()
                      for cid in ranked[:10]
                      if qmap[qid].need and cid not in qrels.get(qid, {})})
    print(f"{len(runs)} runs, {len(missing)} unjudged top-10 pairs")
    if missing and not a.no_judge:
        def run(pair):
            qid, cid = pair
            return judge(qmap[qid].need, cid, judge_doc(by_id[cid]))

        with ThreadPoolExecutor(a.workers) as ex:
            grades = list(tqdm(ex.map(run, missing), total=len(missing)))
        with (EVAL_DIR / "qrels.jsonl").open("a", encoding="utf-8") as f:
            for (qid, cid), g in zip(missing, grades):
                f.write(json.dumps({"qid": qid, "course_id": cid, "grade": max(g, 0),
                                    "added_by": "rescore"}, ensure_ascii=False) + "\n")
        queries, qrels = load_eval()

    meta = {c.course_id: _course_metadata(c) for c in courses}
    names = {c.course_id: c.name for c in courses}
    scorer = Scorer(queries, qrels, meta, names)
    table = {}
    for name, ranking in runs.items():
        s = summarize(scorer, ranking)
        old = runs_dir / f"{name}.json"
        cfg = json.loads(old.read_text()).get("_config", {}) if old.exists() else {}
        s["_config"] = cfg
        old.write_text(json.dumps(s, ensure_ascii=False, indent=2))
        table[name] = s
    print(f"{'run':24s} {'set':3s} {'nDCG':>6s} {'Succ':>6s} {'Pcond':>6s} {'judged':>6s}")
    for name, s in table.items():
        for g in ("ALL", "T", "M", "C"):
            if g not in s:
                continue
            row = s[g]
            print(f"{name:24s} {g:3s} {row['ndcg']:6.3f} {row['success']:6.3f} "
                  f"{row.get('p_cond', float('nan')):6.3f} {row['judged']:6.2f}")


if __name__ == "__main__":
    main()
