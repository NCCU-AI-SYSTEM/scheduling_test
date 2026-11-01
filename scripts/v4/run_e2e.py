"""End-to-end evaluation of the v4 pipeline on realistic (R) and synthetic (v3) queries.

Stages:
  plan    cache planner output per model                    data/v4/plans/<model>.jsonl
  run     run one system config, save top-10 per query       results/v4/e2e/<name>.runs.jsonl
  judge   grade every unjudged (need, course) in any run     data/v4/eval/r_qrels.jsonl
  score   metrics per system and per query group             results/v4/e2e/scores.json

Gain of course c for R query q (reference labels from gpt-oss:120b):
  0  if c violates q's hard conditions (DB check)
  lookup queries (named course / teacher): 3 if c is the named course family, else 0
  topical queries: LLM judge grade 0-3 for the need
  condition-only queries: 3
Metrics: nDCG@10, Success@10 (some gain >= 2), Precision@10 (hit = satisfies hard
conditions), MRR@10 on named targets, PrefAlign@5 (mean preference alignment of the
top 5 from aspect evidence, preference queries only), latency.

  python -m scripts.v4.run_e2e plan --planner gemma4:e4b
  python -m scripts.v4.run_e2e run --name v4_full --planner gemma4:e4b --encoder bge-m3 --reranker bge-reranker-v2-m3
  python -m scripts.v4.run_e2e judge && python -m scripts.v4.run_e2e score
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.eval.conditions import satisfies
from src.parsers.base import ConditionResult
from src.v4.catalog import load_catalog
from src.v4.planner import Plan, plan

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "v4" / "rset"
PLANS = ROOT / "data" / "v4" / "plans"
OUT = ROOT / "results" / "v4" / "e2e"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", s)


def _norm(s: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", s or "")


def load_r() -> list[dict]:
    return [r for r in (json.loads(x) for x in (EVAL / "r_gold.jsonl").open(encoding="utf-8")) if r.get("ref")]


def ref_plan(r: dict) -> Plan:
    return Plan.from_dict(r["ref"])


# ── plan ─────────────────────────────────────────────────────────────────────

def stage_plan(model: str, workers: int) -> None:
    PLANS.mkdir(parents=True, exist_ok=True)
    path = PLANS / f"{_slug(model)}.jsonl"
    done = {json.loads(x)["qid"] for x in path.open(encoding="utf-8")} if path.exists() else set()
    todo = [r for r in load_r() if r["qid"] not in done]

    def one(r):
        p = plan(r["query"], model)
        return {"qid": r["qid"], "ok": p.ok, "seconds": round(p.seconds, 3), "need": p.need,
                "named_course": p.named_course, "named_teacher": p.named_teacher,
                "must": p.cond.must, "must_not": p.cond.must_not, "prefs": p.prefs, "intents": p.intents}

    with ThreadPoolExecutor(workers) as ex, path.open("a", encoding="utf-8") as f:
        for row in tqdm(ex.map(one, todo), total=len(todo)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("plans", path)


def load_plans(model: str) -> dict[str, Plan]:
    out = {}
    for x in (PLANS / f"{_slug(model)}.jsonl").open(encoding="utf-8"):
        d = json.loads(x)
        out[d["qid"]] = Plan.from_dict(d, d["seconds"], d["ok"])
    return out


# ── run ──────────────────────────────────────────────────────────────────────

def stage_run(a) -> None:
    from src.v4.aspects import build_aspects
    from src.v4.pipeline import Config, V4Search

    rows = load_r()
    cat = None if a.struct else load_catalog()
    if a.system == "prod":
        from src.prod_replica import ProdReplica

        s = ProdReplica.build()
        t0 = time.time()
        cands = s.candidates([r["query"] for r in rows], "rrf", 50)
        runs = {r["qid"]: [cid for cid, _ in c[:10]] for r, c in zip(rows, cands)}
        _save(a.name, runs, {"system": "prod", "ms_per_q": round((time.time() - t0) / len(rows) * 1000, 1)})
        return
    enc = rr = None
    if a.encoder:
        from src.retrievers.encoders import SPECS, Encoder

        enc = Encoder(SPECS[a.encoder])
    if a.reranker:
        from src.rerankers.models import load

        rr = load(a.reranker)
    bundle = None
    if a.struct:  # runtime path: everything from course_struct.db, nothing from 1142.db
        from src.v4.struct_store import load_struct

        bundle = load_struct(embedding_models=(a.encoder,) if a.encoder else ())
        cat = bundle.catalog
    hype = None
    if a.hype:
        hype = {d["course_id"]: d["questions"] for d in (json.loads(x) for x in Path(a.hype).open(encoding="utf-8"))}
    aspects = bundle.aspects if bundle else build_aspects(cat, Path(a.aspects) if a.aspects else None)
    offerings = None
    if a.merge:
        from src.v4.offering import group_offerings

        offerings = bundle.offerings if bundle else group_offerings(cat)
    search = V4Search(cat, enc, rr, aspects, hype=hype, offerings=offerings,
                      doc_emb=bundle.embeddings.get(a.encoder) if bundle else None,
                      aliases=bundle.aliases if bundle else None, assess=bundle.assess if bundle else None)
    cfg = Config(use_conditions=not a.no_cond, use_named=not a.no_named, use_dense=enc is not None,
                 rerank=rr is not None, use_prefs=not a.no_prefs, use_hype=hype is not None,
                 use_bm25=not a.no_bm25, merge_offerings=a.merge, named_skips_conditions=a.named_fix)
    plans = {r["qid"]: ref_plan(r) for r in rows} if a.planner == "reference" else load_plans(a.planner)
    if a.raw_query:  # baseline: no planning at all, the sentence goes straight to retrieval
        plans = {r["qid"]: Plan(need=r["query"]) for r in rows}
    if a.text_raw:  # ablation: keep the plan's conditions, but retrieve with the full sentence
        for r in rows:
            plans[r["qid"]].need = r["query"]
    runs, times, plan_s = {}, [], []
    for r in tqdm(rows):
        p = plans[r["qid"]]
        hits = search.search(p, r["query"], 10, cfg)
        runs[r["qid"]] = [h.course_id for h in hits]
        times.append(search.timing["total"])
        plan_s.append(p.seconds)
    _save(a.name, runs, {"system": "v4", "struct": a.struct, "planner": a.planner, "encoder": a.encoder, "reranker": a.reranker,
                         "hype": bool(hype), "cfg": cfg.__dict__, "raw_query": a.raw_query,
                         "search_ms_p50": round(float(np.median(times)) * 1000, 1),
                         "search_ms_p95": round(float(np.percentile(times, 95)) * 1000, 1),
                         "plan_s_p50": round(float(np.median(plan_s)), 3) if plan_s else 0})


def _save(name: str, runs: dict, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.runs.jsonl").open("w", encoding="utf-8") as f:
        for qid, ranked in runs.items():
            f.write(json.dumps({"qid": qid, "ranked": ranked}, ensure_ascii=False) + "\n")
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(name, json.dumps(meta, ensure_ascii=False)[:300])


def load_runs() -> dict[str, dict[str, list[str]]]:
    return {f.name.removesuffix(".runs.jsonl"): {json.loads(x)["qid"]: json.loads(x)["ranked"]
                                                 for x in f.open(encoding="utf-8")}
            for f in sorted(OUT.glob("*.runs.jsonl"))}


# ── judge ────────────────────────────────────────────────────────────────────

def _is_lookup(r: dict) -> bool:
    return bool(r["ref"].get("named_course") or r["ref"].get("named_teacher"))


def stage_judge(workers: int) -> None:
    from scripts.build_eval_v3 import judge_doc
    from src.eval.judge import judge

    rows = {r["qid"]: r for r in load_r()}
    by_id = {c.course_id: c.course for c in load_catalog(with_reviews=False)}
    qpath = EVAL / "r_qrels.jsonl"
    have = set()
    if qpath.exists():
        have = {(d["qid"], d["course_id"]) for d in (json.loads(x) for x in qpath.open(encoding="utf-8"))}
    todo = sorted({(qid, cid) for run in load_runs().values() for qid, ranked in run.items()
                   for cid in ranked[:10] if qid in rows and rows[qid]["ref"]["need"].strip()
                   and (qid, cid) not in have and cid in by_id})
    print("judging", len(todo))

    def one(pair):
        qid, cid = pair
        return judge(rows[qid]["ref"]["need"], cid, judge_doc(by_id[cid]))

    with ThreadPoolExecutor(workers) as ex:
        grades = list(tqdm(ex.map(one, todo), total=len(todo)))
    with qpath.open("a", encoding="utf-8") as f:
        for (qid, cid), g in zip(todo, grades):
            f.write(json.dumps({"qid": qid, "course_id": cid, "grade": max(g, 0)}, ensure_ascii=False) + "\n")


# ── score ────────────────────────────────────────────────────────────────────

def stage_score(aspect_file: str) -> None:
    from src.v4.aspects import build_aspects

    rows = {r["qid"]: r for r in load_r()}
    cat = load_catalog()
    meta = {c.course_id: c.meta for c in cat}
    cmap = {c.course_id: c for c in cat}
    aspects = build_aspects(cat, Path(aspect_file) if aspect_file else None)
    qrels: dict[str, dict[str, int]] = {}
    qpath = EVAL / "r_qrels.jsonl"
    if qpath.exists():
        for x in qpath.open(encoding="utf-8"):
            d = json.loads(x)
            qrels.setdefault(d["qid"], {})[d["course_id"]] = d["grade"]

    def gain(r: dict, cid: str) -> int | None:
        ref = r["ref"]
        if cid not in meta:
            return 0
        cond = ConditionResult(must=ref["must"], must_not=ref["must_not"])
        if (cond.must or cond.must_not) and not satisfies(meta[cid], cond):
            return 0
        if _is_lookup(r) and not ref["need"].strip():
            c = cmap[cid].course
            ok_c = not ref.get("named_course") or _norm(ref["named_course"]) in _norm(c.name)
            ok_t = not ref.get("named_teacher") or ref["named_teacher"] in c.teacher
            return 3 if ok_c and ok_t else 0
        if ref["need"].strip():
            return qrels.get(r["qid"], {}).get(cid)
        return 3

    def ideal(r: dict) -> float:
        ref = r["ref"]
        if ref["need"].strip():
            gs = sorted((gain(r, c) or 0 for c in qrels.get(r["qid"], {})), reverse=True)[:10]
        else:
            gs = sorted((gain(r, c) or 0 for c in meta), reverse=True)[:10]
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs))

    ideal_cache = {qid: ideal(r) for qid, r in rows.items()}

    # class level: cross-listed codes of one class count once, gain = best member gain
    from src.v4.offering import group_offerings

    offs = group_offerings(cat)
    key_of = {cid: o.key for o in offs for cid in o.course_ids}
    members = {o.key: o.course_ids for o in offs}

    def cgain(r: dict, key) -> int | None:
        known = [g for g in (gain(r, c) for c in members[key]) if g is not None]
        return max(known) if known else None

    def ideal_class(r: dict) -> float:
        keys = ({key_of[c] for c in qrels.get(r["qid"], {}) if c in key_of} if r["ref"]["need"].strip()
                else set(members))
        gs = sorted((cgain(r, k) or 0 for k in keys), reverse=True)[:10]
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs))

    ideal_class_cache = {qid: ideal_class(r) for qid, r in rows.items()}

    def pref_align(r: dict, top: list[str]) -> float | None:
        prefs = [p for p in r["ref"].get("prefs", []) if p.get("want") in ("low", "high")]
        vals = []
        for cid in top[:5]:
            for p in prefs:
                v = aspects.get(cid, {}).get(p["aspect"])
                if v is not None:
                    vals.append((3 - v) / 2 if p["want"] == "low" else (v - 3) / 2)
        return float(np.mean(vals)) if vals else None

    out = {}
    for name, run in load_runs().items():
        groups: dict[str, list[dict]] = {}
        for qid, ranked in run.items():
            r = rows.get(qid)
            if r is None or {"out_of_scope", "unsupported"} & set(r["ref"].get("intents", [])):
                continue
            raw = [gain(r, c) for c in ranked[:10]]
            g = [x or 0 for x in raw]
            dcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(g))
            m = {"ndcg": dcg / ideal_cache[qid] if ideal_cache[qid] else 0.0,
                 "success": float(any(x >= 2 for x in g)),
                 "judged": sum(x is not None for x in raw) / max(1, len(raw))}
            seen, cg = set(), []
            for c in ranked[:10]:
                key = key_of.get(c, c)
                cg.append(0 if key in seen or key not in members else (cgain(r, key) or 0))
                seen.add(key)
            cdcg = sum((2 ** x - 1) / math.log2(i + 2) for i, x in enumerate(cg))
            m["ndcg_class"] = cdcg / ideal_class_cache[qid] if ideal_class_cache[qid] else 0.0
            m["distinct"] = len(seen)
            ref = r["ref"]
            cond = ConditionResult(must=ref["must"], must_not=ref["must_not"])
            if cond.must or cond.must_not:
                m["p_cond"] = sum(satisfies(meta[c], cond) for c in ranked[:10] if c in meta) / max(1, len(ranked[:10]))
            if _is_lookup(r) and not ref["need"].strip():
                rank = next((i for i, x in enumerate(g) if x == 3), None)
                m["mrr"] = 1 / (rank + 1) if rank is not None else 0.0
                m["hit1"] = float(rank == 0)
            pa = pref_align(r, ranked)
            if pa is not None and ref.get("prefs"):
                m["pref_align"] = pa
            pref_only = not ref["need"].strip() and not _is_lookup(r) and not (cond.must or cond.must_not)
            if pref_only:  # every course "matches"; only the preference ordering is measurable
                if "pref_align" in m:
                    groups.setdefault("pref_only", []).append({"pref_align": m["pref_align"]})
                    groups.setdefault("has_pref", []).append({"pref_align": m["pref_align"]})
                continue
            kinds = ["ALL", "lookup" if _is_lookup(r) else ("topic" if ref["need"].strip() else "cond_only")]
            if cond.must or cond.must_not:
                kinds.append("has_cond")
            if ref.get("prefs"):
                kinds.append("has_pref")
            for k in kinds:
                groups.setdefault(k, []).append(m)
        out[name] = {k: {"n": len(v)} | {key: round(float(np.mean([x[key] for x in v if key in x])), 4)
                                          for key in sorted({kk for x in v for kk in x})}
                     for k, v in groups.items()}
        meta_path = OUT / f"{name}.meta.json"
        if meta_path.exists():
            out[name]["_meta"] = json.loads(meta_path.read_text())
    (OUT / "scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    hdr = (f"{'system':30s} {'nDCG':>6s} {'Succ':>6s} {'Pcond':>6s} {'MRR':>6s} {'Pref@5':>6s} {'topic':>6s} "
           f"{'judged':>6s} {'nDCGcls':>7s} {'distinct':>8s}")
    print(hdr)
    for name, g in out.items():
        a = g.get("ALL", {})
        print(f"{name:30s} {a.get('ndcg', 0):6.3f} {a.get('success', 0):6.3f} "
              f"{g.get('has_cond', {}).get('p_cond', float('nan')):6.3f} {g.get('lookup', {}).get('mrr', float('nan')):6.3f} "
              f"{g.get('has_pref', {}).get('pref_align', float('nan')):6.3f} {g.get('topic', {}).get('ndcg', float('nan')):6.3f} "
              f"{a.get('judged', 0):6.2f} {a.get('ndcg_class', 0):7.3f} {a.get('distinct', 0):8.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["plan", "run", "judge", "score"])
    p.add_argument("--planner", default="gemma4:e4b")
    p.add_argument("--name", default="")
    p.add_argument("--system", default="v4", choices=["v4", "prod"])
    p.add_argument("--encoder", default="")
    p.add_argument("--reranker", default="")
    p.add_argument("--hype", default="")
    p.add_argument("--aspects", default="")
    p.add_argument("--no-cond", action="store_true")
    p.add_argument("--no-named", action="store_true")
    p.add_argument("--no-prefs", action="store_true")
    p.add_argument("--no-bm25", action="store_true")
    p.add_argument("--raw-query", action="store_true")
    p.add_argument("--text-raw", action="store_true")
    p.add_argument("--merge", action="store_true", help="one hit per class (cross-listed codes merged)")
    p.add_argument("--named-fix", action="store_true", help="named-course queries skip condition filtering")
    p.add_argument("--struct", action="store_true", help="read courses, aspects, vectors, aliases from course_struct.db")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.stage == "plan":
        stage_plan(a.planner, a.workers)
    elif a.stage == "run":
        stage_run(a)
    elif a.stage == "judge":
        stage_judge(a.workers)
    else:
        stage_score(a.aspects)


if __name__ == "__main__":
    main()
