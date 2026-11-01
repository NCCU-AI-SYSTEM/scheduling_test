"""Build eval v3: multi-gold, graded, constraint-aware.

Why: eval_synth has one gold per query. 58% of courses have same-name sibling
sections and constraint queries have hundreds of valid answers, so single-gold
R@10 both under-counts (siblings, other valid courses) and saturates (topic
queries are paraphrases of their own gold doc). Rahmani et al. (SIGIR 2024)
show sparse single-source judgments collapse system-ranking agreement.

Three query sets:
  T  topic + colloquial from eval_synth_clean        relevance = LLM grade 0-3
  M  constraint queries from eval_synth_clean         relevance = LLM grade on the
                                                       topical part AND DB check on
                                                       the structured part
  C  eval_conditions_v1 (pure conditions, oracle)     relevance = DB check only

Stages (each re-runnable, outputs under data/eval_v3/):
  sample    pick queries                               -> queries.jsonl
  decompose LLM splits M queries into need + conditions, kept only where the
            gold course satisfies every kept clause    -> queries.jsonl (updated)
  pool      top-20 from several systems (GPU)          -> pool.jsonl
  judge     LLM grades every pooled (need, course)     -> qrels.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.eval.conditions import check, satisfies
from src.parsers.base import ConditionResult
from src.parsers.ollama_parser import _PROMPT, sanitize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval_v3"
SYNTH = ROOT / "data" / "raw" / "eval_synth_clean.jsonl"
COND = ROOT / "data" / "raw" / "eval_conditions_v1.jsonl"
DECOMP_MODEL = "qwen3:14b"


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── sample ───────────────────────────────────────────────────────────────────


def stage_sample(n_topic: int, n_colloquial: int, n_mixed: int, n_cond: int, seed: int) -> None:
    rng = random.Random(seed)
    synth = _read(SYNTH)
    by_type: dict[str, list[dict]] = {}
    for r in synth:
        by_type.setdefault(r["type"], []).append(r)
    rows: list[dict] = []
    for qtype, n, qset in (("topic", n_topic, "T"), ("colloquial", n_colloquial, "T"),
                           ("constraint", n_mixed, "M")):
        for r in rng.sample(by_type[qtype], n):
            rows.append({"qid": r["qid"], "set": qset, "qtype": qtype, "query": r["query"],
                         "gold": [r["gold_course_id"]], "need": r["query"] if qset == "T" else None,
                         "must": {}, "must_not": {}})
    # C: stratify by complexity x negation so complex / negated queries are not drowned out
    cond = _read(COND)
    strata: dict[tuple, list[dict]] = {}
    for r in cond:
        strata.setdefault((r["complexity"], r["has_negation"]), []).append(r)
    per = max(1, n_cond // len(strata))
    for key, items in sorted(strata.items()):
        for r in rng.sample(items, min(per, len(items))):
            rows.append({"qid": r["qid"], "set": "C", "qtype": f"{key[0]}{'-neg' if key[1] else ''}",
                         "query": r["queries"]["zh"], "gold": [r["gold_course_id"]], "need": "",
                         "must": r["must"], "must_not": r["must_not"]})
    _write(OUT / "queries.jsonl", rows)
    print(Counter((r["set"], r["qtype"]) for r in rows))


# ── decompose ────────────────────────────────────────────────────────────────

_GENERIC = re.compile(
    r"課程|課|的|有沒有|有什麼|哪些|想找|想修|想上|推薦|請問|我|嗎|呢|啊|一門|一些|可以|選|"
    r"中文|英文|日文|授課|上課|學分|必修|選修|通識|系|週.|星期.|早上|下午|晚上|中午|[0-9一二三四五六七八九十]"
)


def _clean_need(need: str) -> str:
    """Drop needs that are only condition words / filler ("中文課", "三學分的課")."""
    return need if len(_GENERIC.sub("", need)) >= 2 else ""


_SCHEMA = {
    "type": "object",
    "properties": {
        "need": {"type": "string"},
        "must": {"type": "object"},
        "must_not": {"type": "object"},
    },
    "required": ["need", "must", "must_not"],
}


def _decompose(query: str) -> dict:
    import ollama

    resp = ollama.chat(
        model=DECOMP_MODEL,
        messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
        format=_SCHEMA,
        think=False,
        options={"temperature": 0},
        keep_alive="30m",
    )
    return json.loads(resp["message"]["content"])


def stage_decompose(workers: int) -> None:
    from src.doc_builders import _course_metadata
    from src.loader import load_courses

    meta = {c.course_id: _course_metadata(c) for c in load_courses()}
    rows = _read(OUT / "queries.jsonl")
    todo = [r for r in rows if r["set"] == "M" and r["need"] is None]
    with ThreadPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(lambda r: _decompose(r["query"]), todo), total=len(todo)))
    dropped: Counter = Counter()
    for r, d in zip(todo, results):
        cond = ConditionResult(must=sanitize(d.get("must")), must_not=sanitize(d.get("must_not")))
        verdict = check(meta[r["gold"][0]], cond)
        # keep only clauses the gold course satisfies: a clause the gold violates is
        # either a parse error or a synth-generation error, and neither is a fair test
        for key, ok in verdict.items():
            if not ok:
                side, dim = key.split(":")
                (cond.must if side == "must" else cond.must_not).pop(dim, None)
                dropped[dim] += 1
        r["need"] = _clean_need((d.get("need") or "").strip())
        r["must"], r["must_not"] = cond.must, cond.must_not
        r["decomp_raw"] = d
    _write(OUT / "queries.jsonl", rows)
    print("dropped clauses (gold violates):", dict(dropped))
    print("M with empty need:", sum(1 for r in todo if not r["need"]))


# ── pool ─────────────────────────────────────────────────────────────────────


def stage_pool(depth: int) -> None:
    from scripts.run_experiment import _expand_for_bm25
    from src.doc_builders import BUILDERS
    from src.pipeline_v3 import Searcher

    rows = _read(OUT / "queries.jsonl")
    judged = [r for r in rows if r["need"]]  # C and need-less M are DB-judged only
    queries = [r["query"] for r in judged]
    needs = [r["need"] for r in judged]

    s = Searcher.build("d-v2")
    pools: dict[str, dict[str, list[str]]] = {r["qid"]: {} for r in judged}

    def add(name: str, ranked_lists) -> None:
        for r, ranked in zip(judged, ranked_lists):
            pools[r["qid"]][name] = [cid for cid, _ in ranked[:depth]]

    add("bm25_v2", s.candidates(queries, "bm25", depth))
    add("dense_v2", s.candidates(queries, "dense", depth))
    add("dense_v2_need", s.candidates(needs, "dense", depth))
    e7_cands = s.candidates(queries, "rrf", 50, bm25_queries=[_expand_for_bm25(q) for q in queries])
    add("e7", s.rerank(queries, e7_cands, depth))
    # constraint-respecting pool: without it, the pool for M queries is dominated by
    # topical courses that violate the conditions and the valid ones go unjudged
    allowed = []
    for r in judged:
        cond = ConditionResult(must=r["must"], must_not=r["must_not"])
        allowed.append({cid for cid in s.ids if satisfies(s.doc(cid).metadata, cond)}
                       if r["must"] or r["must_not"] else None)
    add("oracle_dense_need", s.candidates(needs, "dense", depth, allowed=allowed))

    obj = Searcher.build(docs=BUILDERS["d-obj"](s.courses))
    add("dense_obj", obj.candidates(queries, "dense", depth))

    out = []
    for r in judged:
        systems = pools[r["qid"]]
        union = list(dict.fromkeys([*r["gold"], *(c for lst in systems.values() for c in lst)]))
        out.append({"qid": r["qid"], "systems": systems, "pool": union})
    _write(OUT / "pool.jsonl", out)
    sizes = [len(p["pool"]) for p in out]
    print(f"pooled {len(out)} queries, avg pool {sum(sizes)/len(sizes):.1f}")


# ── judge ────────────────────────────────────────────────────────────────────


def judge_doc(c) -> str:
    """What the judge sees: raw course fields only, no LLM-generated meta (avoids
    rewarding systems that index the same generated text the judge would read)."""
    topics = "、".join(c.weekly_topics[:18])
    return (f"課名：{c.name}（{c.name_en}）\n開課單位：{c.unit}\n教師：{c.teacher}\n"
            f"課程目標：{(c.objective or '')[:500]}\n週次主題：{topics[:500]}")


def stage_judge(workers: int) -> None:
    from src.eval.judge import judge
    from src.loader import load_courses

    by_id = {c.course_id: c for c in load_courses()}
    rows = {r["qid"]: r for r in _read(OUT / "queries.jsonl")}
    pools = _read(OUT / "pool.jsonl")
    pairs = [(rows[p["qid"]]["need"], cid) for p in pools for cid in p["pool"]]
    print(f"judging {len(pairs)} pairs")

    def run(pair):
        need, cid = pair
        return judge(need, cid, judge_doc(by_id[cid]))

    with ThreadPoolExecutor(workers) as ex:
        grades = list(tqdm(ex.map(run, pairs), total=len(pairs)))
    qrels = []
    i = 0
    for p in pools:
        for cid in p["pool"]:
            qrels.append({"qid": p["qid"], "course_id": cid, "grade": max(grades[i], 0)})
            i += 1
    _write(OUT / "qrels.jsonl", qrels)
    print("grade dist:", Counter(q["grade"] for q in qrels))
    gold = {(r["qid"], r["gold"][0]) for r in rows.values()}
    g = [q["grade"] for q in qrels if (q["qid"], q["course_id"]) in gold]
    print(f"gold graded >=2: {sum(x >= 2 for x in g)}/{len(g)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sample", "decompose", "pool", "judge"])
    p.add_argument("--n-topic", type=int, default=120)
    p.add_argument("--n-colloquial", type=int, default=120)
    p.add_argument("--n-mixed", type=int, default=200)
    p.add_argument("--n-cond", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.stage == "sample":
        stage_sample(a.n_topic, a.n_colloquial, a.n_mixed, a.n_cond, a.seed)
    elif a.stage == "decompose":
        stage_decompose(a.workers)
    elif a.stage == "pool":
        stage_pool(a.depth)
    else:
        stage_judge(a.workers)


if __name__ == "__main__":
    main()
