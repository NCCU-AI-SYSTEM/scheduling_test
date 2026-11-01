"""Metrics for eval v3: graded, multi-gold, constraint-aware.

A course's effective relevance for a query is

    g'(q, c) = grade(q, c)  if c satisfies every hard condition of q  else 0

where `grade` is the LLM judge's 0-3 topical grade (or 3 for condition-only
queries, whose topical need is empty), and satisfaction is checked against the
DB. Unjudged courses count as 0; `judged@10` reports how often that happens so a
new system that surfaces unseen courses is visible rather than silently penalised.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from src.eval.conditions import satisfies
from src.parsers.base import ConditionResult

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "eval_v3"


def _norm_name(name: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", name or "")


@dataclass
class V3Query:
    qid: str
    set: str
    qtype: str
    query: str
    need: str
    cond: ConditionResult
    gold: list[str]

    @property
    def has_cond(self) -> bool:
        return bool(self.cond.must or self.cond.must_not)


def load_eval(eval_dir: Path = EVAL_DIR) -> tuple[list[V3Query], dict[str, dict[str, int]]]:
    queries = []
    for line in (eval_dir / "queries.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["need"] is None:  # M query not yet decomposed
            continue
        queries.append(V3Query(r["qid"], r["set"], r["qtype"], r["query"], r["need"],
                               ConditionResult(must=r["must"], must_not=r["must_not"]), r["gold"]))
    qrels: dict[str, dict[str, int]] = {}
    qpath = eval_dir / "qrels.jsonl"
    if qpath.exists():
        for line in qpath.open(encoding="utf-8"):
            r = json.loads(line)
            qrels.setdefault(r["qid"], {})[r["course_id"]] = r["grade"]
    return queries, qrels


class Scorer:
    def __init__(self, queries: list[V3Query], qrels: dict[str, dict[str, int]],
                 meta: dict[str, dict], names: dict[str, str]):
        self.q = {x.qid: x for x in queries}
        self.qrels = qrels
        self.meta = meta
        self.names = names

    def gain(self, q: V3Query, cid: str) -> int | None:
        """g'(q, c); None when the topical grade is needed but unjudged."""
        if q.has_cond and not satisfies(self.meta[cid], q.cond):
            return 0
        if not q.need:
            return 3
        return self.qrels.get(q.qid, {}).get(cid)

    def _ideal(self, q: V3Query, k: int) -> list[int]:
        if not q.need:  # every satisfying course is a perfect answer
            n_ok = sum(1 for m in self.meta.values() if not q.has_cond or satisfies(m, q.cond))
            return [3] * min(k, n_ok)
        gains = sorted((self.gain(q, c) or 0 for c in self.qrels.get(q.qid, {})), reverse=True)
        return gains[:k]

    def score(self, qid: str, ranked: list[str], k: int = 10) -> dict[str, float]:
        q = self.q[qid]
        top = ranked[:k]
        raw = [self.gain(q, c) for c in top]
        gains = [g or 0 for g in raw]
        dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = self._ideal(q, k)
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        gold_names = {_norm_name(self.names.get(g, "")) for g in q.gold}
        out = {
            "ndcg": dcg / idcg if idcg else 0.0,
            "p_rel": sum(g >= 2 for g in gains) / k,  # Precision@k, hit = gain >= 2
            "success": float(any(g >= 2 for g in gains)),
            "sib_recall": float(any(_norm_name(self.names.get(c, "")) in gold_names for c in top)),
            "strict_recall": float(any(c in q.gold for c in top)),
            "judged": sum(r is not None for r in raw) / max(1, len(top)),
        }
        if q.has_cond:
            out["p_cond"] = sum(satisfies(self.meta[c], q.cond) for c in top) / max(1, len(top))
        return out


def summarize(scorer: Scorer, runs: dict[str, list[str]], k: int = 10) -> dict[str, dict]:
    """Mean metrics overall and per set / per qtype."""
    groups: dict[str, list[dict]] = {}
    for qid, ranked in runs.items():
        q = scorer.q[qid]
        s = scorer.score(qid, ranked, k)
        for g in ("ALL", q.set, f"{q.set}:{q.qtype}"):
            groups.setdefault(g, []).append(s)
    out = {}
    for g, rows in sorted(groups.items()):
        keys = {key for r in rows for key in r}
        out[g] = {"n": len(rows)} | {
            key: round(sum(r[key] for r in rows if key in r) / max(1, sum(key in r for r in rows)), 4)
            for key in sorted(keys)
        }
    return out
