"""Retrieval evaluation metrics.

Each query has a *set* of relevant doc_ids (gold). We support binary relevance.

Metrics:
  Recall@K, MRR@K, nDCG@K, Hit@K
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class QueryEval:
    qid: str
    relevant: set[str]
    retrieved: list[str]


def recall_at_k(qe: QueryEval, k: int) -> float:
    if not qe.relevant:
        return 0.0
    top = qe.retrieved[:k]
    hit = sum(1 for d in top if d in qe.relevant)
    return hit / len(qe.relevant)


def hit_at_k(qe: QueryEval, k: int) -> float:
    return 1.0 if any(d in qe.relevant for d in qe.retrieved[:k]) else 0.0


def mrr_at_k(qe: QueryEval, k: int) -> float:
    for i, d in enumerate(qe.retrieved[:k], start=1):
        if d in qe.relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(qe: QueryEval, k: int) -> float:
    if not qe.relevant:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(qe.retrieved[:k], start=1):
        if d in qe.relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(qe.relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(evals: list[QueryEval], ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    n = len(evals)
    if n == 0:
        return {}
    out: dict[str, float] = {"n_queries": n}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(e, k) for e in evals) / n
        out[f"hit@{k}"] = sum(hit_at_k(e, k) for e in evals) / n
        out[f"mrr@{k}"] = sum(mrr_at_k(e, k) for e in evals) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(e, k) for e in evals) / n
    return out
