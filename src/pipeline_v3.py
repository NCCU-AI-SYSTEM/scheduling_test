"""Batched retrieval pipeline for the v3 experiments.

One `Searcher` holds the corpus, a BM25 index and a dense matrix, and answers a
whole query set at once:

    candidates  : BM25 / dense / RRF, optionally restricted to an allowed set per
                  query (true pre-filter: the mask is applied before top-k, so a
                  filtered query still gets `retrieve_k` candidates)
    rerank      : cross-encoder over the candidates
    soft score  : optional per-query multiplier from structured conditions

Course ids are the unit everywhere; `RetrievalDoc` is only looked up for text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.doc_builders import BUILDERS, RetrievalDoc
from src.loader import Course, load_courses
from src.retrievers.bm25 import BM25Retriever, tokenize

Ranked = list[tuple[str, float]]


@dataclass
class Searcher:
    courses: list[Course]
    docs: list[RetrievalDoc]
    bm25: BM25Retriever
    dense_model: object | None = None
    dense_emb: np.ndarray | None = None
    ids: list[str] = field(default_factory=list)
    pos: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        doc: str = "d-v2",
        dense_model: str | None = "BAAI/bge-m3",
        year: str = "114",
        semester: str = "2",
        docs: list[RetrievalDoc] | None = None,
    ) -> "Searcher":
        courses = load_courses(year=year, semester=semester)
        docs = docs if docs is not None else BUILDERS[doc](courses)
        bm25 = BM25Retriever.from_docs(docs)
        model = emb = None
        if dense_model:
            from src.retrievers import DenseRetriever

            dr = DenseRetriever.from_docs(docs, model_name=dense_model)
            model, emb = dr.model, dr.embeddings
        ids = [d.course_id for d in docs]
        return cls(courses, docs, bm25, model, emb, ids, {c: i for i, c in enumerate(ids)})

    def doc(self, course_id: str) -> RetrievalDoc:
        return self.docs[self.pos[course_id]]

    # ── first stage ──────────────────────────────────────────────────────────

    def _mask(self, allowed: set[str] | None) -> np.ndarray | None:
        if allowed is None:
            return None
        m = np.zeros(len(self.ids), dtype=bool)
        for cid in allowed:
            if cid in self.pos:
                m[self.pos[cid]] = True
        return m

    @staticmethod
    def _topk(scores: np.ndarray, k: int, mask: np.ndarray | None, ids: list[str]) -> Ranked:
        s = scores.astype(np.float64, copy=True)
        if mask is not None:
            s[~mask] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(ids[i], float(s[i])) for i in idx]

    def bm25_scores(self, query: str) -> np.ndarray:
        toks = tokenize(query)
        if not toks:
            return np.zeros(len(self.ids))
        return np.asarray(self.bm25.bm25.get_scores(toks))

    def dense_scores(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        q = self.dense_model.encode(
            queries, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return q @ self.dense_emb.T  # (Q, N)

    def candidates(
        self,
        queries: list[str],
        mode: str = "rrf",
        k: int = 50,
        bm25_queries: list[str] | None = None,
        allowed: list[set[str] | None] | None = None,
        rrf_k: int = 60,
    ) -> list[Ranked]:
        n = len(queries)
        bm25_queries = bm25_queries or queries
        allowed = allowed or [None] * n
        dense = self.dense_scores(queries) if mode in ("dense", "rrf") else None
        out: list[Ranked] = []
        for i in range(n):
            mask = self._mask(allowed[i])
            runs: list[Ranked] = []
            if mode in ("bm25", "rrf"):
                runs.append(self._topk(self.bm25_scores(bm25_queries[i]), k, mask, self.ids))
            if mode in ("dense", "rrf"):
                runs.append(self._topk(dense[i], k, mask, self.ids))
            out.append(runs[0] if len(runs) == 1 else rrf(runs, k, rrf_k))
        return out

    # ── second stage ─────────────────────────────────────────────────────────

    def rerank(self, queries: list[str], cands: list[Ranked], top_k: int = 20) -> list[Ranked]:
        from src.rerankers import batch_rerank

        hits = [[(self.doc(cid), s) for cid, s in c] for c in cands]
        ranked = batch_rerank(queries, hits, top_k=top_k)
        return [[(d.course_id, s) for d, s in r] for r in ranked]


def rrf(runs: list[Ranked], top_k: int, k: int = 60) -> Ranked:
    fused: dict[str, float] = {}
    for run in runs:
        for rank, (cid, _) in enumerate(run):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
