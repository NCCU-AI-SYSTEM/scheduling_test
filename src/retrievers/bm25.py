"""BM25 retriever with jieba Chinese tokenisation.

Pure Python (rank_bm25) — fast enough for 2.8k docs and removes any GPU/embedding
dependency for baseline runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from src.doc_builders import RetrievalDoc

# Drop pure punctuation/whitespace tokens after jieba cut
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in jieba.lcut(text, cut_all=False) if _TOKEN_RE.match(t)]


@dataclass(slots=True)
class BM25Retriever:
    docs: list[RetrievalDoc]
    bm25: BM25Okapi
    k: int = 20

    @classmethod
    def from_docs(cls, docs: list[RetrievalDoc], k: int = 20) -> "BM25Retriever":
        tokenised = [tokenize(d.text) for d in docs]
        return cls(docs=docs, bm25=BM25Okapi(tokenised), k=k)

    def search(self, query: str, k: int | None = None) -> list[tuple[RetrievalDoc, float]]:
        kk = k or self.k
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        # Top-k by score
        idx_sorted = sorted(range(len(scores)), key=lambda i: -scores[i])[:kk]
        return [(self.docs[i], float(scores[i])) for i in idx_sorted if scores[i] > 0]
