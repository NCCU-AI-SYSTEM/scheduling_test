"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Batch-reranks all queries at once (far faster than per-query calls on CPU/MPS).
Typical throughput on M-series Apple Silicon: ~30-50 pairs/s.

Two entry points:
  rerank(query, hits)        -- single query
  batch_rerank(queries, hits_list) -- all queries in one forward pass
"""

from __future__ import annotations

import torch

from src.doc_builders import RetrievalDoc

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        import os
        from FlagEmbedding import FlagReranker
        force_cpu = os.environ.get("FORCE_CPU", "0") == "1"
        if force_cpu:
            use_fp16 = False
        else:
            use_fp16 = torch.backends.mps.is_available() or torch.cuda.is_available()
        _reranker = FlagReranker(_MODEL_NAME, use_fp16=use_fp16)
    return _reranker


def rerank(
    query: str,
    hits: list[tuple[RetrievalDoc, float]],
    top_k: int = 10,
) -> list[tuple[RetrievalDoc, float]]:
    if not hits:
        return []
    ranker = _get_reranker()
    pairs = [[query, doc.text] for doc, _ in hits]
    scores = ranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
    return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]


def batch_rerank(
    queries: list[str],
    hits_list: list[list[tuple[RetrievalDoc, float]]],
    top_k: int = 10,
    batch_size: int = 512,
) -> list[list[tuple[RetrievalDoc, float]]]:
    """Rerank all queries in a single batched forward pass.

    More efficient than calling rerank() per query because the model
    processes all pairs together with optimal batching.
    """
    if not queries:
        return []
    ranker = _get_reranker()

    # flatten all pairs
    all_pairs: list[list[str]] = []
    offsets: list[int] = [0]
    for q, hits in zip(queries, hits_list):
        for doc, _ in hits:
            all_pairs.append([q, doc.text])
        offsets.append(len(all_pairs))

    # score in batches
    all_scores: list[float] = []
    for i in range(0, len(all_pairs), batch_size):
        chunk = all_pairs[i : i + batch_size]
        s = ranker.compute_score(chunk, normalize=True)
        if isinstance(s, float):
            s = [s]
        all_scores.extend(s)

    # distribute back
    results: list[list[tuple[RetrievalDoc, float]]] = []
    for idx, (hits, start, end) in enumerate(
        zip(hits_list, offsets[:-1], offsets[1:])
    ):
        scores = all_scores[start:end]
        ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
        results.append([(doc, float(sc)) for (doc, _), sc in ranked[:top_k]])
    return results


__all__ = ["batch_rerank", "rerank"]
