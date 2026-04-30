"""Reciprocal Rank Fusion for combining multiple retrievers."""

from __future__ import annotations

from collections import defaultdict

from src.doc_builders import RetrievalDoc


def rrf_fuse(
    runs: list[list[tuple[RetrievalDoc, float]]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[RetrievalDoc, float]]:
    """Reciprocal Rank Fusion. `k` is the dampening constant (default 60).

    Documents are matched by `course_id` so multi-field doc copies collapse.
    """
    fused: dict[str, float] = defaultdict(float)
    rep: dict[str, RetrievalDoc] = {}
    for run in runs:
        for rank, (doc, _) in enumerate(run):
            cid = doc.course_id
            fused[cid] += 1.0 / (k + rank + 1)
            rep.setdefault(cid, doc)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [(rep[cid], score) for cid, score in ordered]
