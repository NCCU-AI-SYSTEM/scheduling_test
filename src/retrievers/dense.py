"""Dense retriever using sentence-transformers (default: BAAI/bge-m3).

Caches embeddings to disk keyed by (model_name, doc_count + sha256 of texts).
Uses MPS on Apple Silicon, CUDA if available, else CPU.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.doc_builders import RetrievalDoc

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "dense_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _device() -> str:
    # MPS has intermittent hangs on first query encode; prefer CPU for stability
    # Set env FORCE_MPS=1 to override
    import os
    if os.environ.get("FORCE_MPS") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _cache_key(model_name: str, texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass(slots=True)
class DenseRetriever:
    docs: list[RetrievalDoc]
    embeddings: np.ndarray  # (N, D) L2-normalised
    model: SentenceTransformer
    k: int = 20

    @classmethod
    def from_docs(
        cls,
        docs: list[RetrievalDoc],
        model_name: str = "BAAI/bge-m3",
        k: int = 20,
        batch_size: int = 16,
        cache: bool = True,
    ) -> "DenseRetriever":
        device = _device()
        model = SentenceTransformer(model_name, device=device)
        texts = [d.text for d in docs]
        cache_path = CACHE_DIR / f"{model_name.replace('/', '_')}__{_cache_key(model_name, texts)}.pkl"
        if cache and cache_path.exists():
            with cache_path.open("rb") as f:
                emb = pickle.load(f)
        else:
            emb = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            if cache:
                with cache_path.open("wb") as f:
                    pickle.dump(emb, f)
        return cls(docs=docs, embeddings=emb, model=model, k=k)

    def search(self, query: str, k: int | None = None) -> list[tuple[RetrievalDoc, float]]:
        kk = k or self.k
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q  # cosine since both normalised
        idx = np.argsort(-scores)[:kk]
        return [(self.docs[i], float(scores[i])) for i in idx]
