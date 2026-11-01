"""Dense encoders with each model's own query/document prompt conventions.

Getting the prompt wrong silently costs several nDCG points (e5 without
"query: ", Qwen3-Embedding without an instruction), so the conventions live in
one table instead of at call sites.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed" / "dense_cache"

QWEN_INSTRUCT = "Instruct: 根據學生的選課需求，找出內容符合的大學課程\nQuery: "


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_len: int = 512
    trust_remote_code: bool = False
    dtype: str = "float16"


SPECS: dict[str, EncoderSpec] = {s.name: s for s in [
    EncoderSpec("bge-m3", "BAAI/bge-m3", max_len=1024),
    EncoderSpec("bge-base-zh", "BAAI/bge-base-zh-v1.5", query_prefix="为这个句子生成表示以用于检索相关文章："),
    EncoderSpec("qwen3-emb-0.6b", "Qwen/Qwen3-Embedding-0.6B", query_prefix=QWEN_INSTRUCT, max_len=1024),
    EncoderSpec("qwen3-emb-4b", "Qwen/Qwen3-Embedding-4B", query_prefix=QWEN_INSTRUCT, max_len=1024),
    EncoderSpec("e5-large-instruct", "intfloat/multilingual-e5-large-instruct",
                query_prefix="Instruct: 根據學生的選課需求，找出內容符合的大學課程\nQuery: "),
    EncoderSpec("gte-multi-base", "Alibaba-NLP/gte-multilingual-base", trust_remote_code=True, max_len=1024),
]}


class Encoder:
    def __init__(self, spec: EncoderSpec, device: str = "cuda"):
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs = {"torch_dtype": torch.float16} if device == "cuda" and spec.dtype == "float16" else {}
        self.spec = spec
        self.device = device
        self.model = SentenceTransformer(spec.hf_id, device=device, trust_remote_code=spec.trust_remote_code,
                                         model_kwargs=kwargs)
        self.model.max_seq_length = spec.max_len

    def encode_docs(self, texts: list[str], batch_size: int = 16) -> tuple[np.ndarray, float]:
        """(embeddings, seconds). Cached on disk by model + text hash."""
        h = hashlib.sha256((self.spec.hf_id + str(self.spec.max_len)).encode())
        for t in texts:
            h.update(t.encode("utf-8", "replace"))
        path = CACHE / f"v4__{self.spec.name}__{h.hexdigest()[:16]}.npy"
        if path.exists():
            return np.load(path, allow_pickle=False), 0.0
        t0 = time.time()
        emb = self.model.encode([self.spec.doc_prefix + t for t in texts], batch_size=batch_size,
                                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        secs = time.time() - t0
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, emb.astype(np.float32), allow_pickle=False)
        return emb, secs

    def encode_queries(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode([self.spec.query_prefix + q for q in queries], batch_size=batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def query_latency(self, queries: list[str], n: int = 40) -> float:
        """Median seconds to encode ONE query (batch size 1), the online cost."""
        times = []
        for q in queries[:n]:
            t0 = time.time()
            self.model.encode([self.spec.query_prefix + q], normalize_embeddings=True, show_progress_bar=False)
            times.append(time.time() - t0)
        return float(np.median(times[3:])) if len(times) > 3 else float(np.median(times))
