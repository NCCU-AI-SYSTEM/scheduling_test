"""Deployment presets: the same search on a 16 GB GPU or on a CPU-only server.

Everything comes from data/v4/course_struct.db (document vectors included), so start-up
encodes nothing. The two presets differ only in where models run and whether the
reranker is used; numbers behind each choice are in docs/v4/REPORT.md 6.6 and 7.1.

  gpu  planner gemma4:12b on the GPU, e5 query encoding on the GPU, Qwen3-Reranker-0.6B on the head
  cpu  planner on CPU threads (Ollama num_gpu=0), e5 on CPU, no reranker (15-45 s per query on CPU
       for +0.011 nDCG end to end)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.v4.pipeline import Config, V4Search
from src.v4.struct_store import StructBundle, load_struct


@dataclass(frozen=True)
class Preset:
    name: str
    planner: str
    planner_options: dict = field(default_factory=dict)
    encoder: str = "e5-large-instruct"
    device: str = "cuda"
    reranker: str | None = "qwen3-rerank-0.6b"
    threads: int | None = None


def _threads() -> int:
    return int(os.environ.get("V4_THREADS", os.cpu_count() or 4))


def preset(name: str, planner: str | None = None, threads: int | None = None) -> Preset:
    if name == "gpu":
        return Preset("gpu", planner or "gemma4:12b")
    if name == "cpu":
        t = threads or _threads()
        return Preset("cpu", planner or os.environ.get("V4_CPU_PLANNER", "gemma4:e4b"),
                      {"num_gpu": 0, "num_thread": t}, device="cpu", reranker=None, threads=t)
    raise ValueError(f"unknown preset {name!r}: use gpu or cpu")


def build(p: Preset, bundle: StructBundle | None = None) -> tuple[V4Search, StructBundle, Config]:
    from src.retrievers.encoders import SPECS, Encoder

    if p.threads:
        import torch

        torch.set_num_threads(p.threads)
    b = bundle or load_struct(embedding_models=(p.encoder,))
    rr = None
    if p.reranker:
        from src.rerankers.models import load as load_reranker

        rr = load_reranker(p.reranker, device=p.device)
    search = V4Search(b.catalog, Encoder(SPECS[p.encoder], device=p.device), rr, b.aspects,
                      offerings=b.offerings, doc_emb=b.embeddings[p.encoder], aliases=b.aliases, assess=b.assess)
    cfg = Config(rerank=rr is not None, merge_offerings=True, named_skips_conditions=True)
    return search, b, cfg
