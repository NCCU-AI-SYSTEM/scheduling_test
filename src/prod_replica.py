"""Offline replica of the production retrieve_tool (CourseLangChain feat/mcp-test).

Mirrors CourseDataPrep (src/prep/chunk.py, init-scripts/03-schema.sql) and
CourseLangChain tools/retrieve.py `_RRF_SQL`, minus PostgreSQL:

  BM25   jieba over content_text = name nameen teacher objective syllabus
         schedule evaluation textbook   (ParadeDB `|||` = OR, like rank_bm25)
  dense  BAAI/bge-base-zh-v1.5 over per-field chunks (320 tokens, overlap 40,
         header "課程名稱 / 授課老師" + section label); a course scores by its
         best chunk (MIN distance), query encoded without instruction prefix
  fuse   RRF k=60 over the top-50 of each leg; no reranker

Known deviations: rank_bm25 vs Tantivy BM25 scoring constants, and 114-2 data
instead of 115-1. Both apply equally to every system compared here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.doc_builders import RetrievalDoc, _course_metadata
from src.loader import Course, load_courses
from src.pipeline_v3 import Ranked, Searcher, rrf
from src.retrievers.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "processed" / "dense_cache"
EMBED_MODEL = "BAAI/bge-base-zh-v1.5"

CHUNK_FIELDS = (
    ("課程目標", "objective"),
    ("課程簡介", "syllabus"),
    ("每週進度", "schedule"),
    ("評量方式", "evaluation"),
    ("指定書目", "textbook"),
)
SEPARATORS = ["\n\n", "\n", "|", "。", "；", "，", " ", ""]


def content_text(c: Course) -> str:
    return " ".join([c.name, c.name_en, c.teacher, c.objective, c.syllabus,
                     c.schedule, c.evaluation, c.textbook])


def build_chunks(splitter, c: Course) -> list[str]:
    title = " / ".join(x for x in (c.name, c.name_en) if x).strip()
    header = f"課程名稱: {title}"
    if c.teacher.strip():
        header += f"\n授課老師: {c.teacher.strip()}"
    chunks = []
    for label, attr in CHUNK_FIELDS:
        text = (getattr(c, attr) or "").strip()
        for seg in splitter.split_text(text) if text else []:
            if seg.strip():
                chunks.append(f"{header}\n{label}: {seg.strip()}")
    return chunks or [header]


@dataclass
class ProdReplica(Searcher):
    chunk_owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    @classmethod
    def build(cls, year: str = "114", semester: str = "2", **_) -> "ProdReplica":
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer

        courses = load_courses(year=year, semester=semester)
        docs = [RetrievalDoc(c.course_id, content_text(c), _course_metadata(c)) for c in courses]
        model = SentenceTransformer(EMBED_MODEL, device="cuda")
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            model.tokenizer, chunk_size=320, chunk_overlap=40,
            separators=SEPARATORS, keep_separator=True)
        texts, owner = [], []
        for i, c in enumerate(courses):
            for t in build_chunks(splitter, c):
                texts.append(t)
                owner.append(i)
        key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
        path = CACHE / f"prod_chunks__{key}.npy"
        if path.exists():
            emb = np.load(path, allow_pickle=False)
        else:
            emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True)
            np.save(path, emb, allow_pickle=False)
        ids = [d.course_id for d in docs]
        return cls(courses, docs, BM25Retriever.from_docs(docs), model, emb, ids,
                   {c: i for i, c in enumerate(ids)}, np.asarray(owner))

    def dense_scores(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        """Course score = best chunk similarity (prod: MIN cosine distance per course)."""
        q = self.dense_model.encode(queries, batch_size=batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True)
        sims = q @ self.dense_emb.T  # (Q, n_chunks)
        out = np.full((len(queries), len(self.ids)), -np.inf, dtype=np.float32)
        np.maximum.at(out.T, self.chunk_owner, sims.T)
        return out


__all__ = ["ProdReplica", "rrf", "Ranked"]
