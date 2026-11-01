"""BM25 with interchangeable Chinese tokenizers, plus bge-m3 learned sparse.

Tokenizers compared:
  jieba        default dictionary (trained on Simplified Chinese)
  jieba_dict   jieba + a user dictionary of course names and teacher names from the
               catalogue, so 社會語言學 / 資料結構 stay whole
  char12       character unigrams + bigrams (no dictionary at all)
  ckip         CKIP Transformers word segmentation (Traditional Chinese, BERT-based)
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[\w一-鿿]+")
_HAN = re.compile(r"[一-鿿]")


def _jieba_cut(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.lcut(text) if _WORD.fullmatch(t)]


def _char12(text: str) -> list[str]:
    out: list[str] = []
    for run in re.findall(r"[一-鿿]+|[A-Za-z0-9]+", text):
        if _HAN.match(run):
            out.extend(run)
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            out.append(run.lower())
    return out


@lru_cache(maxsize=1)
def _ckip():
    from ckip_transformers.nlp import CkipWordSegmenter

    return CkipWordSegmenter(model="bert-base", device=0)


def load_user_dict(words: list[str]) -> None:
    import jieba

    for w in sorted(set(words)):
        w = w.strip()
        if 2 <= len(w) <= 12 and _HAN.search(w):
            jieba.add_word(w, freq=50000)


def tokenize_all(texts: list[str], tokenizer: str) -> list[list[str]]:
    if tokenizer in ("jieba", "jieba_dict"):
        return [_jieba_cut(t) for t in texts]
    if tokenizer == "char12":
        return [_char12(t) for t in texts]
    if tokenizer == "ckip":
        ws = _ckip()(texts, batch_size=32, max_length=500, show_progress=False)
        return [[w for w in toks if _WORD.fullmatch(w)] for toks in ws]
    raise ValueError(tokenizer)


class BM25:
    def __init__(self, doc_tokens: list[list[str]]):
        self.bm25 = BM25Okapi(doc_tokens)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        if not query_tokens:
            return np.zeros(self.bm25.corpus_size)
        return np.asarray(self.bm25.get_scores(query_tokens))


class BGEM3Sparse:
    """Lexical weights from bge-m3; score = sum over shared tokens of q_w * d_w."""

    def __init__(self, docs: list[str]):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        out = self.model.encode(docs, batch_size=16, max_length=1024, return_dense=False,
                                return_sparse=True, return_colbert_vecs=False)
        self.doc_w = out["lexical_weights"]

    def scores_many(self, queries: list[str]) -> np.ndarray:
        q = self.model.encode(queries, batch_size=64, max_length=128, return_dense=False,
                              return_sparse=True, return_colbert_vecs=False)["lexical_weights"]
        out = np.zeros((len(queries), len(self.doc_w)), dtype=np.float32)
        for i, qw in enumerate(q):
            for j, dw in enumerate(self.doc_w):
                s = 0.0
                for tok, w in qw.items():
                    v = dw.get(tok)
                    if v is not None:
                        s += w * v
                out[i, j] = s
        return out
