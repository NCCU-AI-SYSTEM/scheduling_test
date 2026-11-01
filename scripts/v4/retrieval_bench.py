"""Component benchmark for retrieval: tokenizers, sparse, dense encoders, rerankers.

Isolates the topical retrieval step: T queries are searched as typed; M queries
use their topical need with the ORACLE conditions as a pre-filter, so condition
parsing errors do not leak into encoder scores. C queries (no topic) are skipped.

  python -m scripts.v4.retrieval_bench --stage sparse
  python -m scripts.v4.retrieval_bench --stage dense --models bge-m3,qwen3-emb-0.6b
  python -m scripts.v4.retrieval_bench --stage rerank --models bge-reranker-v2-m3,qwen3-rerank-0.6b

Runs are written in the v3 run format so scripts/rescore_v3.py can judge the
courses no pooled system surfaced before.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.doc_builders import BUILDERS
from src.eval.conditions import satisfies
from src.eval.v3_metrics import Scorer, load_eval
from src.loader import load_courses

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v4" / "retrieval"
K = 50


def topk(scores: np.ndarray, mask: np.ndarray | None, k: int = K) -> list[int]:
    s = scores.astype(np.float64, copy=True)
    if mask is not None:
        s[~mask] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])].tolist()


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    acc: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            acc[i] = acc.get(i, 0.0) + 1 / (k + rank + 1)
    return [i for i, _ in sorted(acc.items(), key=lambda x: -x[1])][:K]


class Bench:
    def __init__(self, doc: str):
        self.courses = load_courses()
        self.docs = BUILDERS[doc](self.courses)
        self.ids = [d.course_id for d in self.docs]
        self.texts = [d.text for d in self.docs]
        meta = {d.course_id: d.metadata for d in self.docs}
        qs, qrels = load_eval()
        self.queries = [q for q in qs if q.set in ("T", "M") and q.need]
        self.qtext = [q.query if q.set == "T" else q.need for q in self.queries]
        self.masks = []
        for q in self.queries:
            if q.set == "M" and q.has_cond:
                self.masks.append(np.array([satisfies(meta[c], q.cond) for c in self.ids]))
            else:
                self.masks.append(None)
        self.scorer = Scorer(qs, qrels, meta, {c.course_id: c.name for c in self.courses})
        self.qrels = qrels
        self.doc_name = doc

    def evaluate(self, name: str, ranked_idx: list[list[int]], extra: dict) -> dict:
        runs = {q.qid: [self.ids[i] for i in r] for q, r in zip(self.queries, ranked_idx)}
        out = {"name": name, "doc": self.doc_name} | extra
        for group in ("ALL", "T", "M"):
            sel = [q for q in self.queries if group == "ALL" or q.set == group]
            nd, rec, jud = [], [], []
            for q in sel:
                s = self.scorer.score(q.qid, runs[q.qid], 10)
                nd.append(s["ndcg"])
                jud.append(s["judged"])
                rel = {c for c in self.qrels.get(q.qid, {}) if (self.scorer.gain(q, c) or 0) >= 2}
                if rel:
                    rec.append(len(rel & set(runs[q.qid][:K])) / len(rel))
            out[group] = {"n": len(sel), "ndcg10": round(float(np.mean(nd)), 4),
                          "recall50": round(float(np.mean(rec)), 4) if rec else None,
                          "judged10": round(float(np.mean(jud)), 3)}
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / f"{name}.runs.jsonl").open("w", encoding="utf-8") as f:
            for qid, r in runs.items():
                f.write(json.dumps({"qid": qid, "ranked": r}, ensure_ascii=False) + "\n")
        with (OUT / "summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(json.dumps(out, ensure_ascii=False))
        return out


def stage_sparse(b: Bench, tokenizers: list[str]) -> None:
    from src.retrievers.sparse import BM25, BGEM3Sparse, load_user_dict, tokenize_all

    for tok in tokenizers:
        if tok == "bge-m3-sparse":
            t0 = time.time()
            m = BGEM3Sparse(b.texts)
            t_index = time.time() - t0
            t0 = time.time()
            scores = m.scores_many(b.qtext)
            per_q = (time.time() - t0) / len(b.qtext)
            ranked = [topk(scores[i], b.masks[i]) for i in range(len(b.qtext))]
        else:
            if tok == "jieba_dict":
                load_user_dict([c.name for c in b.courses] + [c.teacher for c in b.courses])
            t0 = time.time()
            bm = BM25(tokenize_all(b.texts, tok))
            t_index = time.time() - t0
            t0 = time.time()
            qtoks = tokenize_all(b.qtext, tok)
            ranked = [topk(bm.scores(qtoks[i]), b.masks[i]) for i in range(len(b.qtext))]
            per_q = (time.time() - t0) / len(b.qtext)
        b.evaluate(f"sparse__{tok}__{b.doc_name}", ranked,
                   {"stage": "sparse", "model": tok, "index_s": round(t_index, 1), "ms_per_q": round(per_q * 1000, 1)})


def stage_dense(b: Bench, models: list[str], cpu_latency: bool) -> None:
    import torch

    from src.retrievers.encoders import SPECS, Encoder

    for name in models:
        enc = Encoder(SPECS[name])
        demb, t_doc = enc.encode_docs(b.texts)
        qemb = enc.encode_queries(b.qtext)
        scores = qemb @ demb.T
        ranked = [topk(scores[i], b.masks[i]) for i in range(len(b.qtext))]
        extra = {"stage": "dense", "model": name, "doc_encode_s": round(t_doc, 1),
                 "gpu_ms_per_q": round(enc.query_latency(b.qtext) * 1000, 1),
                 "dim": int(demb.shape[1])}
        del enc
        torch.cuda.empty_cache()
        if cpu_latency:
            cpu = Encoder(SPECS[name], device="cpu")
            extra["cpu_ms_per_q"] = round(cpu.query_latency(b.qtext, n=25) * 1000, 1)
            del cpu
        b.evaluate(f"dense__{name}__{b.doc_name}", ranked, extra)


def stage_rerank(b: Bench, models: list[str], first: str) -> None:
    import torch

    from src.rerankers.models import load, timed_score

    cand = [json.loads(x)["ranked"] for x in (OUT / f"{first}.runs.jsonl").open(encoding="utf-8")]
    pos = {c: i for i, c in enumerate(b.ids)}
    for name in models:
        m = load(name)
        ranked, times = [], []
        for qi, c in enumerate(cand):
            idx = [pos[x] for x in c[:20]]
            s, dt = timed_score(m, b.qtext[qi], [b.texts[i] for i in idx])
            times.append(dt)
            ranked.append([i for _, i in sorted(zip(s, idx), key=lambda x: -x[0])] + [pos[x] for x in c[20:]])
        b.evaluate(f"rerank__{name}__on_{first}", ranked,
                   {"stage": "rerank", "model": name, "first_stage": first, "rerank_k": 20,
                    "ms_per_q": round(float(np.median(times)) * 1000, 1)})
        del m
        torch.cuda.empty_cache()


def stage_fuse(b: Bench, a: str, c: str) -> None:
    pos = {x: i for i, x in enumerate(b.ids)}
    ra = [[pos[x] for x in json.loads(l)["ranked"]] for l in (OUT / f"{a}.runs.jsonl").open(encoding="utf-8")]
    rc = [[pos[x] for x in json.loads(l)["ranked"]] for l in (OUT / f"{c}.runs.jsonl").open(encoding="utf-8")]
    b.evaluate(f"rrf__{a}__{c}", [rrf(x, y) for x, y in zip(ra, rc)], {"stage": "fuse", "model": f"{a}+{c}"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["sparse", "dense", "rerank", "fuse"])
    p.add_argument("--models", default="")
    p.add_argument("--doc", default="d-raw")
    p.add_argument("--first", default="")
    p.add_argument("--second", default="")
    p.add_argument("--cpu-latency", action="store_true")
    a = p.parse_args()
    b = Bench(a.doc)
    models = [m for m in a.models.split(",") if m]
    if a.stage == "sparse":
        stage_sparse(b, models or ["jieba", "jieba_dict", "char12", "ckip", "bge-m3-sparse"])
    elif a.stage == "dense":
        stage_dense(b, models, a.cpu_latency)
    elif a.stage == "rerank":
        stage_rerank(b, models, a.first)
    else:
        stage_fuse(b, a.first, a.second)


if __name__ == "__main__":
    main()
