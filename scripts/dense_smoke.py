"""Smoke test: build dense index for 2795 courses, encode 50 queries.

Standalone script avoiding the run_experiment.py path that was hanging.
"""
# ruff: noqa: E402

import sys
import time

print("[smoke] importing torch / sentence_transformers...")
sys.stdout.flush()
import torch
from sentence_transformers import SentenceTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[smoke] device={device}")
sys.stdout.flush()

t = time.time()
model = SentenceTransformer("BAAI/bge-m3", device=device)
print(f"[smoke] model loaded in {time.time()-t:.1f}s")
sys.stdout.flush()

print("[smoke] importing project modules...")
sys.stdout.flush()
from src.loader import load_courses
from src.doc_builders import build_d_obj
from src.eval import from_objective

courses = load_courses(year="114", semester="2")
docs = build_d_obj(courses)
texts = [d.text for d in docs]
print(f"[smoke] {len(texts)} docs ready")
sys.stdout.flush()

t = time.time()
emb = model.encode(
    texts, batch_size=8, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=True,
)
print(f"[smoke] doc encode {time.time()-t:.1f}s, shape={emb.shape}")
sys.stdout.flush()

queries = from_objective(courses, n=50)
print(f"[smoke] {len(queries)} eval queries")
sys.stdout.flush()

import numpy as np
t = time.time()
hits_at_10 = 0
for q in queries:
    qv = model.encode([q.query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = emb @ qv
    top = np.argsort(-scores)[:10]
    if any(docs[i].course_id in q.gold for i in top):
        hits_at_10 += 1
print(f"[smoke] {len(queries)} queries in {time.time()-t:.1f}s, hit@10 = {hits_at_10}/{len(queries)}")
