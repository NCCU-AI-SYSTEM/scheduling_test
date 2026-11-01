"""CPU-only cost of encoders and rerankers: can the search run on a machine without a GPU?

hopelab's CPU is a 12-core Ryzen 9 9900X, so each model is timed with 12 threads and
with 4 threads (a typical laptop). Each (model, threads) runs in its own subprocess so
peak resident memory belongs to that model alone.

Measured per model:
  encoder   query latency p50 (one query, batch 1), document indexing time for the
            2,655 courses extrapolated from a 200-document sample, peak RSS
  reranker  latency p50 to score the 20-course head for one query, peak RSS

  python -m scripts.v4.cpu_bench                      # all, writes results/v4/cpu_bench.json
  python -m scripts.v4.cpu_bench --one encoder e5-large-instruct 4   # child process
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v4" / "cpu_bench.json"
ENCODERS = ["bge-base-zh", "e5-large-instruct", "bge-m3", "qwen3-emb-0.6b"]
RERANKERS = ["bge-reranker-v2-m3", "qwen3-rerank-0.6b"]
THREADS = [12, 4]
N_DOCS = 2655


def _texts() -> tuple[list[str], list[str]]:
    from scripts.v4.run_e2e import load_r
    from src.v4.catalog import load_catalog
    from src.v4.pipeline import doc_text

    docs = [doc_text(c) for c in load_catalog(with_reviews=False)]
    queries = [r["query"] for r in load_r()]
    return docs, queries


def _p50(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def child(kind: str, name: str, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    docs, queries = _texts()
    t0 = time.time()
    if kind == "encoder":
        from src.retrievers.encoders import SPECS, Encoder

        enc = Encoder(SPECS[name], device="cpu")
        load_s = time.time() - t0
        q_ms = enc.query_latency(queries, n=33) * 1000
        sample = docs[:200]
        t1 = time.time()
        enc.model.encode([enc.spec.doc_prefix + d for d in sample], batch_size=16, show_progress_bar=False)
        index_min = (time.time() - t1) / len(sample) * N_DOCS / 60
        res = {"query_ms_p50": round(q_ms, 1), "index_min_est": round(index_min, 1)}
    else:
        from src.rerankers.models import load

        rr = load(name, device="cpu")
        load_s = time.time() - t0
        times = []
        for i, q in enumerate(queries[:13]):
            head = docs[i * 20:(i + 1) * 20]
            t1 = time.time()
            rr.score(q, head)
            times.append(time.time() - t1)
        res = {"rerank20_ms_p50": round(_p50(times[3:]) * 1000, 1)}
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KiB
    return {"kind": kind, "model": name, "threads": threads, "load_s": round(load_s, 1),
            "peak_rss_mb": round(rss_mb), **res}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--one", nargs=3, metavar=("KIND", "NAME", "THREADS"))
    a = p.parse_args()
    if a.one:
        print("RESULT " + json.dumps(child(a.one[0], a.one[1], int(a.one[2])), ensure_ascii=False))
        return
    rows = []
    jobs = [("encoder", m) for m in ENCODERS] + [("reranker", m) for m in RERANKERS]
    for kind, name in jobs:
        for t in THREADS:
            env_cmd = [sys.executable, "-m", "scripts.v4.cpu_bench", "--one", kind, name, str(t)]
            proc = subprocess.run(env_cmd, capture_output=True, text=True, cwd=ROOT,
                                  env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                       "HOME": str(Path.home()), "OMP_NUM_THREADS": str(t)})
            line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None)
            if line is None:
                print(f"FAILED {kind} {name} {t}: {proc.stderr.strip().splitlines()[-1:]}", flush=True)
                continue
            row = json.loads(line[7:])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("CPU BENCH DONE")


if __name__ == "__main__":
    main()
