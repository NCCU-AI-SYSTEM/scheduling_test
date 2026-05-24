"""
Phase 5：pre-filter + retrieval 組合實驗（batch 版）。

比較以下四種設定在 eval_conditions_v1.jsonl 上的 R@10：
  no_filter  : E7 baseline（BM25 expand + RRF + Rerank，無 filter）
  oracle     : gold SQL conditions → pre-filter → 同 pipeline
  p1         : Regex parser → pre-filter → 同 pipeline
  p3         : LLM few-shot parser → pre-filter → 同 pipeline

批次設計（避免 per-query encode bottleneck）：
  Step 1. 所有 query 一次 dense encode（~30s for 2160q）
  Step 2. 全庫 BM25 scores（per-query 快，rank_bm25 的 get_scores）
  Step 3. pre-filter → 各 query 分別取自己的 allowed_ids 做 argsort
  Step 4. RRF fusion per-query
  Step 5. 全部 hits 一次 batch_rerank
  → 整體 2160q 預估 ~30-40 分鐘（vs 序列 18 小時）

結果存：results/tables/retrieval_filter_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.filter import conditions_to_sql
from src.db.pg_client import get_connection
from src.parsers import regex_v1
from src.parsers.base import ConditionResult
from src.query_rewriters import parse_constraints

EVAL_PATH   = Path("data/raw/eval_conditions_v1.jsonl")
RESULTS_DIR = Path("results/tables")

WEEKDAY_ZH = ["", "一", "二", "三", "四", "五", "六", "日"]


def _expand_for_bm25(query: str) -> str:
    """BM25 constraint expand（與 run_experiment.py 相同邏輯）。"""
    c = parse_constraints(query)
    parts: list[str] = []
    if len(c.weekday_include) == 1:
        wd = next(iter(c.weekday_include))
        parts.append(f"星期{WEEKDAY_ZH[wd]}")
    if c.hour_min is not None and c.hour_max is not None:
        mid = (int(c.hour_min) + int(c.hour_max)) // 2
        seen: set[int] = set()
        for h in (int(c.hour_min), mid, int(c.hour_max)):
            if h not in seen:
                parts.append(f"{h:02d}:00")
                seen.add(h)
    elif c.hour_min is not None:
        parts.append(f"{int(c.hour_min):02d}:00")
    for lang in sorted(c.lang_include):
        parts.append(lang)
    for unit in sorted(c.unit_include):
        parts.append(f"{unit}系")
    return f"{query} {' '.join(parts)}" if parts else query


# ---------- SQL pre-filter ----------

def _get_pg_ids(conditions: ConditionResult) -> set[str] | None:
    """
    回傳滿足 conditions 的 course_id set。
    若無條件（空）回傳 None（代表不過濾）。
    """
    must_empty     = all(not v for v in conditions.must.values())
    must_not_empty = all(not v for v in conditions.must_not.values())
    if must_empty and must_not_empty:
        return None
    try:
        sql, params = conditions_to_sql(conditions.must, conditions.must_not)
    except Exception as e:
        log.warning(f"conditions_to_sql 失敗: {e}")
        return None
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute(sql, params)
        ids = {r[0] for r in cur.fetchall()}
        conn.close()
        return ids
    except Exception as e:
        log.warning(f"SQL filter 查詢失敗: {e}")
        return None


def _row_to_conditions(row: dict) -> ConditionResult:
    return ConditionResult(
        must=row.get("must", {}),
        must_not=row.get("must_not", {}),
    )


# ---------- 批次 filter：收集每筆 allowed_ids ----------

def build_filter_sets(rows: list[dict], variant: str, p3_fn=None) -> list[set[str] | None]:
    """
    回傳 list[set | None]，長度 = len(rows)。
    None = 不過濾（no_filter）。
    """
    allowed_list: list[set[str] | None] = []
    for i, row in enumerate(rows):
        if variant == "no_filter":
            allowed_list.append(None)
        elif variant == "oracle":
            cond = _row_to_conditions(row)
            allowed_list.append(_get_pg_ids(cond))
        elif variant == "p1":
            cond = regex_v1.parse(row["queries"]["zh"])
            allowed_list.append(_get_pg_ids(cond))
        elif variant == "p3" and p3_fn:
            cond = p3_fn(row["queries"]["zh"])
            allowed_list.append(_get_pg_ids(cond))
        else:
            allowed_list.append(None)
        if (i + 1) % 500 == 0:
            log.info(f"  filter sets: {i+1}/{len(rows)}")
    return allowed_list


# ---------- 批次 Retrieval ----------

def batch_retrieve_and_rerank(
    rows: list[dict],
    allowed_list: list[set[str] | None],
    dense_retriever,
    bm25_retriever,
    top_k: int = 10,
    retrieve_k: int = 50,
    rrf_k: int = 60,
) -> list[list[str]]:
    """
    批次做 BM25 expand + Dense + RRF + Rerank，回傳每筆的 top_k course_id list。

    dense_retriever.embeddings: (N_docs, D) numpy array
    dense_retriever.model:      SentenceTransformer
    """
    from src.retrievers import rrf_fuse
    from src.rerankers import batch_rerank

    n = len(rows)
    docs = dense_retriever.docs  # list[RetrievalDoc]，與 embeddings row 對應
    N_docs = len(docs)
    doc_ids = [d.course_id for d in docs]
    doc_id_to_idx = {cid: i for i, cid in enumerate(doc_ids)}

    # --- Step 1: batch dense encode ---
    log.info(f"  [dense] encoding {n} queries ...")
    queries_raw      = [row["queries"]["zh"] for row in rows]
    queries_expanded = [_expand_for_bm25(q) for q in queries_raw]

    t0 = time.time()
    q_embs = dense_retriever.model.encode(
        queries_raw,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )  # (n, D)
    log.info(f"  [dense] encoded in {time.time()-t0:.1f}s")

    # --- Step 2: matrix multiply → all dense scores ---
    log.info("  [dense] matrix multiply ...")
    t0 = time.time()
    all_dense_scores = q_embs @ dense_retriever.embeddings.T  # (n, N_docs)
    log.info(f"  [dense] matmul done in {time.time()-t0:.1f}s")

    # --- Step 3: BM25 scores (per-query，但很快) ---
    log.info("  [bm25] computing scores ...")
    from src.retrievers.bm25 import tokenize
    t0 = time.time()
    all_bm25_scores = np.zeros((n, N_docs), dtype=np.float32)
    for i, q_exp in enumerate(queries_expanded):
        toks = tokenize(q_exp)
        if toks:
            s = bm25_retriever.bm25.get_scores(toks)
            all_bm25_scores[i] = s
    log.info(f"  [bm25] done in {time.time()-t0:.1f}s")

    # --- Step 4: per-query RRF + pre-filter → collect top-retrieve_k hits ---
    log.info(f"  [rrf+filter] fusing (retrieve_k={retrieve_k}) ...")
    all_hits: list[list[tuple]] = []  # list of list[tuple(RetrievalDoc, float)]

    for i in range(n):
        allowed = allowed_list[i]

        # Dense top-retrieve_k (with filter)
        d_scores = all_dense_scores[i]
        if allowed is not None:
            mask = np.array([doc_ids[j] in allowed for j in range(N_docs)], dtype=bool)
            d_scores_f = np.where(mask, d_scores, -np.inf)
        else:
            mask = None
            d_scores_f = d_scores
        d_idx = np.argsort(-d_scores_f)[:retrieve_k]
        dense_hits = [(docs[j], float(d_scores[j])) for j in d_idx if d_scores_f[j] > -np.inf]

        # BM25 top-retrieve_k (with filter)
        b_scores = all_bm25_scores[i]
        if mask is not None:
            b_scores_f = np.where(mask, b_scores, -np.inf)
        else:
            b_scores_f = b_scores
        b_idx = np.argsort(-b_scores_f)[:retrieve_k]
        bm25_hits = [(docs[j], float(b_scores[j])) for j in b_idx if b_scores_f[j] > 0]

        # RRF
        merged = rrf_fuse([dense_hits, bm25_hits], top_k=retrieve_k)
        all_hits.append(merged)

    log.info("  [rrf+filter] done")

    # --- Step 5: batch rerank ---
    log.info(f"  [rerank] batch reranking {n} queries ...")
    t0 = time.time()
    reranked = batch_rerank(queries_raw, all_hits, top_k=top_k)
    log.info(f"  [rerank] done in {time.time()-t0:.1f}s")

    return [[doc.course_id for doc, _ in hits] for hits in reranked]


# ---------- Eval ----------

def run_variant(
    rows: list[dict],
    variant: str,
    dense_retriever,
    bm25_retriever,
    p3_fn=None,
    top_k: int = 10,
    retrieve_k: int = 50,
) -> dict:
    log.info(f"[{variant}] building filter sets ...")
    t0 = time.time()
    allowed_list = build_filter_sets(rows, variant, p3_fn)
    t_filter = time.time() - t0

    pool_sizes = [
        len(a) if a is not None else len(dense_retriever.docs)
        for a in allowed_list
    ]
    avg_pool = sum(pool_sizes) / len(pool_sizes)
    log.info(f"  [{variant}] avg_pool={avg_pool:.0f}  filter_time={t_filter:.1f}s")

    log.info(f"[{variant}] batch retrieval ...")
    t1 = time.time()
    retrieved_ids = batch_retrieve_and_rerank(
        rows, allowed_list, dense_retriever, bm25_retriever,
        top_k=top_k, retrieve_k=retrieve_k,
    )
    t_retrieval = time.time() - t1

    hits = sum(
        int(row["gold_course_id"] in ids)
        for row, ids in zip(rows, retrieved_ids)
    )
    r_at_k = hits / len(rows)
    elapsed = time.time() - t0

    summary = {
        "variant":      variant,
        "n":            len(rows),
        "r_at_10":      round(r_at_k, 4),
        "hits":         hits,
        "avg_pool":     round(avg_pool, 1),
        "elapsed_s":    round(elapsed, 1),
        "filter_s":     round(t_filter, 1),
        "retrieval_s":  round(t_retrieval, 1),
        "ms_per_q":     round(elapsed / len(rows) * 1000, 1),
    }
    log.info(
        f"  [{variant}] R@{top_k}={r_at_k:.4f}  hits={hits}/{len(rows)}"
        f"  avg_pool={avg_pool:.0f}  elapsed={elapsed:.1f}s"
    )
    return summary


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="no_filter,oracle,p1",
                   help="comma-separated: no_filter,oracle,p1,p3")
    p.add_argument("--p3-model", default="gpt-4.1-mini",
                   help="LLM model for p3 variant")
    p.add_argument("--limit",   type=int, default=0)
    p.add_argument("--negation-only", action="store_true",
                   help="只評估 has_negation=True 的子集")
    p.add_argument("--eval",    default=str(EVAL_PATH))
    p.add_argument("--top-k",      type=int, default=10)
    p.add_argument("--retrieve-k", type=int, default=20,
                   help="RRF 前各 retriever 取幾筆（rerank candidates，預設 20）")
    args = p.parse_args()

    # 載入 eval data
    rows: list[dict] = []
    for line in Path(args.eval).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    if args.negation_only:
        rows = [r for r in rows if r.get("has_negation")]
        log.info(f"negation-only subset: {len(rows)} rows")
    if args.limit:
        rows = rows[: args.limit]
    log.info(f"Total eval rows: {len(rows)}")

    # 初始化 retrievers（只做一次）
    log.info("Initialising retrievers ...")
    from src.loader.courses import load_courses
    from src.doc_builders import BUILDERS
    from src.retrievers import BM25Retriever, DenseRetriever

    courses = load_courses(year="114", semester="2")
    docs    = BUILDERS["d-v2"](courses)
    bm25    = BM25Retriever.from_docs(docs, k=50)
    dense   = DenseRetriever.from_docs(docs, k=50)
    log.info("Retrievers ready.")

    # p3 parser（lazy）
    p3_fn = None
    if "p3" in args.variants.split(","):
        import functools
        from src.parsers import llm_fewshot
        p3_fn = functools.partial(llm_fewshot.parse, model=args.p3_model)
        log.info(f"p3 parser using model: {args.p3_model}")

    variants = [v.strip() for v in args.variants.split(",")]
    all_summaries = []

    for variant in variants:
        summary = run_variant(
            rows,
            variant=variant,
            dense_retriever=dense,
            bm25_retriever=bm25,
            p3_fn=p3_fn if variant == "p3" else None,
            top_k=args.top_k,
            retrieve_k=args.retrieve_k,
        )
        all_summaries.append(summary)

    # 比較表
    print()
    print("=" * 68)
    print("Phase 5 Pre-filter + Retrieval 結果")
    print("=" * 68)
    print(f"{'Variant':<20} {'R@10':>8} {'avg_pool':>10} {'elapsed':>10}")
    print("-" * 68)
    for s in all_summaries:
        print(f"{s['variant']:<20} {s['r_at_10']:>8.4f} {s['avg_pool']:>10.0f} {s['elapsed_s']:>9.1f}s")

    # 儲存
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "retrieval_filter_result.json"
    out_path.write_text(json.dumps(
        {"config": vars(args), "summaries": all_summaries},
        indent=2, ensure_ascii=False,
    ))
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
