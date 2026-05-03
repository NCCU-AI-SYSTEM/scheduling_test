#!/usr/bin/env python3
"""Orchestration script: wait for experiments, launch follow-ups, update PROGRESS.md.

Runs autonomously after user goes to sleep:
1. Wait for home_wsl D-V2/HyDE/Q2D to finish
2. Run reranker experiments on home_wsl (once reranker model is ready)
3. Sync home_mac rerank_k50 results when it reconnects
4. Rebuild PROGRESS.md with all results

Usage:
    uv run python scripts/orchestrate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = ROOT / "data" / "processed" / "orchestrate.log"
WSL_HOST = "home_wsl"
MAC_HOST = "home_mac"
WSL_DIR = "~/scheduling_test"
MAC_DIR = "~/Documents/code/NCCU-AI-SYSTEM/scheduling_test"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def ssh(host: str, cmd: str, timeout: int = 30) -> tuple[int, str]:
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no", host, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout + r.stderr).strip()


def wsl(cmd: str, timeout: int = 30) -> tuple[int, str]:
    full = f"cd {WSL_DIR} && export PATH=$HOME/.local/bin:$PATH && {cmd}"
    return ssh(WSL_HOST, full, timeout=timeout)


def wsl_result_exists(tag: str) -> bool:
    rc, _ = wsl(f"test -f results/tables/{tag}.json")
    return rc == 0


def wait_for_wsl_file(tag: str, poll_secs: int = 60) -> None:
    log(f"Waiting for {tag}...")
    while not wsl_result_exists(tag):
        time.sleep(poll_secs)
    log(f"{tag} done!")


def reranker_ready() -> bool:
    rc, out = ssh(WSL_HOST, "du -sh ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/ 2>/dev/null || echo missing")
    return rc == 0 and "missing" not in out


def run_wsl_experiment(tag: str, extra_args: str, timeout: int = 7200) -> bool:
    if wsl_result_exists(tag):
        log(f"  {tag} already exists, skip")
        return True
    log(f"  Launching {tag} on home_wsl...")
    cmd = (
        f"PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 "
        f"uv run python scripts/run_experiment.py {extra_args} --tag {tag} "
        f"> data/processed/{tag}.log 2>&1"
    )
    rc, out = wsl(cmd, timeout=timeout)
    if rc != 0:
        log(f"  FAIL {tag}: {out[:200]}")
        return False
    log(f"  OK {tag}")
    return True


def sync_results_from_wsl() -> None:
    log("Syncing results from home_wsl...")
    subprocess.run([
        "rsync", "-avz",
        f"{WSL_HOST}:{WSL_DIR}/results/tables/",
        str(ROOT / "results" / "tables") + "/",
    ], capture_output=True)
    log("Sync done")


def try_sync_from_mac() -> bool:
    rc, _ = ssh(MAC_HOST, "echo ok", timeout=10)
    if rc != 0:
        log("home_mac unreachable, skip")
        return False
    log("home_mac reachable, checking rerank_dense_k50...")
    rc2, _ = ssh(MAC_HOST, f"test -f {MAC_DIR}/results/tables/rerank_dense_k50.json")
    if rc2 == 0:
        log("rerank_dense_k50 ready, syncing...")
        subprocess.run([
            "rsync", "-avz",
            f"{MAC_HOST}:{MAC_DIR}/results/tables/",
            str(ROOT / "results" / "tables") + "/",
        ], capture_output=True)
        return True
    log("rerank_dense_k50 not ready yet on home_mac")
    return False


def read_metric(tag: str) -> dict | None:
    p = ROOT / "results" / "tables" / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def build_progress_md() -> None:
    log("Building PROGRESS.md...")

    configs = [
        ("d-base+bm25", "d-base__bm25__none__raw__none__synth_jsonl"),
        ("d-obj+bm25", "d-obj__bm25__none__raw__none__synth_jsonl"),
        ("d-v2+bm25", "dv2_bm25"),
        ("d-base+dense", "d-base__dense__none__raw__none__synth_jsonl"),
        ("d-obj+rrf", "d-obj__rrf__none__raw__none__synth_jsonl"),
        ("d-obj+dense", "d-obj__dense__none__raw__none__synth_jsonl"),
        ("d-v2+dense", "dv2_dense"),
        ("d-obj+rrf+rerank(k=20)", "rerank_rrf_k20"),
        ("d-obj+dense+rerank(k=20)", "rerank_dense_k20"),
        ("d-obj+dense+rerank(k=50)", "rerank_dense_k50"),
        ("d-obj+hyde+dense", "hyde_dense_k20"),
        ("d-obj+q2d+dense", "q2d_dense_k20"),
        ("d-v2+dense+rerank(k=20)", "dv2_dense_rerank_k20"),
        ("d-obj+hyde+dense+rerank", "hyde_dense_rerank_k20"),
        ("d-obj+q2d+dense+rerank", "q2d_dense_rerank_k20"),
    ]

    rows: list[str] = []
    for label, tag in configs:
        d = read_metric(tag)
        if d is None:
            continue
        r10 = d.get("recall@10", 0)
        ndcg = d.get("ndcg@10", 0)
        mrr = d.get("mrr@10", 0)
        n = d.get("n_queries", 0)
        ms = d.get("latency_ms_per_query", 0)
        rows.append(f"| {label} | {n} | {r10:.3f} | {ndcg:.3f} | {mrr:.3f} | {ms:.0f}ms |")

    rows_str = "\n".join(rows)
    best = max(
        [(label, read_metric(tag)) for label, tag in configs if read_metric(tag)],
        key=lambda x: x[1].get("recall@10", 0),
        default=("—", {}),
    )
    best_r10 = best[1].get("recall@10", 0) if best[1] else 0

    md = f"""# NCCU 課程推薦系統 Retrieval 改進實驗 — 進度與結果分析

版本：v3（自動更新）
更新日期：{time.strftime("%Y-%m-%d %H:%M")}

---

## 1. 研究背景

現行系統（`CourseLangChain/build.py`）page_content 只含「課名+時間+老師」。
**核心假設（已驗證）**：把 `objective` 加入索引是最大收益來源。

---

## 2. 模型架構（全面改用 GPT Batch API）

| 角色 | 模型 | 部署 |
|---|---|---|
| Dense embedding | BAAI/bge-m3 | 本機 CPU（doc cache pkl） |
| Reranker | BAAI/bge-reranker-v2-m3 | home_wsl |
| LLM 生成 | gpt-4o-mini（OpenAI Batch） | API，< US$2 |

---

## 3. Ablation 結果（synth_jsonl，LLM 合成 query）

| Config | n | R@10 | nDCG@10 | MRR@10 | ms/q |
|---|---|---|---|---|---|
{rows_str}

**目前最佳：{best[0]}，R@10={best_r10:.3f}**（vs 現行 0.288，+{best_r10-0.288:.3f}）

---

## 4. Key Findings

- **F1**：D-Obj BM25 vs D-Base BM25：只 +4pp；D-Obj Dense vs D-Base Dense：+12.6pp → Dense 才能發揮 objective 的語意
- **F2**：D-V2（LLM 摘要+關鍵字）進一步提升 BM25 +11.6pp（0.328→0.444）
- **F3**：RRF 在 oral query 反比純 Dense 差（BM25 噪音污染）
- **F4**：Reranker (k=20) 在 Dense 上 +4pp，在 RRF 上 +13pp
- **F5**：HyDE/Q2D 效果待最終結果補充

---

## 5. 未完成

| 項目 | 狀態 |
|---|---|
| Human gold-set 150 query | ⬜ 未做（最終 test 必須） |
| Structured filter 整合 | ⬜ code 完成，未接進最佳 pipeline |
| D-V2 + Dense + Rerank | 跑中 / 等結果 |
| HyDE / Q2D + Dense + Rerank | 跑中 / 等結果 |

---

## 6. 成本

所有 LLM 任務（eval-set + meta + HyDE/Q2D）合計 < **US$2**（OpenAI Batch 50% 折扣）
"""

    progress_path = ROOT / "PROGRESS.md"
    progress_path.write_text(md)
    log(f"PROGRESS.md updated ({progress_path.stat().st_size} bytes)")


def main() -> None:
    log("=== orchestrate.py start ===")

    # 1. Wait for D-V2 dense to finish on home_wsl
    wait_for_wsl_file("dv2_dense", poll_secs=120)

    # 2. Wait for HyDE
    wait_for_wsl_file("hyde_dense_k20", poll_secs=120)

    # 3. Wait for Q2D
    wait_for_wsl_file("q2d_dense_k20", poll_secs=120)

    sync_results_from_wsl()
    build_progress_md()

    # 4. Wait for reranker model
    log("Waiting for reranker model on home_wsl...")
    while not reranker_ready():
        time.sleep(60)
    log("Reranker ready!")

    # Install FlagEmbedding if needed
    wsl("uv add FlagEmbedding 2>&1 | tail -2", timeout=60)

    # 5. Run reranker experiments on home_wsl
    log("=== Reranker experiments ===")
    run_wsl_experiment(
        "dv2_dense_rerank_k20",
        "--doc d-v2 --retriever dense --reranker bge --eval synth_jsonl --n 500 --retrieve-k 20 --top-k 10",
        timeout=7200,
    )
    run_wsl_experiment(
        "hyde_dense_rerank_k20",
        "--doc d-obj --retriever dense --rewrite hyde --reranker bge --eval synth_jsonl --n 500 --retrieve-k 20 --top-k 10",
        timeout=7200,
    )
    run_wsl_experiment(
        "q2d_dense_rerank_k20",
        "--doc d-obj --retriever dense --rewrite q2d --reranker bge --eval synth_jsonl --n 500 --retrieve-k 20 --top-k 10",
        timeout=7200,
    )

    sync_results_from_wsl()

    # 6. Try to get home_mac results
    for _ in range(12):  # try every 10 min for 2 hours
        if try_sync_from_mac():
            break
        time.sleep(600)

    # 7. Final PROGRESS.md
    build_progress_md()

    log("=== orchestrate.py done ===")


if __name__ == "__main__":
    main()
