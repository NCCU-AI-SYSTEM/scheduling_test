#!/usr/bin/env python3
"""Orchestration script: launch remaining reranker experiments on home_wsl,
then sync results and update PROGRESS.md.

Experiments are launched via nohup (non-blocking) then polled every 2 min.
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
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def wsl_result_exists(tag: str) -> bool:
    rc, _ = ssh(WSL_HOST, f"test -f {WSL_DIR}/results/tables/{tag}.json")
    return rc == 0


def wsl_process_running(tag: str) -> bool:
    rc, out = ssh(WSL_HOST, f"pgrep -f 'tag {tag}' 2>/dev/null | head -1")
    return rc == 0 and out.strip() != ""


def launch_wsl_nohup(tag: str, extra_args: str) -> None:
    """Launch experiment on home_wsl via nohup (non-blocking)."""
    cmd = (
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"cd {WSL_DIR} && "
        f"PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 "
        f"nohup uv run python scripts/run_experiment.py {extra_args} --tag {tag} "
        f"> data/processed/{tag}.log 2>&1 &"
    )
    rc, out = ssh(WSL_HOST, cmd, timeout=20)
    if rc == 0:
        log(f"  Launched {tag} on home_wsl")
    else:
        log(f"  FAIL launching {tag}: {out[:100]}")


def wait_for_wsl_result(tag: str, poll_secs: int = 120) -> None:
    """Poll until result exists or process stops."""
    log(f"Waiting for {tag}...")
    fails = 0
    while True:
        if wsl_result_exists(tag):
            log(f"  {tag} done!")
            return
        if fails > 10 and not wsl_process_running(tag):
            log(f"  WARNING: {tag} process not found, may have crashed")
            break
        time.sleep(poll_secs)
        fails += 1


def sync_results_from_wsl() -> None:
    subprocess.run([
        "rsync", "-avz", "--quiet",
        f"{WSL_HOST}:{WSL_DIR}/results/tables/",
        str(ROOT / "results" / "tables") + "/",
    ], capture_output=True)


def try_sync_from_mac() -> bool:
    rc, _ = ssh(MAC_HOST, "echo ok", timeout=10)
    if rc != 0:
        return False
    rc2, _ = ssh(MAC_HOST, f"test -f {MAC_DIR}/results/tables/rerank_dense_k50.json")
    if rc2 == 0:
        subprocess.run([
            "rsync", "-avz", "--quiet",
            f"{MAC_HOST}:{MAC_DIR}/results/tables/",
            str(ROOT / "results" / "tables") + "/",
        ], capture_output=True)
        log("Synced rerank_dense_k50 from home_mac")
        return True
    return False


def read_metric(tag: str) -> dict | None:
    p = ROOT / "results" / "tables" / f"{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def update_progress_md() -> None:
    configs = [
        ("d-base+BM25（現行）", "d-base__bm25__raw__none__synth_jsonl"),
        ("d-obj+BM25", "d-obj__bm25__raw__none__synth_jsonl"),
        ("d-v2+BM25", "dv2_bm25"),
        ("d-base+Dense", "d-base__dense__raw__none__synth_jsonl"),
        ("d-obj+RRF", "d-obj__rrf__raw__none__synth_jsonl"),
        ("d-obj+Dense+Q2D", "q2d_dense_k20"),
        ("d-obj+Dense+HyDE", "hyde_dense_k20"),
        ("d-obj+Dense", "d-obj__dense__raw__none__synth_jsonl"),
        ("d-v2+Dense", "dv2_dense"),
        ("d-obj+RRF+Rerank(k=20)", "rerank_rrf_k20"),
        ("d-obj+Dense+Rerank(k=50)", "rerank_dense_k50"),
        ("d-obj+Dense+Rerank(k=20)", "rerank_dense_k20"),
        ("d-v2+Dense+Rerank(k=20)", "dv2_dense_rerank_k20"),
        ("d-obj+HyDE+Dense+Rerank(k=20)", "hyde_dense_rerank_k20"),
        ("d-obj+Q2D+Dense+Rerank(k=20)", "q2d_dense_rerank_k20"),
    ]

    rows = []
    best_r10, best_label = 0.0, "—"
    for label, tag in configs:
        d = read_metric(tag)
        if d is None:
            rows.append(f"| {label} | — | — | — | — | — |")
            continue
        r10 = d.get("recall@10", 0)
        ndcg = d.get("ndcg@10", 0)
        mrr = d.get("mrr@10", 0)
        n = d.get("n_queries", 0)
        ms = d.get("latency_ms_per_query", 0)
        rows.append(f"| {label} | {n} | {r10:.3f} | {ndcg:.3f} | {mrr:.3f} | {ms:.0f}ms |")
        if r10 > best_r10:
            best_r10, best_label = r10, label

    rows_md = "\n".join(rows)
    delta = best_r10 - 0.288

    p = ROOT / "PROGRESS.md"
    content = p.read_text()
    # Replace table section
    start = content.find("| Config | n | R@10")
    end = content.find("\n---", start)
    if start > 0 and end > 0:
        new_table = f"| Config | n | R@10 | nDCG@10 | MRR@10 | ms/q |\n|---|---|---|---|---|---|\n{rows_md}"
        content = content[:start] + new_table + content[end:]
        content = content.replace(
            content[content.find("**目前最佳"):content.find("\n\n---", content.find("**目前最佳"))],
            f"**目前最佳：{best_label}，R@10={best_r10:.3f}**（vs 現行 0.288，+{delta:.3f}）"
        )
    # update date
    import re
    content = re.sub(r"更新日期：.*", f"更新日期：{time.strftime('%Y-%m-%d %H:%M')}", content)
    p.write_text(content)
    log(f"PROGRESS.md updated. Best: {best_label} R@10={best_r10:.3f}")


def main() -> None:
    log("=== orchestrate.py (v2, nohup mode) start ===")

    # Remaining experiments to run
    experiments = [
        ("dv2_dense_rerank_k20",
         "--doc d-v2 --retriever dense --reranker bge --eval synth_jsonl --n 500 --retrieve-k 20 --top-k 10"),
        ("hyde_dense_rerank_k20",
         "--doc d-obj --retriever dense --rewrite hyde --reranker bge --eval synth_jsonl --n 500 --retrieve-k 20 --top-k 10"),
        ("q2d_dense_rerank_k20",
         "--doc d-obj --retriever dense --rewrite q2d --reranker bge --eval synth_jsonl --n 500 --retrieve-k 20 --top-k 10"),
    ]

    for tag, args in experiments:
        if wsl_result_exists(tag):
            log(f"  {tag} already done, skip")
            continue
        # If not running, launch
        if not wsl_process_running(tag):
            launch_wsl_nohup(tag, args)
            time.sleep(10)
        wait_for_wsl_result(tag, poll_secs=120)
        sync_results_from_wsl()
        update_progress_md()

    # Final sync from home_mac
    for _ in range(24):
        if try_sync_from_mac():
            break
        time.sleep(600)

    sync_results_from_wsl()
    update_progress_md()
    log("=== orchestrate.py done ===")


if __name__ == "__main__":
    main()
