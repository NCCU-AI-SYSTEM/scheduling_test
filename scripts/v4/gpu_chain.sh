#!/usr/bin/env bash
# GPU work queue, run after the LLM benchmark (torch jobs first, then Ollama jobs).
cd "$(dirname "$0")/../.."
LOG=logs/gpu_chain.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
unload() { for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done; }
until grep -q "STRICT DONE" logs/llm_bench.log 2>/dev/null; do sleep 60; done
unload

stamp "install ckip"
uv pip install --python $PY ckip-transformers >> "$LOG" 2>&1
stamp "sparse ckip + bge-m3-sparse"
$PY -m scripts.v4.retrieval_bench --stage sparse --models ckip,bge-m3-sparse >> "$LOG" 2>&1
for m in bge-m3 bge-base-zh qwen3-emb-0.6b e5-large-instruct gte-multi-base; do
  stamp "dense $m"
  $PY -m scripts.v4.retrieval_bench --stage dense --models $m --cpu-latency >> "$LOG" 2>&1
done
stamp "dense qwen3-emb-4b"
$PY -m scripts.v4.retrieval_bench --stage dense --models qwen3-emb-4b >> "$LOG" 2>&1
stamp "fuse"
$PY -m scripts.v4.retrieval_bench --stage fuse --first sparse__char12__d-raw --second dense__bge-m3__d-raw >> "$LOG" 2>&1
$PY -m scripts.v4.retrieval_bench --stage fuse --first sparse__jieba__d-raw --second dense__bge-m3__d-raw >> "$LOG" 2>&1
for r in bge-reranker-v2-m3 qwen3-rerank-0.6b qwen3-rerank-4b; do
  stamp "rerank $r"
  $PY -m scripts.v4.retrieval_bench --stage rerank --models $r --first rrf__sparse__char12__d-raw__dense__bge-m3__d-raw >> "$LOG" 2>&1
done

stamp "hype gemma4:e4b"
$PY -m scripts.v4.gen_hype --model gemma4:e4b --workers 2 >> "$LOG" 2>&1
unload
stamp "aspects gemma4:12b"
$PY -m scripts.v4.extract_aspects --model gemma4:12b --workers 2 >> "$LOG" 2>&1
unload
stamp "judge retrieval runs (qwen3:14b)"
$PY -m scripts.rescore_v3 --runs-dir results/v4/retrieval --workers 4 >> "$LOG" 2>&1
unload
stamp "GPU CHAIN DONE"
