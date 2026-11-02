#!/usr/bin/env bash
# Phase 7: can a reranker run on CPU? Every variant runs the whole R set on CPU (12 threads)
# with the CPU preset's planner (gemma4:e4b plans, precomputed) and e5 on CPU; the GPU stays idle.
cd "$(dirname "$0")/../.."
LOG=logs/phase7.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
base="--struct --planner gemma4:e4b --encoder e5-large-instruct --merge --named-fix --device cpu --threads 12"
run() { stamp "run $1"; CUDA_VISIBLE_DEVICES= $PY -m scripts.v4.run_e2e run --name "$@" $base >> "$LOG" 2>&1; }

run cpu_norerank
run cpu_qwen06_l512_k10 --reranker qwen3-rerank-0.6b --rerank-maxlen 512 --rerank-k 10
run cpu_qwen06_l512_k10_int8 --reranker qwen3-rerank-0.6b --rerank-maxlen 512 --rerank-k 10 --rerank-int8
run cpu_bgebase_l512_k20 --reranker bge-reranker-base --rerank-maxlen 512 --rerank-k 20
run cpu_mminilm_l512_k20 --reranker mminilm-l12 --rerank-maxlen 512 --rerank-k 20
run cpu_bgem3_l512_k10 --reranker bge-reranker-v2-m3 --rerank-maxlen 512 --rerank-k 10
stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects data/v4/aspects/gemma4_12b.jsonl >> "$LOG" 2>&1
stamp "PHASE7 DONE"
