#!/usr/bin/env bash
# After phase 2: HyPE question generation (Ollama keeps one model loaded at a time on
# this host, so it cannot overlap other Ollama jobs), then rerun the sparse runs that
# crashed (CKIP, bge-m3 sparse), then phase 3.
cd "$(dirname "$0")/../../.."
LOG=logs/phase25.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
until grep -q "PHASE2 DONE" logs/phase2.log 2>/dev/null; do sleep 60; done
stamp "hype gemma4:e4b"
$PY -m scripts.v4.analysis.gen_hype --model gemma4:e4b --workers 2 >> "$LOG" 2>&1
for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done
stamp "sparse ckip + bge-m3-sparse"
$PY -m scripts.v4.analysis.retrieval_bench --stage sparse --models ckip,bge-m3-sparse >> "$LOG" 2>&1
stamp "PHASE25 DONE"
ENC=bge-m3 RR=qwen3-rerank-0.6b bash scripts/v4/runs/phase3_e2e.sh
