#!/usr/bin/env bash
# Phase 2 (Ollama jobs): eligibility judgments and R-set planner outputs per model.
cd "$(dirname "$0")/../.."
LOG=logs/phase2.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
unload() { for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done; }
until grep -q "GPU CHAIN DONE" logs/gpu_chain.log 2>/dev/null; do sleep 60; done
for m in gemma4:e4b gemma4:12b qwen3.5:4b qwen3.5:9b qwen3.6:35b-a3b gemma4:31b-cloud; do
  stamp "eligibility $m"
  $PY -m scripts.v4.eligibility_bench score --models $m >> "$LOG" 2>&1
  stamp "plan $m"
  $PY -m scripts.v4.run_e2e plan --planner $m --workers 1 >> "$LOG" 2>&1
  unload
done
stamp "PHASE2 DONE"
