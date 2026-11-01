#!/usr/bin/env bash
# Run llm_bench for each model in turn; waits for a model to finish pulling.
# Cloud models (suffix -cloud) skip the wait.
#   bash scripts/v4/run_llm_bench.sh qwen3.5:2b gemma4:e4b gemma4:31b-cloud
cd "$(dirname "$0")/../.."
LOG=logs/llm_bench.log
mkdir -p logs
for m in "$@"; do
  if [[ "$m" != *-cloud ]]; then
    until ollama list | awk 'NR>1{print $1}' | grep -qx "$m"; do sleep 30; done
  fi
  echo "[$(date +%H:%M)] bench $m" >> "$LOG"
  .venv/bin/python -m scripts.v4.llm_bench --model "$m" --task parse,tool >> "$LOG" 2>&1
  [[ "$m" != *-cloud ]] && ollama stop "$m" >/dev/null 2>&1
done
echo "[$(date +%H:%M)] ALL DONE" >> "$LOG"
