#!/usr/bin/env bash
# After the main LLM bench finishes, rerun the tool task with the strict system prompt.
#   bash scripts/v4/run_tool_strict.sh qwen3.5:2b gemma4:e4b ...
cd "$(dirname "$0")/../.."
LOG=logs/llm_bench.log
until grep -q "ALL DONE" "$LOG"; do sleep 60; done
for m in "$@"; do
  echo "[$(date +%H:%M)] strict-tool $m" >> "$LOG"
  .venv/bin/python -m scripts.v4.llm_bench --model "$m" --task tool --tool-variant strict >> "$LOG" 2>&1
  [[ "$m" != *-cloud ]] && ollama stop "$m" >/dev/null 2>&1
done
echo "[$(date +%H:%M)] STRICT DONE" >> "$LOG"
