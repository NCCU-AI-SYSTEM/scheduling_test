#!/usr/bin/env bash
# Pull a list of Ollama models sequentially; one log line per model.
#   bash scripts/v4/runs/pull_models.sh qwen3.5:2b gemma4:e4b ...
for m in "$@"; do
  echo "[$(date +%H:%M)] pull $m"
  if ollama pull "$m" >/dev/null 2>&1; then echo "ok $m"; else echo "FAIL $m"; fi
done
echo DONE
