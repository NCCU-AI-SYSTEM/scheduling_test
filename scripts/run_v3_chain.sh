#!/usr/bin/env bash
# Eval v3 end-to-end on a single 16GB GPU. Ollama and torch cannot share the card,
# so every stage that switches between them unloads the Ollama model first.
#
#   bash scripts/run_v3_chain.sh [pid-to-wait-for]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=logs/v3_chain.log
mkdir -p logs
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
unload() { for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done; }

if [[ $# -ge 1 ]]; then
  stamp "waiting for pid $1"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

stamp "decompose M (qwen3:14b)"
$PY -m scripts.build_eval_v3 decompose --workers 4 >>"$LOG" 2>&1
unload

stamp "pool (GPU)"
$PY -m scripts.build_eval_v3 pool >>"$LOG" 2>&1

stamp "judge (qwen3:14b)"
$PY -m scripts.build_eval_v3 judge --workers 4 >>"$LOG" 2>&1

stamp "parse all v3 queries (gemma4:e4b)"
$PY - >>"$LOG" 2>&1 <<'EOF'
from src.eval.v3_metrics import load_eval
from src.parsers.ollama_parser import OllamaParser
qs, _ = load_eval()
OllamaParser("gemma4:e4b").parse_many([q.query for q in qs])
print("parsed", len(qs))
EOF
unload

stamp "baselines"
run() { stamp "run $1"; $PY -m scripts.run_v3 --name "$@" 2>&1 | grep -E "^\S+ +(ALL|T|M|C) " | tee -a "$LOG"; }
run bm25          --mode bm25
run dense         --mode dense
run rrf           --mode rrf --expand
run e7            --mode rrf --expand --rerank
run e7_oracle     --mode rrf --expand --rerank --cond oracle --apply hard
run e7_oracle_need --mode rrf --expand --rerank --cond oracle --apply hard --dense-query need
run e7_regex      --mode rrf --expand --rerank --cond regex  --apply hard
run e7_parser     --mode rrf --expand --rerank --cond parser --apply hard
run e7_parser_need --mode rrf --expand --rerank --cond parser --apply hard --dense-query need
stamp "done"
