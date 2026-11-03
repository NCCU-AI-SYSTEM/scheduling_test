#!/usr/bin/env bash
# Phase 3: end-to-end systems on the R-set, then judging and scoring.
#   ENC=bge-m3 RR=bge-reranker-v2-m3 bash scripts/v4/runs/phase3_e2e.sh
cd "$(dirname "$0")/../../.."
LOG=logs/phase3.log
PY=.venv/bin/python
ENC=e5-large-instruct  # judged retrieval bench: 0.834 vs bge-m3 0.800 at the same speed
RR=${RR:-bge-reranker-v2-m3}
ASP=data/v4/aspects/gemma4_12b.jsonl
HYPE=data/v4/hype/gemma4_e4b.jsonl
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
unload() { for m in $(ollama ps | awk 'NR>1{print $1}'); do ollama stop "$m"; done; }
until grep -q "PHASE25 DONE" logs/phase25.log 2>/dev/null; do sleep 60; done
unload
run() { stamp "run $1"; $PY -m scripts.v4.run_e2e run --name "$@" >> "$LOG" 2>&1; }

run prod --system prod
run raw_bm25 --raw-query
run raw_hybrid --raw-query --encoder $ENC --reranker $RR
run v4_e4b --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP
run v4_e4b_nocond --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --no-cond
run v4_e4b_nonamed --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --no-named
run v4_e4b_noprefs --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --no-prefs
run v4_e4b_norerank --planner gemma4:e4b --encoder $ENC --aspects $ASP
run v4_e4b_textraw --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --text-raw
run v4_e4b_bgem3 --planner gemma4:e4b --encoder bge-m3 --reranker $RR --aspects $ASP
run v4_e4b_hype --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --hype $HYPE
run v4_q9b --planner qwen3.5:9b --encoder $ENC --reranker $RR --aspects $ASP
run v4_g12b --planner gemma4:12b --encoder $ENC --reranker $RR --aspects $ASP
run v4_q4b --planner qwen3.5:4b --encoder $ENC --reranker $RR --aspects $ASP
run v4_g31cloud --planner gemma4:31b-cloud --encoder $ENC --reranker $RR --aspects $ASP
run v4_goldplan --planner reference --encoder $ENC --reranker $RR --aspects $ASP

stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
unload
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects $ASP >> "$LOG" 2>&1
stamp "PHASE3 DONE"
