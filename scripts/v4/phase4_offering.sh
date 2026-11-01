#!/usr/bin/env bash
# Phase 4: one hit per class (cross-listed codes merged) and named-course lookups that
# skip condition filtering, against the phase-3 v4 runs; then judge new pairs and rescore.
cd "$(dirname "$0")/../.."
LOG=logs/phase4.log
PY=.venv/bin/python
ENC=e5-large-instruct
RR=qwen3-rerank-0.6b
ASP=data/v4/aspects/gemma4_12b.jsonl
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
run() { stamp "run $1"; $PY -m scripts.v4.run_e2e run --name "$@" >> "$LOG" 2>&1; }

run v4_e4b_merge --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --merge
run v4_e4b_namedfix --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --named-fix
run v4_e4b_merge_namedfix --planner gemma4:e4b --encoder $ENC --reranker $RR --aspects $ASP --merge --named-fix
run v4_g12b_merge_namedfix --planner gemma4:12b --encoder $ENC --reranker $RR --aspects $ASP --merge --named-fix
stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects $ASP >> "$LOG" 2>&1
stamp "PHASE4 DONE"
