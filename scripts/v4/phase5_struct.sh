#!/usr/bin/env bash
# Phase 5: search reads only course_struct.db. Rebuild it with the runtime tables, run the
# best configurations through the struct path, judge new pairs, rescore.
cd "$(dirname "$0")/../.."
LOG=logs/phase5.log
PY=.venv/bin/python
ENC=e5-large-instruct
RR=qwen3-rerank-0.6b
ASP=data/v4/aspects/gemma4_12b.jsonl
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
run() { stamp "run $1"; $PY -m scripts.v4.run_e2e run --name "$@" >> "$LOG" 2>&1; }

stamp "build"
$PY -m scripts.v4.build_structured >> "$LOG" 2>&1
$PY -m scripts.v4.build_runtime --encoder $ENC --aspects $ASP >> "$LOG" 2>&1
$PY -m pytest -q tests/ -k v4 >> "$LOG" 2>&1
run v4s_e4b_merge_namedfix --struct --planner gemma4:e4b --encoder $ENC --reranker $RR --merge --named-fix
run v4s_g12b_merge_namedfix --struct --planner gemma4:12b --encoder $ENC --reranker $RR --merge --named-fix
run v4s_g12b_merge_namedfix_norerank --struct --planner gemma4:12b --encoder $ENC --merge --named-fix
stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects $ASP >> "$LOG" 2>&1
stamp "PHASE5 DONE"
