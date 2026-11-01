#!/usr/bin/env bash
# Phase 6: quality of the CPU deployment preset (e4b planner, no reranker), then judge and score.
cd "$(dirname "$0")/../.."
LOG=logs/phase6.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
stamp "run v4s_e4b_merge_namedfix_norerank"
$PY -m scripts.v4.run_e2e run --name v4s_e4b_merge_namedfix_norerank --struct --planner gemma4:e4b \
  --encoder e5-large-instruct --merge --named-fix >> "$LOG" 2>&1
stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects data/v4/aspects/gemma4_12b.jsonl >> "$LOG" 2>&1
stamp "PHASE6 DONE"
