#!/usr/bin/env bash
# Re-run judging and scoring for every run file still missing grades (phase 7 + 8).
cd "$(dirname "$0")/../.."
LOG=logs/rejudge.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1 || stamp "JUDGE FAILED"
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects data/v4/aspects/gemma4_12b.jsonl >> "$LOG" 2>&1
stamp "REJUDGE DONE"
