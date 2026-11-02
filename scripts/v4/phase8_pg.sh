#!/usr/bin/env bash
# Phase 8: the same v4 search with ranking (and conditions) in PostgreSQL (ParadeDB), against
# the SQLite runs v4s_g12b_merge_namedfix and v4s_g12b_merge_namedfix_norerank.
cd "$(dirname "$0")/../.."
LOG=logs/phase8.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
base="--struct --planner gemma4:12b --encoder e5-large-instruct --merge --named-fix"
run() { stamp "run $1"; $PY -m scripts.v4.run_e2e run --name "$@" $base >> "$LOG" 2>&1; }
# runs need no Ollama (plans are precomputed); only judging does, and it waits its turn there

for tok in whitespace ngram jieba; do
  run "pg_${tok}_g12b" --pg $tok --reranker qwen3-rerank-0.6b
  run "pg_${tok}_g12b_norerank" --pg $tok
done
run pg_whitespace_cond_g12b --pg whitespace --pg-conditions --reranker qwen3-rerank-0.6b
stamp "judge"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
stamp "score"
$PY -m scripts.v4.run_e2e score --aspects data/v4/aspects/gemma4_12b.jsonl >> "$LOG" 2>&1
stamp "PHASE8 DONE"
