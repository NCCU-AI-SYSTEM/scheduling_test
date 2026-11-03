#!/usr/bin/env bash
# Is v4 ready to run from a release alone? On hopelab, against the release in CourseDataPrep's database.
#   bash scripts/v4/runs/port_readiness.sh
cd "$(dirname "$0")/../../.."
LOG=logs/port_readiness.log
PY=.venv/bin/python
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }
DSN=postgresql://postgres:postgres@127.0.0.1:5433/academic
RELEASE=../CourseDataPrep/releases/current

stamp "bundle parity"
V4_STRUCT_DB="$RELEASE/course_struct.db" V4_PG_PARITY_DSN=$DSN $PY -m pytest -q tests/test_v4_struct_pg.py >> "$LOG" 2>&1
stamp "bundle parity exit $?"

stamp "search from postgres only"
export V4_PG_DSN=$DSN V4_PG_DOC=v4.doc V4_PG_SESSION=v4.doc_session
base=(--struct-pg --planner gemma4:12b --encoder e5-large-instruct --merge --named-fix --pg whitespace --pg-conditions)
$PY -m scripts.v4.run_e2e run --name port_pg_g12b "${base[@]}" --reranker qwen3-rerank-0.6b >> "$LOG" 2>&1
stamp "search exit $?"
$PY -m scripts.v4.compare_runs release_g12b port_pg_g12b >> "$LOG" 2>&1

stamp "planner over the OpenAI-compatible endpoint"
ollama stop qwen3:14b >/dev/null 2>&1
V4_OPENAI_EXTRA='{"reasoning_effort": "none"}' $PY -m scripts.v4.analysis.planner_endpoint_parity \
  --base-url http://localhost:11434/v1 --model gemma4:12b --model gemma4:e4b >> "$LOG" 2>&1
stamp "planner exit $?"

stamp "judge and score"
$PY -m scripts.v4.run_e2e judge --workers 4 >> "$LOG" 2>&1
$PY -m scripts.v4.run_e2e score --aspects data/v4/aspects/gemma4_12b.jsonl >> "$LOG" 2>&1
stamp "PORT READINESS DONE"
