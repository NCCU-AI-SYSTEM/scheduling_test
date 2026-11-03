#!/usr/bin/env bash
# The v4 search against a restored CourseDataPrep release (data.sql.gz in a fresh ParadeDB),
# compared with the phase 8 runs on the experiment tables.
#   V4_PG_DSN=postgresql://postgres:postgres@127.0.0.1:5437/academic bash scripts/v4/runs/release_e2e.sh
cd "$(dirname "$0")/../../.."
export V4_PG_DOC=v4.doc V4_PG_SESSION=v4.doc_session
PY=.venv/bin/python
base=(--struct --planner gemma4:12b --encoder e5-large-instruct --merge --named-fix --pg whitespace --pg-conditions)
$PY -m scripts.v4.run_e2e run --name release_g12b_norerank "${base[@]}" > logs/release_e2e.log 2>&1 || exit 1
$PY -m scripts.v4.run_e2e run --name release_g12b "${base[@]}" --reranker qwen3-rerank-0.6b >> logs/release_e2e.log 2>&1 || exit 1
$PY -m scripts.v4.compare_runs pg_whitespace_g12b release_g12b
$PY -m scripts.v4.compare_runs pg_whitespace_g12b_norerank release_g12b_norerank
