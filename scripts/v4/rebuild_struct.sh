#!/usr/bin/env bash
# Run v4 tests, rebuild data/v4/course_struct.db from cached answers, print the coverage table.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
$PY -m pytest -q tests/ -k v4 2>&1 | tail -2
$PY -m scripts.v4.build_structured 2>&1 | tail -1 | cut -c1-160
$PY -m scripts.v4.verify_structured 2>&1 | grep -E '^\| (eval_item|eval_pct|ai_policy) '
