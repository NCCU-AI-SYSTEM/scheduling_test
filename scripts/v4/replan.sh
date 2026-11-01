#!/usr/bin/env bash
# Re-plan the R-set with the current planner prompt, keeping the previous plans as <slug>.<tag>.jsonl.
#   bash scripts/v4/replan.sh gemma4:12b v1prompt
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
SLUG=$(echo "$1" | sed 's/[^A-Za-z0-9.]\{1,\}/_/g')
mv "data/v4/plans/$SLUG.jsonl" "data/v4/plans/$SLUG.$2.jsonl"
$PY -m scripts.v4.run_e2e plan --planner "$1" --workers 1
$PY -m scripts.v4.score_planner "$1"
echo REPLAN DONE
