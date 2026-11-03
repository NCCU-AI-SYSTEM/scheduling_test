#!/usr/bin/env bash
# Run the test suite on clean git exports (no leftover synced files), sharing the v4 venv and data.
#   bash scripts/v4/runs/clean_tree_tests.sh <dir> [<dir> ...]
BASE=/home/mrfrog/workspace/code/nccu-course
for d in "$@"; do
  ln -sfn "$BASE/1142.db" "$d/data/1142.db"
  [ -e "$BASE/v4/data/v4/course_struct.db" ] && ln -sfn "$BASE/v4/data/v4/course_struct.db" "$d/data/v4/course_struct.db"
  echo "== $d"
  (cd "$d" && "$BASE/v4/.venv/bin/python" -m pytest -q tests/ 2>&1 | grep -E '^FAILED|passed|failed' | tail -8)
done
