#!/usr/bin/env bash
# Validate prepare_semester without touching the live DB: 114-2 should pass and match the live
# build; 114-1 has no syllabus fields in 1142.db, so its gate should fail.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for s in 114-2 114-1; do
  echo "===== $s"
  $PY -m scripts.v4.prepare_semester --semester "$s" --skip-aspects
  echo "exit $?"
  cat "data/v4/staging/$s/prepare_report.md" 2>/dev/null
done
echo VALIDATE DONE
