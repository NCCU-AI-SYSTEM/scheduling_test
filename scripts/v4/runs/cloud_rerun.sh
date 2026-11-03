#!/usr/bin/env bash
# Rerun the cloud Gemma 31B eligibility and planner jobs after the fenced-JSON fix.
# Cloud calls use no local GPU, so this can overlap local Ollama jobs.
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
rm -f data/v4/plans/gemma4_31b_cloud.jsonl
$PY -m scripts.v4.analysis.eligibility_bench score --models gemma4:31b-cloud,hybrid:gemma4:31b-cloud
$PY -m scripts.v4.run_e2e plan --planner gemma4:31b-cloud --workers 1
$PY -m scripts.v4.score_planner gemma4:e4b gemma4:12b qwen3.5:4b qwen3.5:9b qwen3.6:35b-a3b gemma4:31b-cloud
echo CLOUD RERUN DONE
