#!/usr/bin/env bash
# Prod replica runs on eval v3 — the "should we port E7 into prod?" comparison.
# Waits for run_v3_chain.sh to finish so the GPU is free.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=logs/v3_chain.log
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
until grep -q "\] done" "$LOG" 2>/dev/null; do sleep 30; done
run() { stamp "run $1"; .venv/bin/python -m scripts.run_v3 --name "$@" 2>&1 | grep -E "^\S+ +(ALL|T|M|C) |Error|Traceback" | tee -a "$LOG"; }
run prod              --system prod --mode rrf
run prod_rerank       --system prod --mode rrf --rerank
run prod_oracle       --system prod --mode rrf --cond oracle --apply hard
run prod_rerank_oracle --system prod --mode rrf --rerank --cond oracle --apply hard
stamp "prod done"
