#!/usr/bin/env bash
# Launch a repo script detached from the ssh session, from the repo root.
#   ssh hopelab bash ~/workspace/code/nccu-course/v4/scripts/v4/bg.sh <log-name> <command...>
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
log="logs/$1.out"
shift
nohup "$@" > "$log" 2>&1 < /dev/null &
echo "started pid $! -> $log"
