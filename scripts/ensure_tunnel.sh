#!/usr/bin/env bash
# Ensure SSH tunnel to home_mac:11434 is alive. Idempotent: starts only if absent.
set -e
HOST="${1:-home_mac}"
PORT="${2:-11434}"

if curl -s --max-time 3 "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
  echo "[tunnel] alive"
  exit 0
fi

# Kill any zombie ssh forwarders for this port
pkill -f "ssh.*${PORT}:127.0.0.1:${PORT}.*${HOST}" 2>/dev/null || true
sleep 1

ssh -fNL "${PORT}:127.0.0.1:${PORT}" "${HOST}"
sleep 2

if curl -s --max-time 3 "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
  echo "[tunnel] reconnected to ${HOST}:${PORT}"
  exit 0
fi

echo "[tunnel] FAILED to reconnect to ${HOST}:${PORT}" >&2
exit 1
