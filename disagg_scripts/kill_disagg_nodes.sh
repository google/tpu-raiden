#!/bin/bash
# Copyright 2026 Google LLC.
#
# Kill leftover harness processes (node runner + discovery proxy) on BOTH hosts.
# Safe to run anytime. Unlike scripts/kill_*.sh this does NOT touch vllm.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/disagg_env.sh"

_kill_local() {
  echo "[local ${LOCAL_IP}] killing disagg_node_runner / disagg_proxy ..."
  pkill -9 -f "disagg_node_runner.py" 2>/dev/null || true
  pkill -9 -f "disagg_proxy.py" 2>/dev/null || true
  local pids
  pids=$(lsof -ti:"${PROXY_PORT}" 2>/dev/null || true)
  [ -n "${pids}" ] && kill -9 ${pids} 2>/dev/null || true
}

_kill_remote() {
  echo "[remote ${REMOTE_HOST}] killing disagg_node_runner ..."
  ssh "${REMOTE_HOST}" "pkill -9 -f 'disagg_node_runner.py' 2>/dev/null; \
    pkill -9 -f 'disagg_proxy.py' 2>/dev/null; true"
}

_kill_local
_kill_remote
echo "Cleanup done."
