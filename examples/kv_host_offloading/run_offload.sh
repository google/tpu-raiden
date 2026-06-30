#!/bin/bash

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# One-shot KV host-offloading demo: start the offload server, wait until it is
# healthy, run the benchmark against it, print results + an offload-activity
# summary, then tear the server down.
#
# ASSUMES THE LOCAL TPU IS FREE -- there is NO pre-run cleanup / TPU process
# killing. Stops only the server this script started.
#
# Prerequisites: a python venv with tpu_raiden, vllm, and tpu_inference installed
# is ACTIVE (see README). Then:
#   bash run_offload.sh
set -u
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
LOG_DIR="${SCRIPTS_DIR}/tmp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs: ${LOG_DIR}"

wait_for_server() {
  local host=$1 port=$2 pid=$3
  echo "Waiting for server at ${host}:${port} (timeout ${SERVER_TIMEOUT}s)..."
  timeout "${SERVER_TIMEOUT}" bash -c "
    until curl -s ${host}:${port}/health >/dev/null 2>&1; do
      if [ -n '${pid}' ] && ! kill -0 ${pid} 2>/dev/null; then
        echo \"Error: server on ${host}:${port} (PID ${pid}) exited before becoming healthy\" >&2
        exit 1
      fi
      sleep 2
    done" && echo "Server at ${host}:${port} is ready." && return 0
  echo "Error: timed out waiting for ${host}:${port}" >&2
  return 1
}

# Stop ONLY the server this run launched. This is run teardown, not a TPU-wide
# cleanup.
shutdown() {
  echo "Stopping the server started by this run..."
  [ -n "${SERVER_PID:-}" ] && kill "${SERVER_PID}" 2>/dev/null
  pkill -9 -u "$(whoami)" -f "bin/vllm serve" 2>/dev/null
  pkill -9 -u "$(whoami)" -f "VLLM::Eng" 2>/dev/null
}
trap shutdown EXIT INT TERM

# 1. Start the offload server.
echo "Starting offload server (:${PORT})... (log: ${LOG_DIR}/server.log)"
bash "${SCRIPTS_DIR}/server.sh" >"${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

# 2. Wait for health (bails early if the server process dies during startup).
wait_for_server localhost "${PORT}" "${SERVER_PID}" || exit 1

# 3. Benchmark.
echo "Server ready. Running benchmark... (log: ${LOG_DIR}/benchmark.log)"
bash "${SCRIPTS_DIR}/benchmark.sh" >"${LOG_DIR}/benchmark.log" 2>&1
cat "${LOG_DIR}/benchmark.log"

# 4. Offload-activity summary from the server log.
echo ""
echo "==================== Offload activity ===================="
grep -F "External prefix cache hit rate" "${LOG_DIR}/server.log" | tail -1 \
  || echo "(no host-cache hit-rate line; raise VLLM_LOGGING_LEVEL or check the workload)"
grep "Offload Metrics Snapshot" "${LOG_DIR}/server.log" \
  | grep -vE "requests=0, hits=0, miss=0" | tail -5 || true
echo "========================================================="
echo "Done. Logs in ${LOG_DIR}"
