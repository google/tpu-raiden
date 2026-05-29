#!/bin/bash
# Copyright 2026 Google LLC.
#
# Orchestrate a REAL two-node DisaggKVCacheManager transfer:
#   - prefill engine on this (LOCAL) host
#   - decode  engine on REMOTE_HOST
#   - discovery proxy on LOCAL so the two find each other's ephemeral ports
#
# Steps: clean both hosts -> sync code -> start proxy -> launch decode (remote)
# and prefill (local) -> wait for both -> report PASS/FAIL -> tear down.
#
# Usage:   ./run_two_node_disagg_test.sh
# Env:     REMOTE_HOST, DEVICE (tpu|cpu), DTYPE, N_LAYERS, ... (see disagg_env.sh)

set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/disagg_env.sh"

LOG_DIR="${DISAGG_SCRIPTS_DIR}/tmp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs: ${LOG_DIR}"
echo "LOCAL(prefill)=${LOCAL_IP}  REMOTE(decode)=${REMOTE_HOST}  proxy=${PROXY_ENDPOINT}"

PROXY_PID=""
DECODE_SSH_PID=""
PREFILL_PID=""

cleanup() {
  echo "Tearing down..."
  [ -n "${PREFILL_PID}" ]    && kill "${PREFILL_PID}"    2>/dev/null
  [ -n "${DECODE_SSH_PID}" ] && kill "${DECODE_SSH_PID}" 2>/dev/null
  [ -n "${PROXY_PID}" ]      && kill "${PROXY_PID}"      2>/dev/null
  bash "${DISAGG_SCRIPTS_DIR}/kill_disagg_nodes.sh" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

# 0. Clean stale processes on both hosts.
bash "${DISAGG_SCRIPTS_DIR}/kill_disagg_nodes.sh"

# 1. Sync code to the decode host (fails loud on md5 mismatch).
#    SKIP_SYNC=1 lets a batch runner sync once up front and skip per-run.
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  bash "${DISAGG_SCRIPTS_DIR}/sync_oss_to_remote.sh"
fi

# 2. Start the discovery proxy on LOCAL.
echo "Starting discovery proxy on :${PROXY_PORT} ... (log: ${LOG_DIR}/proxy.log)"
run_local "python '${OSS_DIR}/kv_cache/disagg_proxy.py' ${PROXY_PORT}" \
  >"${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
sleep 2

COMMON_ARGS="--proxy-endpoint ${PROXY_ENDPOINT} \
  --n-layers ${N_LAYERS} --block-size ${BLOCK_SIZE} --dtype ${DTYPE} \
  --device ${DEVICE} --num-requests ${NUM_REQUESTS} --mode ${MODE} \
  --transport-parallelism ${TRANSPORT_PARALLELISM} \
  --worker-parallelism ${WORKER_PARALLELISM} \
  --src-offsets ${SRC_OFFSETS} --dst-offsets ${DST_OFFSETS} --sizes ${SIZES}"

# 3. Launch the DECODE engine on REMOTE_HOST.
echo "Launching decode on ${REMOTE_HOST} ... (log: ${LOG_DIR}/decode.log)"
run_remote "PYTHONPATH='${DISAGG_PYTHONPATH}' \
  python '${DISAGG_SCRIPTS_DIR}/disagg_node_runner.py' \
  --role decode --self-name decode --self-ip ${REMOTE_HOST} ${COMMON_ARGS}" \
  >"${LOG_DIR}/decode.log" 2>&1 &
DECODE_SSH_PID=$!

# 4. Launch the PREFILL engine locally (it resolves 'decode' via the proxy and
#    blocks until decode has registered, so no manual sleep race here).
echo "Launching prefill locally ... (log: ${LOG_DIR}/prefill.log)"
run_local "PYTHONPATH='${DISAGG_PYTHONPATH}' \
  python '${DISAGG_SCRIPTS_DIR}/disagg_node_runner.py' \
  --role prefill --self-name prefill --self-ip ${LOCAL_IP} \
  --peer-name decode ${COMMON_ARGS}" \
  >"${LOG_DIR}/prefill.log" 2>&1 &
PREFILL_PID=$!

# 5. Wait for both, capture exit codes.
wait "${PREFILL_PID}";    PREFILL_RC=$?
wait "${DECODE_SSH_PID}"; DECODE_RC=$?
PREFILL_PID=""; DECODE_SSH_PID=""

echo ""
echo "===== prefill.log ====="; cat "${LOG_DIR}/prefill.log"
echo "===== decode.log  ====="; cat "${LOG_DIR}/decode.log"
echo ""
echo "prefill rc=${PREFILL_RC}  decode rc=${DECODE_RC}"
if [[ "${PREFILL_RC}" -eq 0 && "${DECODE_RC}" -eq 0 ]]; then
  echo "RESULT: PASS"
  exit 0
else
  echo "RESULT: FAIL"
  exit 1
fi
