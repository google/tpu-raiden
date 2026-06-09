#!/bin/bash
# Two-node Raiden TransferEngine benchmark orchestrator.
#
# Launches producer.py locally (prefill host) and consumer.py on the decode
# host over ssh, wiring the consumer to the producer's control endpoint. Uses
# the real Qwen3-32B TP=8 fp8 KV spec (see kv_spec.py).
#
# Env knobs (all optional):
#   CONSUMER_HOST   decode host            (default 10.128.0.8 / aman)
#   PRODUCER_IP     producer host IP       (default: first hostname -I addr)
#   CONTROL_PORT    producer control port  (default 9200)
#   NUM_REQUESTS    pulls to issue         (default 1)
#   NUM_SLOTS       host staging slots     (default: max(NUM_REQUESTS,8))
#   NUM_BLOCKS      blocks/req (64 => 2GiB)(default 64)
#   MAX_BLOCKS      per-req block budget   (default 72)
#   PARALLELISM     H2H sockets per pull   (default 4)
#   SEED            rng seed               (default 1234)
#   VERIFY          1 to byte-verify (N=1) (default 1 when NUM_REQUESTS==1)
#   CONDA_ENV       conda env              (default raiden1)
#
# Examples:
#   bash run_two_node_test.sh                       # single 2GiB pull + verify
#   NUM_REQUESTS=8  NUM_SLOTS=8  bash run_two_node_test.sh   # 8-way contention
#   NUM_REQUESTS=30 NUM_SLOTS=30 bash run_two_node_test.sh   # 30-way (the regression)

set -u

WORKSPACE_DIR="${WORKSPACE_DIR:-/mnt/disks/jcgu/code/ullm/raiden}"
TPU_RAIDEN="${WORKSPACE_DIR}/tpu-raiden"
TEST_DIR="${TPU_RAIDEN}/two_node_test"
CONDA_SH="${CONDA_SH:-/mnt/disks/jcgu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-raiden1}"

CONSUMER_HOST="${CONSUMER_HOST:-10.128.0.8}"
PRODUCER_IP="${PRODUCER_IP:-$(hostname -I | awk '{print $1}')}"
CONTROL_PORT="${CONTROL_PORT:-9200}"
NUM_REQUESTS="${NUM_REQUESTS:-1}"
NUM_SLOTS="${NUM_SLOTS:-0}"   # 0 => python default max(NUM_REQUESTS,8)
NUM_BLOCKS="${NUM_BLOCKS:-64}"
MAX_BLOCKS="${MAX_BLOCKS:-72}"
PARALLELISM="${PARALLELISM:-4}"
SEED="${SEED:-1234}"
MAPPING="${MAPPING:-identity}"   # identity | reorder | sparse
VERIFY_FULL="${VERIFY_FULL:-0}"  # 1 => check every layer/block
# Cache dim0 (full block pool). Each request takes a distinct window, so we need
# >= NUM_REQUESTS * window where window = 2*NUM_BLOCKS (sparse) else NUM_BLOCKS.
if [ "${MAPPING}" = "sparse" ]; then _win=$((2 * NUM_BLOCKS)); else _win=${NUM_BLOCKS}; fi
_need=$((NUM_REQUESTS * _win))
KV_CACHE_NUM_BLOCKS="${KV_CACHE_NUM_BLOCKS:-$(( _need > 1024 ? _need : 1024 ))}"
if [ "${NUM_REQUESTS}" -eq 1 ]; then
  VERIFY="${VERIFY:-1}"
else
  VERIFY="${VERIFY:-0}"
fi

LOG_DIR="${TEST_DIR}/tmp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Producer=${PRODUCER_IP}:${CONTROL_PORT}  Consumer=${CONSUMER_HOST}  "\
"requests=${NUM_REQUESTS} slots=${NUM_SLOTS} blocks=${NUM_BLOCKS} "\
"kv_cache_num_blocks=${KV_CACHE_NUM_BLOCKS} mapping=${MAPPING} "\
"parallelism=${PARALLELISM} verify=${VERIFY} verify_full=${VERIFY_FULL}"

ENV_PREFIX="source ${CONDA_SH} && conda activate ${CONDA_ENV} && cd ${TEST_DIR} && export PYTHONPATH=${TPU_RAIDEN}:\${PYTHONPATH:-} && export RAIDEN_STAGING_MODE=${RAIDEN_STAGING_MODE:-}"

cleanup_local() { pkill -9 -f "two_node_test/producer.py" 2>/dev/null || true; }
cleanup_remote() {
  ssh "${CONSUMER_HOST}" "pkill -9 -f 'two_node_test/consumer.py' 2>/dev/null || true" 2>/dev/null || true
}

finish() {
  echo "Shutting down..."
  [ -n "${PRODUCER_PID:-}" ] && kill "${PRODUCER_PID}" 2>/dev/null || true
  cleanup_local
}
trap finish EXIT INT TERM

echo "== cleanup leftover test procs =="
cleanup_local
cleanup_remote
sleep 2

echo "== sync test files to ${CONSUMER_HOST} =="
ssh "${CONSUMER_HOST}" "mkdir -p '${TEST_DIR}'" 2>/dev/null || true
scp -q "${TEST_DIR}/kv_spec.py" "${TEST_DIR}/consumer.py" "${CONSUMER_HOST}:${TEST_DIR}/" \
  && echo "  synced kv_spec.py consumer.py" || { echo "scp failed" >&2; exit 1; }

echo "== launch producer locally =="
bash -c "${ENV_PREFIX} && python producer.py \
  --control-port ${CONTROL_PORT} --num-blocks ${NUM_BLOCKS} \
  --max-blocks ${MAX_BLOCKS} --kv-cache-num-blocks ${KV_CACHE_NUM_BLOCKS} \
  --num-requests ${NUM_REQUESTS} \
  --num-slots ${NUM_SLOTS} --seed ${SEED} --mapping ${MAPPING}" \
  >"${LOG_DIR}/producer.log" 2>&1 &
PRODUCER_PID=$!

echo "== wait for PRODUCER_READY =="
ready_port=""
for _ in $(seq 1 600); do
  if ! kill -0 "${PRODUCER_PID}" 2>/dev/null; then
    echo "Producer exited early; log:" >&2; cat "${LOG_DIR}/producer.log" >&2; exit 1
  fi
  line=$(grep -m1 "PRODUCER_READY" "${LOG_DIR}/producer.log" 2>/dev/null || true)
  if [ -n "${line}" ]; then
    ready_port=$(echo "${line}" | sed -n 's/.*control_port=\([0-9]*\).*/\1/p')
    break
  fi
  sleep 1
done
if [ -z "${ready_port}" ]; then
  echo "Timed out waiting for PRODUCER_READY" >&2; cat "${LOG_DIR}/producer.log" >&2; exit 1
fi
echo "  producer ready on control_port=${ready_port}"

echo "== launch consumer on ${CONSUMER_HOST} =="
verify_flag=""; [ "${VERIFY}" = "1" ] && verify_flag="--verify"
[ "${VERIFY_FULL}" = "1" ] && verify_flag="${verify_flag} --verify-full"
ssh "${CONSUMER_HOST}" "bash -i -c '${ENV_PREFIX} && python consumer.py \
  --remote-endpoint ${PRODUCER_IP}:${ready_port} --num-blocks ${NUM_BLOCKS} \
  --max-blocks ${MAX_BLOCKS} --kv-cache-num-blocks ${KV_CACHE_NUM_BLOCKS} \
  --num-requests ${NUM_REQUESTS} \
  --num-slots ${NUM_SLOTS} --parallelism ${PARALLELISM} --seed ${SEED} \
  --mapping ${MAPPING} ${verify_flag}'" 2>&1 | tee "${LOG_DIR}/consumer.log"
CONSUMER_RC=${PIPESTATUS[0]}

echo "== wait for producer to finish (done_sending) =="
for _ in $(seq 1 120); do
  kill -0 "${PRODUCER_PID}" 2>/dev/null || break
  grep -q "PRODUCER_DONE\|PRODUCER_TIMEOUT" "${LOG_DIR}/producer.log" 2>/dev/null && break
  sleep 1
done

echo "===================== PRODUCER TIMING ====================="
grep "RAIDEN_TIMING event=producer_pipeline" "${LOG_DIR}/producer.log" 2>/dev/null | tail -n "${NUM_REQUESTS}"
grep -E "PRODUCER_DONE|PRODUCER_TIMEOUT" "${LOG_DIR}/producer.log" 2>/dev/null
echo "===================== CONSUMER TIMING ====================="
grep "RAIDEN_TIMING event=consumer_pipeline" "${LOG_DIR}/consumer.log" 2>/dev/null | tail -n "${NUM_REQUESTS}"

echo "consumer_rc=${CONSUMER_RC}  logs=${LOG_DIR}"
exit "${CONSUMER_RC}"
