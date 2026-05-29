#!/bin/bash
# Copyright 2026 Google LLC.
#
# Run the full two-node DisaggKVCacheManager test matrix and print a summary.
# Each row drives run_two_node_disagg_test.sh with a different config (request
# count, the two parallelism knobs, dtype, layers, transfer plan). Syncs the
# built artifacts to REMOTE_HOST once up front, then runs every case.
#
# Usage:   ./run_all_two_node_tests.sh
# Env:     REMOTE_HOST (default 10.128.1.8). Assumes _kv_cache_manager.so is
#          already built locally (run ../build_raw_transfer.sh first).

set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SELF_DIR}/disagg_env.sh"

echo "LOCAL(prefill)=${LOCAL_IP}  REMOTE(decode)=${REMOTE_HOST}"

# Refuse to run if the remote TPU is held by someone else's job.
if ssh "${REMOTE_HOST}" "pgrep -af generate_wan 2>/dev/null | grep -qv pgrep"; then
  echo "ERROR: remote TPU (${REMOTE_HOST}) is busy with another job; aborting." >&2
  exit 1
fi

# Sync once; the per-run orchestrator skips its own sync via SKIP_SYNC=1.
echo "=== Syncing artifacts to ${REMOTE_HOST} (once) ==="
bash "${SELF_DIR}/sync_oss_to_remote.sh" || { echo "sync failed" >&2; exit 1; }
export SKIP_SYNC=1

# Test matrix. Each entry: "label|VAR=val VAR=val ...".
# Default transfer plan is 2 blocks/request; transport_parallelism is clamped to
# the block count, so cases that need T>2 widen the plan to 4 chunks/request.
TESTS=(
  "baseline (1 req, T=1, W=1)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=1"
  "transport-only (1 req, T=2, W=1)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=1"
  "worker-only (2 req, T=1, W=2)|NUM_REQUESTS=2 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=2"
  "both (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2"
  "heavy (8 req, T=2, W=4)|NUM_REQUESTS=8 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=4"
  "4-stream (2 req, 4 blk, T=4, W=2)|NUM_REQUESTS=2 TRANSPORT_PARALLELISM=4 WORKER_PARALLELISM=2 SRC_OFFSETS=8,10,12,14 DST_OFFSETS=0,2,4,6 SIZES=2,2,2,2"
  "dtype bf16 (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 DTYPE=bf16"
  "dtype fp32 (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 DTYPE=fp32"
  "dtype fp8 (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 DTYPE=fp8"
  "more layers (n=8, 4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 N_LAYERS=8"
)

declare -a RESULTS=()
fail_count=0

for entry in "${TESTS[@]}"; do
  label="${entry%%|*}"
  envs="${entry#*|}"
  echo ""
  echo "================================================================"
  echo ">>> ${label}    [${envs}]"
  echo "================================================================"
  # Run with the per-case env in a subshell so vars don't leak between cases.
  out=$(env ${envs} bash "${SELF_DIR}/run_two_node_disagg_test.sh" 2>&1)
  rc=$?
  verify=$(echo "${out}" | grep -E "verify \(" | tail -1)
  result=$(echo "${out}" | grep -E "^RESULT:" | tail -1)
  echo "${verify:-（no verify line)}"
  echo "${result:-RESULT: (none) }  (exit=${rc})"
  if [[ ${rc} -eq 0 ]]; then
    RESULTS+=("PASS  ${label}")
  else
    RESULTS+=("FAIL  ${label}")
    fail_count=$((fail_count + 1))
    echo "----- tail of failing run -----"
    echo "${out}" | tail -25
  fi
done

echo ""
echo "================== SUMMARY (${#TESTS[@]} cases) =================="
for r in "${RESULTS[@]}"; do echo "  ${r}"; done
echo "================================================================"
if [[ ${fail_count} -eq 0 ]]; then
  echo "ALL ${#TESTS[@]} TWO-NODE TESTS PASSED"
  exit 0
else
  echo "${fail_count} / ${#TESTS[@]} TESTS FAILED"
  exit 1
fi
