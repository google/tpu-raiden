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

# Refuse to run only if the remote TPU is ACTUALLY held (vfio device open by a
# python process). The old check keyed on a generate_wan process name, which
# false-positives on hung/idle/<defunct> leftovers that no longer hold the TPU.
# Set SKIP_BUSY_CHECK=1 to skip entirely.
if [[ "${SKIP_BUSY_CHECK:-0}" != "1" ]]; then
  if ssh "${REMOTE_HOST}" "sudo lsof -w /dev/vfio/* 2>/dev/null | grep -q python"; then
    echo "ERROR: remote TPU (${REMOTE_HOST}) vfio device is in use; aborting." >&2
    exit 1
  fi
  if ssh "${REMOTE_HOST}" "pgrep -f generate_wan.py >/dev/null 2>&1"; then
    echo "WARN: a generate_wan.py process exists on ${REMOTE_HOST} but the TPU" \
         "vfio device is free (likely hung/idle); proceeding." >&2
  fi
fi

# Sync once; the per-run orchestrator skips its own sync via SKIP_SYNC=1.
echo "=== Syncing artifacts to ${REMOTE_HOST} (once) ==="
bash "${SELF_DIR}/sync_oss_to_remote.sh" || { echo "sync failed" >&2; exit 1; }
export SKIP_SYNC=1

# Test matrix. Each entry: "label|VAR=val VAR=val ...".
# Pull-only (await_pull/pull); MODE defaults to pull (see disagg_env.sh). With
# staging now manager-allocated, the caller gives only device offsets:
# SRC_OFFSETS = prefill source device, DST_OFFSETS = decode destination device.
# Default BLOCK_SIZE=1, default plan = 2 blocks/request. transport_parallelism T
# must divide each request's block count, so wider-T cases widen the plan.
TESTS=(
  "baseline (1 req, T=1, W=1)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=1"
  "transport-only (1 req, T=2, W=1)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=1"
  "worker-only (2 req, T=1, W=2)|NUM_REQUESTS=2 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=2"
  "both (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2"
  "heavy (8 req, T=2, W=4)|NUM_REQUESTS=8 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=4"
  "4-stream (2 req, 4 blk, T=4, W=2)|NUM_REQUESTS=2 TRANSPORT_PARALLELISM=4 WORKER_PARALLELISM=2 SRC_OFFSETS=8,9,10,11 DST_OFFSETS=0,1,2,3 SIZES=1,1,1,1"
  "dtype bf16 (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 DTYPE=bf16"
  "dtype fp32 (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 DTYPE=fp32"
  "dtype fp8 (4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 DTYPE=fp8"
  "more layers (n=8, 4 req, T=2, W=2)|NUM_REQUESTS=4 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=2 N_LAYERS=8"
  # ---- non-contiguous DEVICE destination (gapped DST_OFFSETS on the decode
  # device; staging is manager-allocated). The decode skips a device block.
  "non-contig dst (1 req, gap, T=1)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=1 SRC_OFFSETS=4,5 DST_OFFSETS=0,2 SIZES=1,1"
  "non-contig dst (1 req, T=2)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=2 WORKER_PARALLELISM=1 SRC_OFFSETS=8,9,10,11 DST_OFFSETS=0,1,3,4 SIZES=1,1,1,1"
  # ---- multi-block chunks (SIZES a multiple of BLOCK_SIZE): exercises the
  # chunk->unit-block expansion plus producer-side staging auto-allocation over
  # multi-block chunks. sizes 2+3 = 5 blocks/req (T must divide it, so T=1).
  "multiblock (1 req, sizes 2,3, T=1)|NUM_REQUESTS=1 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=1 SRC_OFFSETS=1,4 DST_OFFSETS=10,13 SIZES=2,3"
  "multiblock (2 req, sizes 2,3, W=2)|NUM_REQUESTS=2 TRANSPORT_PARALLELISM=1 WORKER_PARALLELISM=2 SRC_OFFSETS=1,4 DST_OFFSETS=10,13 SIZES=2,3"
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
