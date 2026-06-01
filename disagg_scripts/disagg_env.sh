#!/bin/bash
# Copyright 2026 Google LLC.
#
# Shared configuration + helpers for the two-node DisaggKVCacheManager harness.
# Source this from the other scripts:  source "$(dirname "$0")/disagg_env.sh"
#
# Everything here can be overridden from the environment, e.g.
#   REMOTE_HOST=10.128.1.9 ./run_two_node_disagg_test.sh

# --- Hosts -------------------------------------------------------------------
# LOCAL  = prefill engine (this box).  REMOTE = decode engine.
# LOCAL_IP must be an address the REMOTE host can reach back on (peer pushes
# KV to LOCAL only for H2H_READ; for the push test only REMOTE needs to be
# reachable from LOCAL, but the discovery proxy runs on LOCAL so REMOTE dials
# LOCAL_IP).
export REMOTE_HOST="${REMOTE_HOST:-10.128.1.8}"
export LOCAL_IP="${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"

# --- Workspace + conda -------------------------------------------------------
# This harness lives at <OSS_DIR>/disagg_scripts; OSS_DIR is its parent. The OSS
# tree must exist at the SAME absolute path on both hosts.
export DISAGG_SCRIPTS_DIR="${DISAGG_SCRIPTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export OSS_DIR="${OSS_DIR:-$(dirname "${DISAGG_SCRIPTS_DIR}")}"
export CONDA_ENV="${CONDA_ENV:-raiden1}"

# PYTHONPATH so `from api.jax import disagg_kv_cache_manager` resolves and finds
# the in-tree _kv_cache_manager.so (copied there by build_raw_transfer.sh).
export DISAGG_PYTHONPATH="${OSS_DIR}:${OSS_DIR}/bazel-bin"

# --- Discovery proxy (kv_cache/disagg_proxy.py) ------------------------------
# Runs on LOCAL; both engines REGISTER/RESOLVE ephemeral ports through it.
export PROXY_PORT="${PROXY_PORT:-5599}"
export PROXY_ENDPOINT="tcp://${LOCAL_IP}:${PROXY_PORT}"

# --- Test shape / transfer plan (shared by both roles) -----------------------
# Defaults mirror the known-good single-process test_e2e_disagg_push.
export N_LAYERS="${N_LAYERS:-2}"
# BLOCK_SIZE = page granularity along the MAJOR dim (slices per block). The
# manager stages one block per (offset,size) chunk, so each SIZES entry must
# equal BLOCK_SIZE (one block per chunk). With BLOCK_SIZE=1 each chunk is a
# single major-dim slice and DST_OFFSETS double as the staging block ids.
export BLOCK_SIZE="${BLOCK_SIZE:-1}"
export DTYPE="${DTYPE:-int32}"
export DEVICE="${DEVICE:-tpu}"
# Transfer direction. Only "pull" (decode pulls from prefill via await_pull/pull)
# is supported; push was removed from this harness.
export MODE="${MODE:-pull}"
# Two distinct H2H parallelism knobs (see disagg_kv_cache_manager.py):
#   TRANSPORT_PARALLELISM = parallel TCP streams per single Push/Pull (BlockTransport)
#   WORKER_PARALLELISM    = concurrent H2H worker threads (transfers in flight)
export TRANSPORT_PARALLELISM="${TRANSPORT_PARALLELISM:-1}"
export WORKER_PARALLELISM="${WORKER_PARALLELISM:-1}"
# Number of concurrent requests; each gets a disjoint, independently-verified
# region (request i is shifted by i*sum(SIZES) on both src and dst).
export NUM_REQUESTS="${NUM_REQUESTS:-1}"
# Transfer plan: prefill copies src->dst chunks of these sizes (major-dim slice
# units). With BLOCK_SIZE=1 each chunk is one block, so the default is 2 unit
# blocks (SIZES all 1); DST_OFFSETS are the staging block ids.
export SRC_OFFSETS="${SRC_OFFSETS:-4,5}"
export DST_OFFSETS="${DST_OFFSETS:-0,1}"
export SIZES="${SIZES:-1,1}"

# Run a python module on the REMOTE host inside the conda env, via an
# interactive login shell so `conda activate` works (mirrors run_all.sh).
run_remote() {
  ssh "${REMOTE_HOST}" "bash -i -c 'conda activate ${CONDA_ENV} && $*'"
}

# Run a command locally inside the conda env (assumes conda is on PATH).
run_local() {
  bash -i -c "conda activate ${CONDA_ENV} && $*"
}
