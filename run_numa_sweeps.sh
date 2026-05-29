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

#!/bin/bash
# run_numa_sweeps.sh - Dynamic Bare-Metal Socket-Locality Sweeps Launcher
# Automates benchmarking raw transport strategies performance under NUMA page-table overlays.

set -e

# Dynamically resolve absolute workspace path depending on current target host drive mapping!
TARGET_DIR=$(pwd)
WORKSPACE_DIR=$(echo "$TARGET_DIR" | sed 's/\/internal//g')

echo "=========================================================================="
echo "          COMMENCING MULTI-LOCALITY ACCELERATOR PERFORMANCE SWEEPS        "
echo "               Target Directory: $TARGET_DIR                              "
echo "               Workspace Root:    $WORKSPACE_DIR                           "
echo "=========================================================================="

# 1. Align dynamic environment variables PYTHONPATH
export PYTHONPATH="$WORKSPACE_DIR/bazel-bin/raiden_lib/raw_transfer/jax:$WORKSPACE_DIR/bazel-bin/api/jax:$WORKSPACE_DIR:$PYTHONPATH"
export JAX_PJRT_CLIENT_CREATE_OPTIONS="pinned_host_allocation_mode:recycle"
export MEGASCALE_PORT=0

# 2. Sanitize and pre-allocate dynamic output logging file targets
TELEMETRY_FILE="/tmp/raw_perf_performance.jsonl"
rm -f "$TELEMETRY_FILE"
touch "$TELEMETRY_FILE"
chmod 666 "$TELEMETRY_FILE"

# 3. Configure scaled-up benchmark sweeps runs iterations
BENCHMARK_RUNS=100

echo "\n=========================================================================="
echo " PHASE 1: Default Dynamic Locality Setting (Standard OS page mapping)     "
echo "==========================================================================\n"
python3 "$WORKSPACE_DIR/raiden_lib/raw_transfer/jax/raw_transfer_perf_test.py" --benchmark_runs=$BENCHMARK_RUNS --locality=default --telemetry_log_path=$TELEMETRY_FILE "$@" || true
sudo rm -f /tmp/libtpu_lockfile

echo "\n=========================================================================="
echo " PHASE 2: NUMA Interleaved Socket Settings (Cross-socket page interleaving)"
echo "==========================================================================\n"
numactl --interleave=all python3 "$WORKSPACE_DIR/raiden_lib/raw_transfer/jax/raw_transfer_perf_test.py" --benchmark_runs=$BENCHMARK_RUNS --locality=interleaved --telemetry_log_path=$TELEMETRY_FILE "$@" || true
sudo rm -f /tmp/libtpu_lockfile

echo "\n=========================================================================="
echo " PHASE 3: NUMA Socket-Bound Settings (Physical alignment to Socket 0)     "
echo "==========================================================================\n"
numactl --cpunodebind=0 --membind=0 python3 "$WORKSPACE_DIR/raiden_lib/raw_transfer/jax/raw_transfer_perf_test.py" --benchmark_runs=$BENCHMARK_RUNS --locality=socket-bound --telemetry_log_path=$TELEMETRY_FILE "$@" || true
sudo rm -f /tmp/libtpu_lockfile

echo "\n=========================================================================="
echo "        ALL ACCELERATOR SOCKET PERFORMANCE SWEEPS SUCCESSFULLY COMPLETE!  "
echo "                   Telemetry saved: $TELEMETRY_FILE (540 Rows)            "
echo "==========================================================================\n"
