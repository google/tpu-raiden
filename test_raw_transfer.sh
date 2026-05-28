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

# Run the JAX raw_transfer tests (raiden_lib/raw_transfer/jax):
#   1. raw_transfer_test.py       -- correctness unit tests
#   2. raw_transfer_perf_test.py  -- D2H/H2D benchmark vs jax.device_put
#
# Requires TPU devices and a built raw_transfer.so (run build_raw_transfer.sh first).
#
# Set PERF_ONLY=1 to skip the unit tests and run only the benchmark.

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${WORKSPACE_DIR}"

# Resolve imports of the form `from raiden_lib.raw_transfer.jax import raw_transfer`:
#   - WORKSPACE_DIR  -> source tree (the in-tree raw_transfer.so copied by build)
#   - bazel-bin      -> freshly built artifacts, if not yet copied into the tree
export PYTHONPATH="${WORKSPACE_DIR}:${WORKSPACE_DIR}/bazel-bin:${PYTHONPATH}"

if [[ "${PERF_ONLY:-0}" != "1" ]]; then
  echo "=== [1/2] Unit tests: raiden_lib/raw_transfer/jax/raw_transfer_test.py ==="
  python -m raiden_lib.raw_transfer.jax.raw_transfer_test 2>&1 \
    | tee "${WORKSPACE_DIR}/test.log"
  echo ""
fi

echo "=== [2/2] Benchmark: raiden_lib/raw_transfer/jax/raw_transfer_perf_test.py ==="
# Telemetry is appended to ${TELEMETRY_LOG_PATH:-/tmp/raw_perf_performance.jsonl}.
python -m raiden_lib.raw_transfer.jax.raw_transfer_perf_test \
  --telemetry_log_path="${TELEMETRY_LOG_PATH:-/tmp/raw_perf_performance.jsonl}" \
  ${BENCHMARK_RUNS:+--benchmark_runs="${BENCHMARK_RUNS}"} 2>&1 \
  | tee "${WORKSPACE_DIR}/perf_test.log"

echo ""
echo "=== Done ==="
