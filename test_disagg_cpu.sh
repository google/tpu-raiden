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

# Run all CPU-only tests for DisaggKVCacheManager:
#   1. C++ gunit tests (lifecycle, ports, peer registry, submit preconditions,
#      ThreadSafeQueue, and the mocked E2E push flow) via Bazel.
#   2. Python nanobind binding tests (struct + enum + PjRtCopyFuture).

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${WORKSPACE_DIR}"

BAZEL_VERSION="7.7.0"
if [[ -f "${WORKSPACE_DIR}/.bazelversion" ]]; then
  BAZEL_VERSION="$(tr -d '\r\n ' < "${WORKSPACE_DIR}/.bazelversion")"
fi
BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
BAZEL_CACHE_BASE="${BAZEL_CACHE_DIR:-/mnt/disks/jcgu/bazel_cache}"

echo "=== [1/2] C++ unit tests: //kv_cache:disagg_kv_cache_manager_base_test ==="
"${BAZEL_BIN}" test -c opt --check_visibility=false \
  --experimental_repo_remote_exec --incompatible_disallow_empty_glob=false \
  --repo_env=HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION:-3.12}" \
  --disk_cache="${BAZEL_CACHE_BASE}/disk_cache" \
  --repository_cache="${BAZEL_CACHE_BASE}/repo_cache" \
  --test_output=summary \
  //kv_cache:disagg_kv_cache_manager_base_test

echo ""
echo "=== [2/2] Python binding tests: api/jax/disagg_kv_cache_manager_cpu_unit_test.py ==="
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH}"
python api/jax/disagg_kv_cache_manager_cpu_unit_test.py 2>&1 \
  | tee "${WORKSPACE_DIR}/disagg_cpu.log"

echo ""
echo "=== Done ==="
