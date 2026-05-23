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
set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
BAZEL_CACHE_BASE="/mnt/disk"
BAZEL_DISK_CACHE="${BAZEL_CACHE_BASE}/disk_cache"
BAZEL_REPO_CACHE="${BAZEL_CACHE_BASE}/repo_cache"

echo "=== Navigating to workspace directory ==="
cd "${WORKSPACE_DIR}"

echo "=== Building raw_transfer, kv_cache_manager, and Raiden transfer engine with Bazel ==="
bazel build -c opt --check_visibility=false --verbose_failures --experimental_repo_remote_exec --incompatible_disallow_empty_glob=false \
  --repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION:-3.12} \
  //raiden_lib/raw_transfer/jax:raw_transfer \
  //kv_cache:kv_cache_manager \
  //kv_cache:kv_cache_store \
  //kv_cache:_raiden_transfer_engine.so \
  --disk_cache=${BAZEL_DISK_CACHE} \
  --repository_cache=${BAZEL_REPO_CACHE}


echo "=== Copying compiled shared libraries to source directory ==="
cp -f "${WORKSPACE_DIR}/bazel-bin/raiden_lib/raw_transfer/jax/raw_transfer.so" "${WORKSPACE_DIR}/raiden_lib/raw_transfer/jax/"
cp -f "${WORKSPACE_DIR}/bazel-bin/kv_cache/kv_cache_manager.so" "${WORKSPACE_DIR}/kv_cache/"
cp -f "${WORKSPACE_DIR}/bazel-bin/kv_cache/kv_cache_store.so" "${WORKSPACE_DIR}/kv_cache/"
cp -f "${WORKSPACE_DIR}/bazel-bin/kv_cache/_raiden_transfer_engine.so" "${WORKSPACE_DIR}/kv_cache/"


echo "=== Build Complete! ==="
echo "Artifacts are located in: ${WORKSPACE_DIR}/bazel-bin/raiden_lib/raw_transfer/jax/"

echo "=== Install Python Dependencies! ==="
pip install -r requirements.txt

echo "=== Installation Complete! ==="
