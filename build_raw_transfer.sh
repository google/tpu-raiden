#!/bin/bash
# Copyright 2026 Google LLC
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

set -e

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
BAZEL_DISK_CACHE="${BAZEL_CACHE_DIR:-/mnt/disks/jcgu/bazel_cache/disk_cache}"
BAZEL_REPO_CACHE="${BAZEL_CACHE_DIR:-/mnt/disks/jcgu/bazel_cache/repo_cache}"

echo "=== Navigating to workspace directory ==="
cd "${WORKSPACE_DIR}"

echo "=== Building raw_transfer with Bazel ==="
bazel build -c opt --check_visibility=false //raw_transfer:raw_transfer_binaries --disk_cache=${BAZEL_DISK_CACHE} --repository_cache=${BAZEL_REPO_CACHE}

echo "=== Build Complete! ==="
echo "Artifacts are located in: ${WORKSPACE_DIR}/bazel-bin/raw_transfer/"

echo "=== Install Python Dependencies! ==="
pip install -r requirements.txt

echo "=== Installation Complete! ==="
