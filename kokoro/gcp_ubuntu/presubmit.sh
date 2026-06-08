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
# kokoro/gcp_ubuntu/presubmit.sh
# Kokoro entrypoint script for TPU Raiden GitHub PR verification.
set -e
set -o pipefail
set -x

# Resolve paths
export WORK_DIR="${KOKORO_ARTIFACTS_DIR}/workspace"
export BAZEL_OUTPUT_BASE="${KOKORO_ARTIFACTS_DIR}/bazel_cache"

mkdir -p "${WORK_DIR}"
mkdir -p "${BAZEL_OUTPUT_BASE}"

echo "=== 1. Navigating to checked-out repository ==="
# Kokoro clones the repo to $KOKORO_ARTIFACTS_DIR/github/tpu-raiden
export REPO_ROOT="${KOKORO_ARTIFACTS_DIR}/github/tpu-raiden"
cd "${REPO_ROOT}"

echo "=== 2. Setting up standalone Bazel environment ==="
# Read target Bazel version from metadata
export BAZEL_VERSION="$(tr -d '\r\n ' < ".bazelversion")"
echo "Target Bazel version: ${BAZEL_VERSION}"

BAZEL_BIN="${WORK_DIR}/bazel-${BAZEL_VERSION}"
echo "Downloading standalone Bazel ${BAZEL_VERSION}..."
curl -Lo "${BAZEL_BIN}" "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64"
chmod +x "${BAZEL_BIN}"

"${BAZEL_BIN}" --version

echo "=== 3. Setting up Python Virtual Environment ==="
apt-get update && apt-get install -y python3-venv python3-pip curl git
python3 -m venv "${WORK_DIR}/venv"
source "${WORK_DIR}/venv/bin/activate"
pip install --upgrade pip

echo "=== 4. E2E Validation Build with Bazel Remote Cache ==="
CACHE_BUCKET="tpu-raiden-bazel-cache"

BAZEL_FLAGS=(
  "--output_base=${BAZEL_OUTPUT_BASE}"
  "--remote_cache=https://storage.googleapis.com/${CACHE_BUCKET}"
  "--google_default_credentials"
  "--spawn_strategy=standalone"
  "--strategy=standalone"
  "--jobs=4"
  "--local_ram_resources=4096"
  "--local_cpu_resources=4"
)

echo "Running build_raw_transfer.sh..."
export BAZEL_BIN
./build_raw_transfer.sh jax "${BAZEL_FLAGS[@]}"

echo "=== 5. Running CPU-bound standard unit tests ==="
"${BAZEL_BIN}" "${BAZEL_FLAGS[@]}" test -c opt --check_visibility=false --verbose_failures \
  //kv_cache:logical_block_manager_test

echo "=== 6. Verifying dynamic module binding linkage ==="
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/bazel-bin:${REPO_ROOT}/api/jax:${REPO_ROOT}/frameworks/jax:${PYTHONPATH}"

python3 -c "
import sys
import ctypes
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
try:
    import jax
    import raw_transfer
    import _kv_cache_manager
    import _kv_cache_manager_ffi
    import _weight_synchronizer
    print('Dynamic linkage verified! All modules imported successfully on CPU!')
except Exception as e:
    print('Dynamic linkage verification failed:', e, file=sys.stderr)
    sys.exit(1)
"

echo "=== Kokoro Build Verification Success! ==="
