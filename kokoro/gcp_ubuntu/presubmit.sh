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
export HERMETIC_PYTHON_VERSION=3.12

mkdir -p "${WORK_DIR}"
mkdir -p "${BAZEL_OUTPUT_BASE}"

echo "=== KOKORO TIMING BENCHMARK RUN FORCE RECOMPILE 2 ==="
echo "=== 0. Bootstrapping system packages ==="
if ! command -v git &> /dev/null || ! dpkg -s python3-venv &> /dev/null || ! command -v curl &> /dev/null; then
  echo "Installing required system packages via apt-get..."
  apt-get update && apt-get install -y python3-venv python3-pip curl git ca-certificates
else
  echo "Required system packages already installed. Skipping apt-get..."
fi
git config --global http.sslVerify false

echo "=== 1. Navigating to checked-out repository ==="
# Kokoro clones the repo to $KOKORO_ARTIFACTS_DIR/github/tpu-raiden
export REPO_ROOT="${KOKORO_ARTIFACTS_DIR}/github/tpu-raiden"
cd "${REPO_ROOT}"

echo "=== 2. Setting up standalone Bazel environment ==="
# Read target Bazel version from metadata
export BAZEL_VERSION="$(tr -d '\r\n ' < ".bazelversion")"
echo "Target Bazel version: ${BAZEL_VERSION}"

BAZEL_BIN="${WORK_DIR}/bazel-${BAZEL_VERSION}"
if [[ -x "${BAZEL_BIN}" ]] && "${BAZEL_BIN}" --version 2>&1 | grep -qF "${BAZEL_VERSION}"; then
  echo "Found existing Bazel binary at ${BAZEL_BIN} matching version ${BAZEL_VERSION}."
elif command -v bazel &> /dev/null && bazel --version 2>&1 | grep -qF "${BAZEL_VERSION}"; then
  BAZEL_BIN="$(command -v bazel)"
  echo "Found system Bazel at ${BAZEL_BIN} matching version ${BAZEL_VERSION}."
elif which bazel &> /dev/null && "$(which bazel)" --version 2>&1 | grep -qF "${BAZEL_VERSION}"; then
  BAZEL_BIN="$(which bazel)"
  echo "Found system Bazel at ${BAZEL_BIN} matching version ${BAZEL_VERSION}."
else
  echo "Downloading standalone Bazel ${BAZEL_VERSION}..."
  curl -Lo "${BAZEL_BIN}" "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64"
  chmod +x "${BAZEL_BIN}"
fi

"${BAZEL_BIN}" --version

echo "=== 3. Setting up Python Virtual Environment ==="
if [[ ! -f "${WORK_DIR}/venv/bin/activate" ]]; then
  echo "Creating Python virtual environment at ${WORK_DIR}/venv..."
  python3 -m venv "${WORK_DIR}/venv"
else
  echo "Reusing existing Python virtual environment at ${WORK_DIR}/venv..."
fi
source "${WORK_DIR}/venv/bin/activate"

if command -v pip &> /dev/null && pip --version &> /dev/null; then
  echo "pip is already present and operational. Skipping upgrade."
else
  echo "Installing/upgrading pip..."
  pip install --upgrade pip
fi

echo "=== 4. E2E Validation Build with Bazel Remote Cache ==="
CACHE_BUCKET="tpu-raiden-bazel-cache"

# Set up a dummy torch_tpu module to satisfy Bazel module resolution for JAX-only CI
DUMMY_TORCH_TPU_MODULE="${WORK_DIR}/dummy_torch_tpu_module"
mkdir -p "${DUMMY_TORCH_TPU_MODULE}"
echo 'module(name = "torch_tpu", version = "0.1.1")' > "${DUMMY_TORCH_TPU_MODULE}/MODULE.bazel"

BAZEL_STARTUP_FLAGS=(
  "--output_base=${BAZEL_OUTPUT_BASE}"
)

# === Dynamic Resource Detection (75% Headroom Allocation) ===
echo "Detecting VM resources..."
NPROC=$(nproc)
BAZEL_JOBS=$((NPROC > 1 ? NPROC - 1 : 1))
MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
BAZEL_RAM=$((MEM_KB / 1024 * 75 / 100))
echo "Allocating ${BAZEL_JOBS} jobs/CPUs and ${BAZEL_RAM}MB RAM to Bazel..."

# === Dynamic Cache Write Access Toggle ===
if [[ "${KOKORO_JOB_TYPE}" == "PRESUBMIT_GITHUB" || "${KOKORO_JOB_TYPE}" == "PRESUBMIT" ]]; then
  echo "Presubmit build detected. Setting remote cache to READ-ONLY..."
  UPLOAD_RESULTS="false"
else
  echo "Postsubmit/trusted build detected. Setting remote cache to READ-WRITE..."
  UPLOAD_RESULTS="true"
fi

BAZEL_COMMAND_FLAGS=(
  "--cxxopt=-std=c++20"
  "--host_cxxopt=-std=c++20"
  "--remote_cache=https://storage.googleapis.com/${CACHE_BUCKET}"
  "--google_default_credentials"
  "--remote_download_minimal"
  "--spawn_strategy=standalone"
  "--strategy=standalone"
  "--jobs=${BAZEL_JOBS}"
  "--local_ram_resources=${BAZEL_RAM}"
  "--local_cpu_resources=${BAZEL_JOBS}"
  "--remote_upload_local_results=${UPLOAD_RESULTS}"
  "--remote_max_connections=25"
  "--remote_timeout=300s"
  "--remote_retries=3"
  "--override_module=torch_tpu=${DUMMY_TORCH_TPU_MODULE}"
  "--experimental_repo_remote_exec"
)

echo "Running build.sh..."
export BAZEL_BIN
./build.sh jax "${BAZEL_COMMAND_FLAGS[@]}"

echo "=== 5. Running CPU-bound standard unit tests ==="
"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" test -c opt --check_visibility=false --verbose_failures \
  "${BAZEL_COMMAND_FLAGS[@]}" \
  //tpu_raiden/kv_cache:logical_block_manager_test

# Find Bazel's hermetic Python 3.12 interpreter to avoid ABI mismatches with host Python 3.10
HERMETIC_PYTHON_BIN=$(find "${BAZEL_OUTPUT_BASE}/external" -path "*python_3_12*/bin/python3" | head -n 1)
if [[ -z "${HERMETIC_PYTHON_BIN}" ]]; then
  echo "Error: Could not find hermetic Python 3.12 interpreter under ${BAZEL_OUTPUT_BASE}/external" >&2
  exit 1
fi
echo "Using hermetic Python interpreter: ${HERMETIC_PYTHON_BIN}"

echo "=== 6. Verifying dynamic module binding linkage ==="
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/bazel-bin:${REPO_ROOT}/tpu_raiden/api/jax:${REPO_ROOT}/tpu_raiden/frameworks/jax:${PYTHONPATH}"

echo "Verifying import of all modules in a single process (with JAX mocked)..."
"${HERMETIC_PYTHON_BIN}" -c "
import sys
from unittest.mock import MagicMock
sys.modules['jax'] = MagicMock()
sys.modules['jax.core'] = MagicMock()
sys.modules['jax.extend'] = MagicMock()
sys.modules['jax.extend.ffi'] = MagicMock()

import kv_cache_manager
print('All modules imported successfully!')
"

echo "Dynamic linkage verified! All modules imported successfully on CPU!"

echo "=== Kokoro Build Verification Success! ==="
