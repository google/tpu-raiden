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
export HERMETIC_PYTHON_VERSION=3.13

mkdir -p "${WORK_DIR}"
mkdir -p "${BAZEL_OUTPUT_BASE}"

echo "=== 0. Bootstrapping system packages ==="
apt-get update && apt-get install -y python3-venv python3-pip curl git ca-certificates
git config --global http.sslVerify false

# Fetch OAuth2 token from GCS Metadata Server if available
echo "Fetching GCS OAuth2 token from Metadata Server..."
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
TOKEN_JSON=$(curl -s -H "Metadata-Flavor: Google" "${METADATA_URL}")
if [ $? -ne 0 ] || [ -z "${TOKEN_JSON}" ]; then
  echo "Warning: Failed to fetch token from Metadata Server. GCS download might fail if bucket is private."
  GCS_TOKEN=""
else
  GCS_TOKEN=$(echo "${TOKEN_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))")
fi

# Helper function to download from GCS using curl
download_from_gcs() {
  local bucket="$1"
  local object="$2"
  local dest="$3"
  
  echo "Downloading gs://${bucket}/${object} to ${dest}..."
  local url="https://storage.googleapis.com/${bucket}/${object}"
  
  if [ -n "${GCS_TOKEN}" ]; then
    curl -Lo "${dest}" -H "Authorization: Bearer ${GCS_TOKEN}" "${url}"
  else
    curl -Lo "${dest}" "${url}"
  fi
  
  if [ $? -ne 0 ] || [ ! -f "${dest}" ]; then
    echo "Error: Failed to download gs://${bucket}/${object}"
    return 1
  fi
  return 0
}

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

echo "=== 3. E2E Validation Build with Bazel Remote Cache ==="
CACHE_BUCKET="tpu-raiden-bazel-cache"
CACHE_VERSION="v2"
TORCH_TPU_SOURCE_TAR="torch_tpu-0.1.1-source.tar.gz"

# Download and extract torch_tpu source
echo "=== Downloading torch_tpu source from GCS ==="
TORCH_TPU_DIR="${WORK_DIR}/torch_tpu"
mkdir -p "${TORCH_TPU_DIR}"
if ! download_from_gcs "${CACHE_BUCKET}" "sources/${TORCH_TPU_SOURCE_TAR}" "${WORK_DIR}/${TORCH_TPU_SOURCE_TAR}"; then
  echo "Error: Failed to download torch_tpu source tarball from GCS!" >&2
  exit 1
fi
tar -xzf "${WORK_DIR}/${TORCH_TPU_SOURCE_TAR}" -C "${TORCH_TPU_DIR}"
export TORCH_TPU_MODULE_PATH="${TORCH_TPU_DIR}"

# Inject missing Bzlmod dependency for the pybind shim
echo "=== 3a. Injecting pybind11_bazel dependency into torch_tpu MODULE.bazel ==="
echo 'bazel_dep(name = "pybind11_bazel", version = "2.13.6")' >> "${WORK_DIR}/torch_tpu/MODULE.bazel"

# === 3a. On-the-fly Google3-to-OSS Path Translation ===
echo "=== 3a. Translating Google3 absolute paths to OSS Bazel paths ==="

# 1. Translate Bazel target paths in BUILD files
echo "Translating BUILD files..."
find "${TORCH_TPU_DIR}" -type f -name "BUILD" -exec sed -i \
  -e 's|//third_party/tensorflow/compiler/|@xla//|g' \
  -e 's|//third_party/tensorflow/tsl/|@tsl//tsl/|g' \
  -e 's|//third_party/tensorflow/third_party/py/rules_pywrap:pywrap_bzl|@xla//third_party/py/rules_pywrap:pywrap.default.bzl|g' \
  -e 's|//third_party/tensorflow/third_party/py/rules_pywrap:pywrap.google.bzl|@xla//third_party/py/rules_pywrap:pywrap.default.bzl|g' \
  -e 's|//third_party/llvm/llvm-project/mlir:|@llvm-project//mlir:|g' \
  -e 's|//third_party/llvm/llvm-project/llvm:|@llvm-project//llvm:|g' \
  -e 's|//third_party/absl/|@com_google_absl//absl/|g' \
  -e 's|//third_party/stablehlo/stablehlo/integrations/cpp/builder:|@stablehlo//:|g' \
  -e 's|//third_party/stablehlo/|@stablehlo//|g' \
  -e 's|//third_party/stablehlo:|@stablehlo//:|g' \
  -e 's|//tools/build_defs/license|@rules_license//rules|g' \
  -e 's|//devtools/python/blaze|//shims/py_rules|g' \
  -e 's|//devtools/build_cleaner/skylark|//shims/build_cleaner|g' \
  -e 's|//tools/build_defs/testing|//shims/build_defs_testing|g' \
  -e 's|//third_party/py/torch/|//shims/torch/|g' \
  -e 's|//third_party/py/torch:|//shims/torch:|g' \
  -e 's|//third_party/py/torch"|//shims/torch"|g' \
  -e "s|//third_party/py/torch'|//shims/torch'|g" \
  -e 's|//third_party/re2|@com_googlesource_code_re2//:re2|g' \
  -e 's|//third_party/py/jax/experimental:pallas|//shims/jax:pallas|g' \
  -e 's|//third_party/py/jax/experimental:pallas_tpu|//shims/jax:pallas_tpu|g' \
  {} +

# 2. Translate C++ include paths in source files
echo "Translating C++ include paths..."
find "${TORCH_TPU_DIR}" -type f \( -name "*.cc" -o -name "*.cpp" -o -name "*.h" \) -exec sed -i \
  -e 's|third_party/tensorflow/compiler/xla/|xla/|g' \
  -e 's|third_party/tensorflow/tsl/|tsl/|g' \
  -e 's|third_party/llvm/llvm-project/mlir/include/mlir/|mlir/|g' \
  -e 's|third_party/llvm/llvm-project/llvm/include/llvm/|llvm/|g' \
  -e 's|third_party/absl/|absl/|g' \
  -e 's|third_party/stablehlo/stablehlo/|stablehlo/|g' \
  -e 's|third_party/re2/|re2/|g' \
  {} +

# ==============================================================================
# Bzlmod Shims for Google-Internal Dependencies
# ==============================================================================
echo "=== 3b. Creating shims for Google-internal Bazel dependencies ==="
pushd "${WORK_DIR}/torch_tpu" > /dev/null

# 1. Stub build_cleaner (Line 17)
echo "Creating shim for build_cleaner..."
mkdir -p devtools/build_cleaner/skylark
touch devtools/build_cleaner/skylark/BUILD
cat << 'EOF' > devtools/build_cleaner/skylark/build_defs.bzl
def register_extension_info(**kwargs):
    pass
EOF

# 2. Stub pytype strict rules (Line 18)
echo "Creating shim for pytype..."
mkdir -p devtools/python/blaze
touch devtools/python/blaze/BUILD
cat << 'EOF' > devtools/python/blaze/pytype.bzl
load("@rules_python//python:defs.bzl", "py_library", "py_test")
def pytype_strict_library(name, **kwargs):
    py_library(name = name, **kwargs)
def pytype_strict_contrib_test(name, **kwargs):
    py_test(name = name, **kwargs)
EOF

# 3. Redirect rules_cc cc_library (Line 19)
echo "Creating shim for rules_cc cc_library..."
mkdir -p third_party/bazel_rules/rules_cc/cc
touch third_party/bazel_rules/rules_cc/cc/BUILD
cat << 'EOF' > third_party/bazel_rules/rules_cc/cc/cc_library.bzl
load("@rules_cc//cc:defs.bzl", _cc_library = "cc_library")
cc_library = _cc_library
EOF

# 4. Redirect rules_cc cc_test (Line 20)
echo "Creating shim for rules_cc cc_test..."
cat << 'EOF' > third_party/bazel_rules/rules_cc/cc/cc_test.bzl
load("@rules_cc//cc:defs.bzl", _cc_test = "cc_test")
cc_test = _cc_test
EOF

# 5. Stub rules_python (Line 21)
echo "Creating shim for rules_python..."
mkdir -p third_party/bazel_rules/rules_python/python
touch third_party/bazel_rules/rules_python/python/BUILD
cat << 'EOF' > third_party/bazel_rules/rules_python/python/py_test.bzl
load("@rules_python//python:defs.bzl", _py_test = "py_test")
py_test = _py_test
EOF

# 6. Stub TensorFlow rules_pywrap (Line 24)
echo "Creating shim for rules_pywrap..."
mkdir -p third_party/tensorflow/third_party/py/rules_pywrap
touch third_party/tensorflow/third_party/py/rules_pywrap/BUILD
cat << 'EOF' > third_party/tensorflow/third_party/py/rules_pywrap/pywrap.google.bzl
def use_pywrap_rules():
    return False
EOF

# 7. Stub build_test (Line 28)
echo "Creating shim for build_test..."
mkdir -p tools/build_defs/build_test
touch tools/build_defs/build_test/BUILD
cat << 'EOF' > tools/build_defs/build_test/build_test.bzl
def build_test(**kwargs):
    pass
EOF

# 8. Stub internal py_platform_test (Create directly at the absolute path)
echo "Creating shim for py_platform_test..."
mkdir -p third_party/py/torch_tpu/google/build_rules
touch third_party/py/torch_tpu/google/build_rules/BUILD
cat << 'EOF' > third_party/py/torch_tpu/google/build_rules/py_platform_test.bzl
load("@rules_python//python:defs.bzl", "py_test")
def py_platform_test(name, **kwargs):
    # Drop internal platform arg and fallback to standard py_test
    kwargs.pop("platform", None)
    py_test(name = name, **kwargs)
EOF

# 9. Stub internal google/build_files/build_defs.bzl
echo "Creating shim for google build_defs..."
mkdir -p third_party/py/torch_tpu/google/build_files
touch third_party/py/torch_tpu/google/build_files/BUILD
cat << 'EOF' > third_party/py/torch_tpu/google/build_files/build_defs.bzl
def add_internal_filesystem_dependencies():
    return []

def process_accelerator_tags(tags):
    pass

TT_FRIENDS = []
EOF

# 10. Shim C++ mocks for google/common
echo "Creating C++ shims for google/common..."
mkdir -p google/common

cat << 'EOF' > google/common/BUILD
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "environment",
    srcs = ["environment.cc"],
    hdrs = ["environment.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status",
        "//third_party/py/torch_tpu/distributed/slicebuilder:discovery",
    ],
)

cc_library(
    name = "contain",
    srcs = ["contain.cc"],
    hdrs = ["//third_party/py/torch_tpu/common:contain.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ],
)
EOF

cat << 'EOF' > google/common/environment.h
#ifndef TORCH_TPU_GOOGLE_COMMON_ENVIRONMENT_H_
#define TORCH_TPU_GOOGLE_COMMON_ENVIRONMENT_H_

#include "third_party/absl/status/status.h"
#include "distributed/slicebuilder/discovery.h"

namespace torch_tpu {

absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config);

absl::Status InitializeSingleDeviceEnvironment();

}  // namespace torch_tpu

#endif  // TORCH_TPU_GOOGLE_COMMON_ENVIRONMENT_H_
EOF

cat << 'EOF' > google/common/environment.cc
#include "google/common/environment.h"

namespace torch_tpu {

absl::Status InitializeDistributedEnvironment(
    const DistributedWorkerConfiguration& config) {
  return absl::OkStatus();
}

absl::Status InitializeSingleDeviceEnvironment() {
  return absl::OkStatus();
}

}  // namespace torch_tpu
EOF

cat << 'EOF' > google/common/contain.cc
#include "common/contain.h"
#include "third_party/absl/status/statusor.h"

namespace torch_tpu {

struct ScopedMemMeasuringContainer::Impl {};

ScopedMemMeasuringContainer::ScopedMemMeasuringContainer()
    : impl_(std::make_unique<Impl>()) {}

ScopedMemMeasuringContainer::~ScopedMemMeasuringContainer() = default;

absl::StatusOr<int64_t> ContainerPeakHostMemoryBytes() {
  return 0;
}

void CleanUpContainer() {}

}  // namespace torch_tpu
EOF

# Create symlink to satisfy Bazel package loader looking for absolute Google3 paths
echo "Creating package loader symlink for google/common..."
mkdir -p third_party/py/torch_tpu/google
ln -s ../../../../google/common third_party/py/torch_tpu/google/common

# 11. Stub internal google/shims/pybind.bzl
echo "Creating shim for google/shims/pybind..."
mkdir -p google/shims
touch google/shims/BUILD
cat << 'EOF' > google/shims/pybind.bzl
load("@pybind11_bazel//:build_defs.bzl", _pybind_extension = "pybind_extension")

def pybind_extension(name, **kwargs):
    # Strip Google-internal attributes
    kwargs.pop("common_lib_packages", None)
    kwargs.pop("py_deps", None)
    kwargs.pop("wrap_py_init", None)
    # Pass through to standard OSS pybind_extension
    _pybind_extension(name = name, **kwargs)
EOF

# Create package loader symlink for google/shims
echo "Creating package loader symlink for google/shims..."
mkdir -p third_party/py/torch_tpu/google
ln -s ../../../../google/shims third_party/py/torch_tpu/google/shims

popd > /dev/null
# ==============================================================================

echo "=== Pre-fetching Hermetic Python to parse versions ==="
# Force Bazel to download the hermetic Python toolchain by fetching a python target
"${BAZEL_BIN}" --output_base="${BAZEL_OUTPUT_BASE}" fetch \
  --experimental_repo_remote_exec \
  --check_visibility=false \
  --check_bzl_visibility=false \
  --override_module=torch_tpu="${TORCH_TPU_DIR}" \
  //api/jax:raw_transfer

HERMETIC_PYTHON_BIN=$(find "${BAZEL_OUTPUT_BASE}/external" -path "*python_$(echo ${HERMETIC_PYTHON_VERSION} | tr . _)*/bin/python3" | head -n 1)
if [[ -z "${HERMETIC_PYTHON_BIN}" ]]; then
  echo "Error: Could not find hermetic Python ${HERMETIC_PYTHON_VERSION} interpreter after fetch" >&2
  exit 1
fi
echo "Using hermetic Python interpreter: ${HERMETIC_PYTHON_BIN}"

echo "=== Setting up Python ${HERMETIC_PYTHON_VERSION} Virtual Environment ==="
"${HERMETIC_PYTHON_BIN}" -m venv "${WORK_DIR}/venv"
source "${WORK_DIR}/venv/bin/activate"

# Parse versions from requirements.txt
JAX_VERSION=$(python -c "
with open('requirements.txt') as f:
  for line in f:
    if line.startswith('jax=='):
      print(line.strip().split('==')[1])
      break
")
JAXLIB_VERSION=$(python -c "
with open('requirements.txt') as f:
  for line in f:
    if line.startswith('jaxlib=='):
      print(line.strip().split('==')[1])
      break
")
LIBTPU_VERSION=$(python -c "
with open('requirements.txt') as f:
  for line in f:
    if line.startswith('libtpu=='):
      print(line.strip().split('==')[1])
      break
")

if [[ -z "${JAX_VERSION}" || -z "${JAXLIB_VERSION}" || -z "${LIBTPU_VERSION}" ]]; then
  echo "Error: Failed to parse JAX, Jaxlib, or LibTPU versions from requirements.txt" >&2
  exit 1
fi

TORCH_VERSION="2.10.0"

WHEELS=(
  "filelock-3.29.3-py3-none-any.whl"
  "fsspec-2026.4.0-py3-none-any.whl"
  "jax-${JAX_VERSION}-py3-none-any.whl"
  "jaxlib-${JAXLIB_VERSION}-cp313-cp313-manylinux_2_27_x86_64.whl"
  "jinja2-3.1.6-py3-none-any.whl"
  "libtpu-${LIBTPU_VERSION}-cp313-cp313-manylinux_2_31_x86_64.whl"
  "markupsafe-3.0.3-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
  "ml_dtypes-0.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  "mpmath-1.3.0-py3-none-any.whl"
  "networkx-3.6.1-py3-none-any.whl"
  "numpy-2.4.6-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  "opt_einsum-3.4.0-py3-none-any.whl"
  "scipy-1.17.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  "setuptools-82.0.1-py3-none-any.whl"
  "sympy-1.14.0-py3-none-any.whl"
  "torch-${TORCH_VERSION}+cpu-cp313-cp313-manylinux_2_28_x86_64.whl"
  "typing_extensions-4.15.0-py3-none-any.whl"
)

WHEELS_DIR="${WORK_DIR}/wheels"
mkdir -p "${WHEELS_DIR}"

echo "=== Downloading wheels from GCS ==="
for wheel in "${WHEELS[@]}"; do
  echo "Downloading ${wheel}..."
  if ! download_from_gcs "${CACHE_BUCKET}" "wheels/${wheel}" "${WHEELS_DIR}/${wheel}"; then
    echo "========================================================================"
    echo "ERROR: Failed to download ${wheel} from GCS!"
    echo "This wheel is required for the presubmit workflow but is missing from the cache."
    echo "Please run the internal sync script to download and upload the missing wheels:"
    echo "  third_party/tpu_raiden/internal/sync_wheels.sh"
    echo "========================================================================"
    exit 1
  fi
done

echo "=== Installing wheels ==="
pip install --no-index --find-links="${WHEELS_DIR}" "${WHEELS_DIR}"/*.whl

BAZEL_STARTUP_FLAGS=(
  "--output_base=${BAZEL_OUTPUT_BASE}"
)

BAZEL_COMMAND_FLAGS=(
  "--remote_cache=https://storage.googleapis.com/${CACHE_BUCKET}/${CACHE_VERSION}"
  "--google_default_credentials"
  "--spawn_strategy=standalone"
  "--strategy=standalone"
  "--jobs=4"
  "--local_ram_resources=4096"
  "--local_cpu_resources=4"
  "--remote_max_connections=25"
  "--remote_timeout=300s"
  "--remote_retries=3"
  "--override_module=torch_tpu=${WORK_DIR}/torch_tpu"
  "--experimental_repo_remote_exec"
  "--check_visibility=false"
  "--check_bzl_visibility=false"
)

echo "Running build_raw_transfer.sh..."
export BAZEL_BIN
./build_raw_transfer.sh both "${BAZEL_COMMAND_FLAGS[@]}"

echo "=== 4. Running CPU-bound standard unit tests ==="
"${BAZEL_BIN}" "${BAZEL_STARTUP_FLAGS[@]}" test -c opt --check_visibility=false --verbose_failures \
  "${BAZEL_COMMAND_FLAGS[@]}" \
  //kv_cache:logical_block_manager_test

echo "=== 5. Verifying dynamic module binding linkage ==="
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/bazel-bin:${REPO_ROOT}/api/jax:${REPO_ROOT}/frameworks/jax:${REPO_ROOT}/api/torch:${REPO_ROOT}/frameworks/torch:${PYTHONPATH}"
export TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND=1

echo "Verifying imports using verify_imports.py..."
python "${REPO_ROOT}/kokoro/gcp_ubuntu/verify_imports.py"

echo "=== Kokoro Build Verification Success! ==="

