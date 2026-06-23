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
set -e

download_file() {
  local url="$1"
  local dest="$2"
  if command -v curl > /dev/null; then
    curl -Lo "$dest" "$url"
  elif command -v wget > /dev/null; then
    wget -O "$dest" "$url"
  elif command -v python3 > /dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('$url', '$dest')"
  elif command -v python > /dev/null; then
    python -c "import urllib; urllib.urlretrieve('$url', '$dest')"
  else
    echo "Error: No download tool found (curl, wget, python3, python)." >&2
    return 1
  fi
}

check_disk_space() {
  local dir="$1"
  local desc="$2"
  # Warn if free disk space is under 20 GB (20971520 KB)
  local min_kb=20971520
  mkdir -p "$dir" 2>/dev/null || true
  if command -v df > /dev/null && command -v awk > /dev/null; then
    local free_kb
    free_kb="$(df -P "$dir" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < min_kb )); then
      local free_gb=$((free_kb / 1048576))
      echo "WARNING: Low disk space on ${desc} (${dir}). Only ~${free_gb} GB free (< 20 GB recommended for Bazel)." >&2
    fi
  fi
}

# Define directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
DEFAULT_WORKSPACE_DIR="$SCRIPT_DIR"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
# Check for persistent storage mounted on Cloud TPU VMs (3-4 TB SSDs)
if [[ -d "/mnt/disks/persistent/${USER}" && -w "/mnt/disks/persistent/${USER}" ]] || [[ -d /mnt/disks/persistent && -w /mnt/disks/persistent ]]; then
  DEFAULT_BAZEL_CACHE_BASE="/mnt/disks/persistent/${USER}/tpu-raiden-bazel-cache"
  DEFAULT_BAZEL_OUTPUT_BASE="/mnt/disks/persistent/${USER}/bazel-output-user-root/tpu_raiden_${USER}"
elif [[ -d "/mnt/disk/${USER}" && -w "/mnt/disk/${USER}" ]] || [[ -d /mnt/disk && -w /mnt/disk ]]; then
  DEFAULT_BAZEL_CACHE_BASE="/mnt/disk/${USER}/tpu-raiden-bazel-cache"
  DEFAULT_BAZEL_OUTPUT_BASE="/mnt/disk/${USER}/bazel-output-user-root/tpu_raiden_${USER}"
else
  DEFAULT_BAZEL_CACHE_BASE="${HOME}/.bazel_cache"
  DEFAULT_BAZEL_OUTPUT_BASE="/tmp/tpu_raiden_bazel_output_${USER}"
fi
BAZEL_CACHE_BASE="${BAZEL_CACHE_DIR:-${DEFAULT_BAZEL_CACHE_BASE}}"
BAZEL_DISK_CACHE="${BAZEL_CACHE_BASE}/disk_cache"
BAZEL_REPO_CACHE="${BAZEL_CACHE_BASE}/repo_cache"
BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${DEFAULT_BAZEL_OUTPUT_BASE}}"

check_disk_space "${WORKSPACE_DIR}" "Workspace Directory"
check_disk_space "${BAZEL_CACHE_BASE}" "Bazel Cache Base"
check_disk_space "${BAZEL_OUTPUT_BASE}" "Bazel Output Base"
check_disk_space "/tmp" "Temporary Directory (/tmp)"

echo "=== Navigating to workspace directory ==="
cd "${WORKSPACE_DIR}"
TORCH_TPU_MODULE_PATH="${TORCH_TPU_MODULE_PATH:-../torch_tpu}"

# 0. Set up standalone Bazel environment based on .bazelversion in /tmp
BAZEL_VERSION="7.7.0"
VERSION_FILE="${WORKSPACE_DIR}/.bazelversion"
if [[ -f "${VERSION_FILE}" ]]; then
  BAZEL_VERSION="$(cat "${VERSION_FILE}" | tr -d '\r\n ')"
  echo "Parsed target Bazel version from .bazelversion: ${BAZEL_VERSION}"
fi

BAZEL_BIN="/tmp/bazel-bootstrap-${BAZEL_VERSION}"
if [[ ! -f "${BAZEL_BIN}" ]]; then
  echo "Bootstrapping standalone Bazel ${BAZEL_VERSION} to temporary folder ${BAZEL_BIN}..."
  download_file "https://storage.googleapis.com/bazel/${BAZEL_VERSION}/release/bazel-${BAZEL_VERSION}-linux-x86_64" "${BAZEL_BIN}"
  chmod +x "${BAZEL_BIN}"
fi

"${BAZEL_BIN}" --version

# Default behavior based on auto-detection
BUILD_JAX=true
BUILD_TORCH=true

if [ ! -f "${TORCH_TPU_MODULE_PATH}/MODULE.bazel" ]; then
  echo "torch_tpu checkout not found at ${TORCH_TPU_MODULE_PATH}. Defaulting to JAX-only build."
  BUILD_TORCH=false
fi

# Parse command line arguments
if [ "$#" -gt 0 ]; then
  case "$1" in
    jax)
      BUILD_JAX=true
      BUILD_TORCH=false
      shift
      ;;
    torch)
      BUILD_JAX=false
      BUILD_TORCH=true
      shift
      ;;
    both)
      BUILD_JAX=true
      BUILD_TORCH=true
      shift
      ;;
    -*)
      # Standard Bazel/environment flags: keep defaults and do not shift.
      ;;
    *)
      echo "Usage: $0 [jax|torch|both] [bazel_flags...]"
      exit 1
      ;;
  esac
fi

if [ "${BUILD_TORCH}" = true ]; then
  # Ensure clang-18 is installed on the host (required for PyTorch C++ extensions)
  if ! command -v clang-18 > /dev/null || ! command -v clang++-18 > /dev/null; then
    echo "clang-18 / clang++-18 not found on host. Attempting automatic installation..."
    if command -v apt-get > /dev/null && command -v sudo > /dev/null; then
      sudo apt-get update && sudo apt-get install -y clang-18
    else
      echo "Error: clang-18 / clang++-18 is required to build Torch extensions." >&2
      echo "Please install clang-18 manually on your system." >&2
      exit 1
    fi
  fi

  if [[ -z "${CC:-}" ]]; then
    export CC="$(command -v clang-18)"
  fi
  if [[ -z "${CXX:-}" ]]; then
    export CXX="$(command -v clang++-18)"
  fi
fi

BAZEL_TARGETS=(
  "//tpu_raiden/weight_sync:weight_synchronizer_service_py_pb2"
  "//tpu_raiden/rpc:coordination_py_pb2"
  "//tpu_raiden/rpc:coordination_py_pb2_grpc"
)
DEFINE_FLAGS=""
BAZEL_MODULE_FLAGS=()
TORCH_REPO_ENV_FLAGS=()

if [ "$BUILD_JAX" = true ]; then
  echo "Configuring build for JAX..."
  BAZEL_TARGETS+=(
    "//tpu_raiden/frameworks/jax:_tpu_raiden_jax"
  )
else
  DEFINE_FLAGS+=" --define with_jax=false"
fi

if [ "$BUILD_TORCH" = true ]; then
  echo "Configuring build for Torch..."
  if [[ ! -f "${TORCH_TPU_MODULE_PATH}/MODULE.bazel" ]]; then
    echo "Error: Torch build requires a torch_tpu checkout at ${TORCH_TPU_MODULE_PATH}." >&2
    echo "Set TORCH_TPU_MODULE_PATH to override the default ../torch_tpu location." >&2
    exit 1
  fi
  TORCH_TPU_MODULE_PATH="$(cd "${TORCH_TPU_MODULE_PATH}" && pwd)"
  BAZEL_MODULE_FLAGS+=("--override_module=torch_tpu=${TORCH_TPU_MODULE_PATH}")
  if [[ -z "${TORCH_SOURCE:-}" ]]; then
    TORCH_SOURCE="$(python3 - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("torch")
if spec is None or not spec.submodule_search_locations:
  raise SystemExit("torch package not found on Python path")
print(pathlib.Path(next(iter(spec.submodule_search_locations))).resolve().parent)
PY
)"
  fi
  export TORCH_SOURCE
  echo "Using local torch from: ${TORCH_SOURCE}"
  DEFINE_FLAGS+=" --define=TORCH_SOURCE=local"
  TORCH_REPO_ENV_FLAGS+=("--repo_env=TORCH_SOURCE=${TORCH_SOURCE}")
  BAZEL_TARGETS+=(
    "//tpu_raiden/frameworks/torch:_tpu_raiden_torch"
  )
else
  DEFINE_FLAGS+=" --define with_torch=false"
  DUMMY_TORCH_TPU_MODULE="${BAZEL_CACHE_BASE}/dummy_torch_tpu_module"
  mkdir -p "${DUMMY_TORCH_TPU_MODULE}"
  cat > "${DUMMY_TORCH_TPU_MODULE}/MODULE.bazel" <<'EOF'
module(name = "torch_tpu", version = "0.1.1")
EOF
  BAZEL_MODULE_FLAGS+=("--override_module=torch_tpu=${DUMMY_TORCH_TPU_MODULE}")
fi

if [ ${#BAZEL_TARGETS[@]} -eq 0 ]; then
  echo "No targets selected to build!"
  exit 1
fi

mkdir -p "${BAZEL_DISK_CACHE}" "${BAZEL_REPO_CACHE}" "$(dirname "${BAZEL_OUTPUT_BASE}")"

echo "=== Building targets with Bazel ==="
"${BAZEL_BIN}" --install_base="${BAZEL_OUTPUT_BASE}/install_base" --output_base="${BAZEL_OUTPUT_BASE}" --host_jvm_args="-Xmx32g" --host_jvm_args="-Xms2g" build -c opt --check_visibility=false --verbose_failures --experimental_repo_remote_exec --incompatible_disallow_empty_glob=false \
  --repo_env=HERMETIC_PYTHON_VERSION=${HERMETIC_PYTHON_VERSION:-3.12} \
  --repo_env=PIP_INDEX_URL="https://pypi.org/simple" \
  --repo_env=PIP_EXTRA_INDEX_URL="" \
  --repo_env=PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring" \
  --repo_env=PIP_CONFIG_FILE="/dev/null" \
  "${BAZEL_MODULE_FLAGS[@]}" \
  "${TORCH_REPO_ENV_FLAGS[@]}" \
  "${BAZEL_TARGETS[@]}" \
  ${DEFINE_FLAGS} \
  --disk_cache=${BAZEL_DISK_CACHE} \
  --repository_cache=${BAZEL_REPO_CACHE} \
  "$@"


echo "=== Copying generated protobuf Python modules ==="
cp -f "${WORKSPACE_DIR}/bazel-bin/weight_sync/weight_synchronizer_service_pb2.py" "${WORKSPACE_DIR}/weight_sync/" 2>/dev/null || true
cp -f "${WORKSPACE_DIR}/bazel-bin/rpc/coordination_pb2.py" "${WORKSPACE_DIR}/rpc/" 2>/dev/null || true
cp -f "${WORKSPACE_DIR}/bazel-bin/rpc/coordination_pb2_grpc.py" "${WORKSPACE_DIR}/rpc/" 2>/dev/null || true

echo "=== Copying compiled shared libraries to source directory ==="
if [ "$BUILD_JAX" = true ]; then
  echo "Copying JAX artifacts..."
  cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_raiden/frameworks/jax/_tpu_raiden_jax.so" "${WORKSPACE_DIR}/tpu_raiden/frameworks/jax/"
fi

if [ "$BUILD_TORCH" = true ]; then
  echo "Copying Torch artifacts..."
  TORCH_SO="${WORKSPACE_DIR}/tpu_raiden/frameworks/torch/_tpu_raiden_torch.so"
  cp -f "${WORKSPACE_DIR}/bazel-bin/tpu_raiden/frameworks/torch/_tpu_raiden_torch.so" "${TORCH_SO}"
  chmod u+w "${TORCH_SO}"
  # The torch extension statically links its own XLA and references a few
  # torch_tpu symbols (MaterializeAndReturn, AwaitBuffer). Add a NEEDED
  # dependency on libpywrap so those resolve in *local* scope at import time
  # (the loader imports the extension RTLD_LOCAL, see api/torch/
  # kv_cache_manager.py). This keeps raiden's XLA private and avoids the
  # duplicate AllocatorFactory registration that a global libpywrap preload
  # would trigger. torch_tpu must already be imported (libpywrap loaded) when
  # the extension imports, so no RUNPATH is required.
  if command -v patchelf > /dev/null; then
    patchelf --add-needed libpywrap_torch_tpu_common.so "${TORCH_SO}"
    echo "patchelf: added NEEDED libpywrap_torch_tpu_common.so to torch extension"
  else
    echo "ERROR: patchelf not found! The PyTorch extension requires patchelf" \
         "to inject NEEDED libpywrap_torch_tpu_common.so so symbols resolve in local" \
         "scope without duplicate XLA allocator crashes. Please install patchelf" \
         "(e.g., 'sudo apt-get install -y patchelf') and rebuild." >&2
    exit 1
  fi
fi


echo "=== Build Complete! ==="
if [ "$BUILD_JAX" = true ]; then
  echo "JAX Artifacts are located in: ${WORKSPACE_DIR}/tpu_raiden/frameworks/jax/"
fi
if [ "$BUILD_TORCH" = true ]; then
  echo "Torch Artifacts are located in: ${WORKSPACE_DIR}/tpu_raiden/frameworks/torch/"
fi

echo "=== Install Python Dependencies! ==="
pip install --index-url=https://pypi.org/simple -r requirements.txt || echo "Warning: pip installation returned a non-zero status. Proceeding anyway."

echo "=== Installation Complete! ==="
