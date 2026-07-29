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
#
# In-container build of a tpu_raiden wheel. Must run inside the ml-build
# container (glibc 2.35, matching the TPU runtime) — either via the docker
# wrapper ci/build_wheel.sh, or directly as a GitHub Actions job step whose
# job declares `container: ml-build:latest` (see .github/workflows/
# nightly_build.yml). Requires root (installs clang-18 via apt).
#
# Produces exactly one wheel per invocation, selected by WITH_TORCH:
#   WITH_TORCH=1 -> tpu_raiden_torch (needs a torch_tpu checkout)
#   WITH_TORCH=0 -> tpu_raiden_jax
# The wheel lands in ${REPO_ROOT}/dist/, scoped by WHEEL_VERSION_EXTRAS.
#
# Environment:
#   WITH_TORCH            1 (default) or 0
#   TORCH_TPU_SRC         torch_tpu checkout path (default /torch_tpu)
#   WHEEL_VERSION_EXTRAS  suffix appended to the pyproject.toml base version
#                         (default .dev<UTC timestamp>; set to the empty string
#                         for a stable-release wheel)
#   BAZEL_CACHE_DIR       bazel disk/repo cache root (default /cache)
#   BAZEL_OUTPUT_BASE     bazel output base (default ${BAZEL_CACHE_DIR}/output_base)
#   EXTRA_BAZEL_FLAGS     extra bazel flags, e.g. "--config=ci" for the RBE
#                         remote cache (space-separated; optional)

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WITH_TORCH="${WITH_TORCH:-1}"
TORCH_TPU_SRC="${TORCH_TPU_SRC:-/torch_tpu}"
# Default only when UNSET: an empty-but-set WHEEL_VERSION_EXTRAS is meaningful
# (a stable release build, whose wheel version is the bare pyproject version).
WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS-.dev$(date -u +%Y%m%d%H%M%S)}"
export WHEEL_VERSION_EXTRAS
export BAZEL_CACHE_DIR="${BAZEL_CACHE_DIR:-/cache}"
export BAZEL_OUTPUT_BASE="${BAZEL_OUTPUT_BASE:-${BAZEL_CACHE_DIR}/output_base}"
EXTRA_BAZEL_FLAGS="${EXTRA_BAZEL_FLAGS:-}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates patchelf >/dev/null
# Add the LLVM jammy-18 apt repo manually (the container's add-apt-repository is
# broken: python apt_pkg is missing for python3.12).
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
  > /etc/apt/sources.list.d/llvm18.list
apt-get update -qq
apt-get install -y -qq clang-18 >/dev/null
ln -sf /usr/bin/clang-18 /usr/bin/clang
ln -sf /usr/bin/clang++-18 /usr/bin/clang++
clang --version | head -1

if [[ "${WITH_TORCH}" == "1" ]]; then
  if [[ ! -f "${TORCH_TPU_SRC}/MODULE.bazel" ]]; then
    echo "ERROR: torch build needs a torch_tpu checkout at ${TORCH_TPU_SRC}" >&2
    echo "       set TORCH_TPU_SRC=<path> or WITH_TORCH=0 for a JAX-only wheel." >&2
    exit 1
  fi
  # raiden is built on top of torch_tpu: raiden's _tpu_raiden_torch.so and
  # torch_tpu's libpywrap_torch_tpu_common.so must resolve the SAME libtorch
  # symbols at runtime. So raiden MUST compile against the EXACT torch that
  # torch_tpu was built against — never a floating `torch>=X` specifier, which
  # drifts to the latest release and breaks ABI (e.g. torch 2.13.x drops
  # `torch::autograd::deleteNode`, which torch_tpu's libpywrap needs → dlopen
  # `undefined symbol` at import). torch_tpu's source of truth is its per-Python
  # requirements lock, which pins an exact `torch==VERSION+cpu`.
  PYTAG="$(python3 -c 'import sys;print(f"{sys.version_info.major}_{sys.version_info.minor}")')"
  TORCH_REQ_FILE="${TORCH_TPU_SRC}/requirements/requirements_${PYTAG}.txt"
  TORCH_PIN=""
  if [[ -f "${TORCH_REQ_FILE}" ]]; then
    # e.g. line `torch==2.11.0+cpu \` -> `torch==2.11.0+cpu`
    TORCH_PIN=$(sed -n -E 's/^(torch==[0-9][0-9A-Za-z.+_-]*).*/\1/p' "${TORCH_REQ_FILE}" | head -1 || true)
  fi
  if [[ -n "${TORCH_PIN}" ]]; then
    echo "Installing torch pinned by torch_tpu (${TORCH_REQ_FILE}): ${TORCH_PIN}"
    pip install -q "${TORCH_PIN}" --index-url https://download.pytorch.org/whl/cpu
  else
    # Fallback: the (looser) specifier from torch_tpu's pyproject.toml. This can
    # float to the latest release and may NOT match torch_tpu's ABI, so warn.
    TORCH_VERSION=""
    if [[ -f "${TORCH_TPU_SRC}/pyproject.toml" ]]; then
      TORCH_VERSION=$(sed -n -E 's/.*["'\''`]torch[[:space:]]*([>=<~=]+[0-9.a-zA-Z+-]+)["'\''`].*/\1/p' "${TORCH_TPU_SRC}/pyproject.toml" 2>/dev/null | head -1 || true)
    fi
    if [[ -z "${TORCH_VERSION}" ]]; then
      echo "WARNING: could not determine torch pin from ${TORCH_REQ_FILE} or ${TORCH_TPU_SRC}/pyproject.toml. Installing latest torch — this may NOT match torch_tpu's ABI." >&2
      pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    else
      echo "WARNING: no exact pin in ${TORCH_REQ_FILE}; falling back to torch_tpu pyproject specifier 'torch${TORCH_VERSION}', which may float to a torch that does not match torch_tpu's ABI." >&2
      pip install -q "torch${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
  TORCH_SOURCE="$(python3 -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).resolve().parent.parent)')"
  export TORCH_SOURCE
  export TORCH_TPU_MODULE_PATH="${TORCH_TPU_SRC}"
fi

# Separate per-framework wheels: tpu_raiden_torch (no jax deps) vs
# tpu_raiden_jax. Pick by WITH_TORCH.
if [[ "${WITH_TORCH}" == "1" ]]; then
  BUILD_MODE="both"
  WHEEL_TARGET="//ci/wheel:raiden_torch_wheel"
  WHEEL_DIST="tpu_raiden_torch"
else
  BUILD_MODE="jax"
  WHEEL_TARGET="//ci/wheel:raiden_jax_wheel"
  WHEEL_DIST="tpu_raiden_jax"
fi
# Match ONLY the wheel this build just produced. The bazel bin dir is shared
# across builds via the persistent cache/output base, so it accumulates wheels
# from earlier runs, each with a distinct .dev<timestamp>. A broad
# "${WHEEL_DIST}-*.whl" glob would also match those stale wheels and hand
# multiple paths to the single-wheel patchelf step below (which then fails).
# WHEEL_VERSION_EXTRAS (.dev<timestamp>) is unique per build and appears
# verbatim in the filename, so scope to it.
WHEEL_GLOB="${WHEEL_DIST}-*${WHEEL_VERSION_EXTRAS}-*.whl"

./build.sh "${BUILD_MODE}" "${WHEEL_TARGET}" \
  --repo_env=WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  ${EXTRA_BAZEL_FLAGS}

mkdir -p "${REPO_ROOT}/dist"
cp "${REPO_ROOT}"/bazel-bin/ci/wheel/${WHEEL_GLOB} "${REPO_ROOT}/dist/"

# The bazel-built _tpu_raiden_torch.so does not link libpywrap; the torch
# extension loader (tpu_raiden/api/torch/kv_cache_manager.py) requires it as a
# NEEDED so the torch_tpu symbols (MaterializeAndReturn, AwaitBuffer, ...)
# resolve in RTLD_LOCAL scope at import. build.sh injects this for its
# source-tree copy, but the wheel packages the raw bazel .so -- so inject it
# into the wheel here and repack (which regenerates RECORD with valid hashes).
if [[ "${WITH_TORCH}" == "1" ]]; then
  pip install -q wheel
  WHL="$(ls "${REPO_ROOT}"/dist/${WHEEL_GLOB} | head -1)"
  UNPACK_DIR="$(mktemp -d)"
  wheel unpack "${WHL}" -d "${UNPACK_DIR}"
  PKG_DIR="$(ls -d "${UNPACK_DIR}"/*/)"
  patchelf --add-needed libpywrap_torch_tpu_common.so \
    "${PKG_DIR}tpu_raiden/frameworks/torch/_tpu_raiden_torch.so"
  rm -f "${WHL}"
  wheel pack "${PKG_DIR}" -d "${REPO_ROOT}/dist"
  echo "patchelf: injected NEEDED libpywrap_torch_tpu_common.so into wheel .so"
fi

echo "===> Wheel(s) built:"
ls -lh "${REPO_ROOT}"/dist/${WHEEL_GLOB}
