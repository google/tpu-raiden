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
# Builds the tpu_raiden wheel hermetically inside the ml-build container
# (glibc 2.35, matching the TPU runtime), mirroring torch_tpu/ci/build_wheel.sh.
#
# Unlike torch_tpu, raiden needs (a) clang-18 for XLA's .ll codegen targets and
# (b) a local torch for the torch_tpu shim headers, plus the torch_tpu module.
# This script installs clang-18 + CPU torch into the container, mounts a sibling
# torch_tpu checkout, and runs build.sh for the wheel target.
#
# Usage:
#   ci/build_wheel.sh                  # JAX + Torch (needs ../torch_tpu)
#   WITH_TORCH=0 ci/build_wheel.sh     # JAX only
#   TORCH_TPU_SRC=/path ci/build_wheel.sh

set -exu -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WHEEL_VERSION_EXTRAS=".dev$(date +%Y%m%d%H%M%S)"
export WHEEL_VERSION_EXTRAS
echo "WHEEL_VERSION_EXTRAS: ${WHEEL_VERSION_EXTRAS}"

WITH_TORCH="${WITH_TORCH:-1}"
TORCH_TPU_SRC="${TORCH_TPU_SRC:-${REPO_ROOT}/../torch_tpu}"
WHEEL_DIR="${KOKORO_ARTIFACTS_DIR:-${HOME}/raiden_artifacts}/dist"
CACHE_DIR="${RAIDEN_CONTAINER_CACHE:-${HOME}/.bazel_cache_container}"
mkdir -p "${WHEEL_DIR}" "${REPO_ROOT}/dist" "${CACHE_DIR}"

CONTAINER_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest"
echo "===> Pulling ${CONTAINER_IMAGE}..."
docker pull "${CONTAINER_IMAGE}"

DOCKER_MOUNTS=(
  -v "${REPO_ROOT}:/workspace"
  -v "${CACHE_DIR}:/cache"
)
BUILD_MODE="jax"
if [[ "${WITH_TORCH}" == "1" ]]; then
  if [[ ! -f "${TORCH_TPU_SRC}/MODULE.bazel" ]]; then
    echo "ERROR: torch build needs a torch_tpu checkout at ${TORCH_TPU_SRC}" >&2
    echo "       set TORCH_TPU_SRC=<path> or WITH_TORCH=0 for a JAX-only wheel." >&2
    exit 1
  fi
  TORCH_TPU_SRC="$(cd "${TORCH_TPU_SRC}" && pwd)"
  DOCKER_MOUNTS+=(-v "${TORCH_TPU_SRC}:/torch_tpu")  # sibling ../torch_tpu == /torch_tpu
  BUILD_MODE="both"
fi

# The in-container build: install clang-18 (XLA .ll targets) + CPU torch (shim
# headers), then drive the existing build.sh for the wheel target.
read -r -d '' INNER <<'INNER_EOF' || true
set -exu -o pipefail
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
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  TORCH_SOURCE="$(python3 -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).resolve().parent.parent)')"
  export TORCH_SOURCE
  export TORCH_TPU_MODULE_PATH=/torch_tpu
fi

# Persistent, resumable bazel cache + output base on the mounted volume.
export BAZEL_CACHE_DIR=/cache
export BAZEL_OUTPUT_BASE=/cache/output_base

# Separate per-framework wheels: tpu_raiden_torch (no jax deps) vs
# tpu_raiden_jax. Pick by WITH_TORCH.
if [[ "${WITH_TORCH}" == "1" ]]; then
  WHEEL_TARGET="//ci/wheel:raiden_torch_wheel"
  WHEEL_GLOB="tpu_raiden_torch-*.whl"
else
  WHEEL_TARGET="//ci/wheel:raiden_jax_wheel"
  WHEEL_GLOB="tpu_raiden_jax-*.whl"
fi

cd /workspace
./build.sh "${BUILD_MODE}" "${WHEEL_TARGET}" \
  --repo_env=WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}"

mkdir -p /workspace/dist
cp /cache/output_base/execroot/_main/bazel-out/k8-opt/bin/ci/wheel/${WHEEL_GLOB} /workspace/dist/

# The bazel-built _tpu_raiden_torch.so does not link libpywrap; the torch
# extension loader (tpu_raiden/api/torch/kv_cache_manager.py) requires it as a
# NEEDED so the torch_tpu symbols (MaterializeAndReturn, AwaitBuffer, ...)
# resolve in RTLD_LOCAL scope at import. build.sh injects this for its
# source-tree copy, but the wheel packages the raw bazel .so -- so inject it
# into the wheel here and repack (which regenerates RECORD with valid hashes).
if [[ "${WITH_TORCH}" == "1" ]]; then
  pip install -q wheel
  WHL="$(ls /workspace/dist/${WHEEL_GLOB})"
  UNPACK_DIR="$(mktemp -d)"
  wheel unpack "${WHL}" -d "${UNPACK_DIR}"
  PKG_DIR="$(ls -d "${UNPACK_DIR}"/*/)"
  patchelf --add-needed libpywrap_torch_tpu_common.so \
    "${PKG_DIR}tpu_raiden/frameworks/torch/_tpu_raiden_torch.so"
  rm -f "${WHL}"
  wheel pack "${PKG_DIR}" -d /workspace/dist
  echo "patchelf: injected NEEDED libpywrap_torch_tpu_common.so into wheel .so"
fi
INNER_EOF

echo "===> Building ${BUILD_MODE} wheel in ${CONTAINER_IMAGE}..."
docker run --rm \
  "${DOCKER_MOUNTS[@]}" \
  -w /workspace \
  -e WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  -e WITH_TORCH="${WITH_TORCH}" \
  -e BUILD_MODE="${BUILD_MODE}" \
  "${CONTAINER_IMAGE}" \
  bash -c "${INNER}"

if [[ -n "$(ls -A "${REPO_ROOT}"/dist/*.whl 2>/dev/null)" ]]; then
  cp "${REPO_ROOT}"/dist/*.whl "${WHEEL_DIR}/"
  echo "===> Wheel(s) built:"; ls -lh "${WHEEL_DIR}"
else
  echo "ERROR: wheel build produced no .whl in dist/" >&2; exit 1
fi

echo "===> twine check..."
docker run --rm -v "${WHEEL_DIR}:/dist" "${CONTAINER_IMAGE}" \
  bash -c "uv run --isolated --with twine twine check /dist/*.whl"
echo "===> raiden wheel build successful!"
