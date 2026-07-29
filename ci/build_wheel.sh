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

# The in-container build (clang-18 install + pinned torch + build.sh wheel
# target) lives in ci/build_wheel_impl.sh so GitHub Actions jobs that already
# run inside the ml-build container can invoke it without docker-in-docker.
echo "===> Building ${BUILD_MODE} wheel in ${CONTAINER_IMAGE}..."
docker run --rm \
  "${DOCKER_MOUNTS[@]}" \
  -w /workspace \
  -e WHEEL_VERSION_EXTRAS="${WHEEL_VERSION_EXTRAS}" \
  -e WITH_TORCH="${WITH_TORCH}" \
  -e TORCH_TPU_SRC=/torch_tpu \
  -e BAZEL_CACHE_DIR=/cache \
  -e EXTRA_BAZEL_FLAGS="${EXTRA_BAZEL_FLAGS:-}" \
  "${CONTAINER_IMAGE}" \
  bash ci/build_wheel_impl.sh

# Scope to THIS build's wheel(s) (.dev<timestamp>); REPO_ROOT/dist and WHEEL_DIR
# are persistent and may hold wheels from earlier runs.
if [[ -n "$(ls -A "${REPO_ROOT}"/dist/*"${WHEEL_VERSION_EXTRAS}"-*.whl 2>/dev/null)" ]]; then
  cp "${REPO_ROOT}"/dist/*"${WHEEL_VERSION_EXTRAS}"-*.whl "${WHEEL_DIR}/"
  echo "===> Wheel(s) built:"; ls -lh "${WHEEL_DIR}"/*"${WHEEL_VERSION_EXTRAS}"-*.whl
else
  echo "ERROR: wheel build produced no .whl for ${WHEEL_VERSION_EXTRAS} in dist/" >&2; exit 1
fi

echo "===> twine check..."
docker run --rm -v "${WHEEL_DIR}:/dist" "${CONTAINER_IMAGE}" \
  bash -c "uv run --isolated --with twine twine check /dist/*${WHEEL_VERSION_EXTRAS}-*.whl"
echo "===> raiden wheel build successful!"
