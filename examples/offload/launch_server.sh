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
# Launch script for high-performance Raiden KV Cache Offloading server.
# Sourced/Executed within a Google3 JAX TPU environment.
set -eu

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Expose JAX and Raiden paths
RAIDEN_ROOT="${RAIDEN_ROOT:-$(cd "${SCRIPTS_DIR}/../.." && pwd)}"

# Enforce a safe memory limit (8000 blocks = 512 GB host cache pool)
# This leaves an enormous safety headroom on 1.4 TiB physical RAM hosts.
export TPU_OFFLOAD_NUM_CPU_CHUNKS=${TPU_OFFLOAD_NUM_CPU_CHUNKS:-8000}

# Locate the compiled FFI shared library for LD_PRELOAD.
# This is mandatory to prevent Protobuf descriptor double-registration segfaults.
SO_PATH="${RAIDEN_ROOT}/bazel-bin/third_party/tpu_raiden/tpu_raiden/frameworks/jax/_tpu_raiden_jax.so"
if [ ! -f "${SO_PATH}" ]; then
  # Fallback to legacy path or build output
  SO_PATH="${RAIDEN_ROOT}/bazel-bin/third_party/tpu_raiden/tpu_raiden/frameworks/jax/_kv_cache_manager.so"
fi

if [ ! -f "${SO_PATH}" ]; then
  echo "ERROR: Compiled JAX FFI shared library not found in bazel-bin." >&2
  echo "Please build it first from the repo root using:" >&2
  echo "  blaze build //third_party/tpu_raiden/tpu_raiden/frameworks/jax:_tpu_raiden_jax" >&2
  exit 1
fi

export LD_PRELOAD="${SO_PATH}"
echo "launch_server.sh --> Launching vLLM with Raiden Offloading"
echo "LD_PRELOAD: ${LD_PRELOAD}"
echo "TPU_OFFLOAD_NUM_CPU_CHUNKS: ${TPU_OFFLOAD_NUM_CPU_CHUNKS}"

# Launch vLLM serve sharded across 8 TPUs
exec vllm serve Qwen/Qwen3-32B \
  --kv-transfer-config '{"kv_connector":"RaidenOffloadConnector","kv_connector_module_path":"tpu_inference.offload.raiden_offload_connector","kv_role":"kv_both"}' \
  --port 8000 \
  --enable-chunked-prefill \
  --tensor-parallel-size 8 \
  --seed 42 \
  --enable_prefix_caching \
  --gpu-memory-utilization 0.8 \
  --disable-hybrid-kv-cache-manager \
  --kv-cache-dtype fp8
