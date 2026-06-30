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
# vLLM server with the RaidenOffloadConnector: a single TPU engine that offloads
# its KV-cache prefix blocks to a host-RAM pool via tpu-raiden, and loads them
# back on a cache hit. Single process, kv_role=kv_both (saves and loads locally).
set -u
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tpu-raiden repo root: examples/kv_host_offloading/ -> ../.. (holds the source
# tree + the importable tpu_raiden package, for the build-from-source path).
RAIDEN_ROOT="${RAIDEN_ROOT:-$(cd "${SCRIPTS_DIR}/../.." && pwd)}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-info}"

# Make tpu_raiden importable (no-op if wheel-installed; adds the source tree to
# PYTHONPATH if built from source). vllm + tpu_inference are pip-installed, so they
# need no PYTHONPATH.
source "${SCRIPTS_DIR}/raiden_env.sh"

# --- Offloading knobs ---
# Host KV pool size, in blocks. The logical store capacity AND the physical host
# pool are both sized from this. Keep it ABOVE the benchmark's working set so the
# pool never needs to evict (see README).
export TPU_OFFLOAD_NUM_CPU_CHUNKS="${TPU_OFFLOAD_NUM_CPU_CHUNKS:-8192}"
# Also save KV produced during decode (not just prefill). Off by default.
export TPU_OFFLOAD_DECODE_SAVE="${TPU_OFFLOAD_DECODE_SAVE:-0}"

# --- Serving config (override via env) ---
MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

# Using RaidenOffloadConnector for offloading
KV_TRANSFER_CONFIG='{"kv_connector":"RaidenOffloadConnector","kv_connector_module_path":"tpu_inference.offload.raiden_offload_connector","kv_role":"kv_both"}'
echo "server.sh --> RaidenOffloadConnector | model=${MODEL} tp=${TENSOR_PARALLEL_SIZE} pool=${TPU_OFFLOAD_NUM_CPU_CHUNKS}"

# Offloading requires prefix caching ON 
exec env MODEL_IMPL_TYPE=vllm vllm serve "${MODEL}" \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  --port "${PORT}" \
  --enable-chunked-prefill \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --seed 42 \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --disable-hybrid-kv-cache-manager \
  --kv-cache-dtype fp8
