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
# Serving benchmark client for Raiden offloading prefix caching performance.
# Sends concurrent requests with a shared 32,000 token prefix.
set -eu

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIDEN_ROOT="${RAIDEN_ROOT:-$(cd "${SCRIPTS_DIR}/../.." && pwd)}"

echo "launch_benchmark.sh --> Running serving benchmark with 32k shared prefix"

# Run serving benchmark with 32k shared prefix and 128 output tokens
exec python3 "${RAIDEN_ROOT}/../tpu-inference/scripts/vllm/benchmarking/benchmark_serving.py" \
  --backend vllm \
  --model Qwen/Qwen3-32B \
  --dataset-name random \
  --random-prefix-len 32000 \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 75 \
  --request-rate 4 \
  --seed 42
