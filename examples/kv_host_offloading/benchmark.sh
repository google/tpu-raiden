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
# Serving benchmark for the offload server. Sends requests sharing a large prefix
# so that, after the first request offloads the prefix KV to host RAM, later
# requests hit the host cache and load it back. MODEL must match server.sh.
#
# Uses tpu-inference's benchmark_serving.py. Requires the `evaluate` and `nltk`
# packages in the venv (imported by that script's utils).
set -u

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
RANDOM_PREFIX_LEN="${RANDOM_PREFIX_LEN:-32000}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-75}"
REQUEST_RATE="${REQUEST_RATE:-4}"

# Path to tpu-inference's benchmark_serving.py. Defaults to the copy inside the
# installed tpu_inference repo; override with BENCH_SCRIPT if needed. (A sentinel
# prefix is used because importing tpu_inference prints banner logs to stdout.)
if [ -z "${BENCH_SCRIPT:-}" ]; then
  BENCH_SCRIPT="$(python -c 'import os, tpu_inference; print("BENCHPATH:" + os.path.join(os.path.dirname(os.path.dirname(tpu_inference.__file__)), "scripts/vllm/benchmarking/benchmark_serving.py"))' 2>/dev/null | sed -n 's/^BENCHPATH://p')"
fi
if [ -z "${BENCH_SCRIPT}" ] || [ ! -f "${BENCH_SCRIPT}" ]; then
  echo "ERROR: could not find tpu-inference's benchmark_serving.py." >&2
  echo "       Set BENCH_SCRIPT=/path/to/benchmark_serving.py" >&2
  exit 1
fi

echo "benchmark.sh --> ${RANDOM_PREFIX_LEN}-token shared-prefix benchmark | model=${MODEL}"
echo "  benchmark script: ${BENCH_SCRIPT}"

exec python3 "${BENCH_SCRIPT}" \
  --backend vllm \
  --model "${MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dataset-name random \
  --random-prefix-len "${RANDOM_PREFIX_LEN}" \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --random-output-len "${RANDOM_OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --request-rate "${REQUEST_RATE}" \
  --seed 42
