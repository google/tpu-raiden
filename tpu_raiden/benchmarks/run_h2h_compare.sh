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

#!/usr/bin/env bash
# Single-host cross-socket H2H: C++ (h2h_benchmark_runner) vs Python (push) side
# by side across the SAME setups, to check the measured bandwidth agrees. Both
# report GB/s (bytes) so the numbers are directly comparable (no 8x unit fixup).
#
# Run ON the TPU runner, from the repo root, after building both binaries. The
# C++ target is co-located with its source under tpu_raiden/examples/... (no file
# move needed). If jaxlib trips -Wbitwise-instead-of-logical, add
# --copt=-Wno-error=bitwise-instead-of-logical to the build.
#
#   bazel build -c opt --config=oss \
#     //tpu_raiden/examples/microbenchmarks:h2h_benchmark_runner \
#     //tpu_raiden/benchmarks:h2h_perf_test_oss
#   bash tpu_raiden/benchmarks/run_h2h_compare.sh
#
# Both use loopback (--data_interface=lo / 127.0.0.1) so neither egresses a NIC:
# pure cross-socket memory + net-stack (apples-to-apples, B-tier).
set -u

CC=bazel-bin/tpu_raiden/examples/microbenchmarks/h2h_benchmark_runner
PY=bazel-bin/tpu_raiden/benchmarks/h2h_perf_test_oss
SENDER_NUMA=0
RECEIVER_NUMA=1
WARMUP=3
ITERS=50
LAYERS=32       # C++ kNumLayers (fixed)
SHARDS=1        # C++ kNumShards (fixed)
PORT=9990

# Sweep: "block_size_bytes num_blocks parallelism"  (defaults mirror the C++ ones)
CONFIGS=(
  "1048576   64  1"     # 1 MB  x64  P1   (C++ default block_size, P1)
  "1048576   64  8"     # 1 MB  x64  P8
  "16777216  64  1"     # 16 MB x64  P1
  "16777216  64  8"     # 16 MB x64  P8
  "134217728 64  8"     # 128 MB x64 P8  (H2H.md example)
)

grab() { grep -oE "$2" "$1" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | tail -1; }

printf "%-11s %-6s %-4s | %10s %10s %8s\n" "block(B)" "blocks" "P" "cpp_GB/s" "py_GB/s" "py/cpp"
printf -- "------------------------------------------------------------------\n"

for cfg in "${CONFIGS[@]}"; do
  read -r BS NB P <<<"$cfg"
  PORT=$((PORT + 1))
  rp=$(mktemp); sp=$(mktemp); pp=$(mktemp)

  # ---- C++ : receiver (bg) + sender (timed, prints Throughput GB/s) ----
  "$CC" --role=receiver --data_interface=lo --peer_control_port="$PORT" \
        --numa_node="$RECEIVER_NUMA" --block_size="$BS" --num_blocks="$NB" \
        --parallelism="$P" >"$rp" 2>&1 &
  RECV=$!
  sleep 3   # let the control server bind
  "$CC" --role=sender --peer_control_ip=127.0.0.1 --data_interface=lo \
        --peer_control_port="$PORT" --numa_node="$SENDER_NUMA" \
        --block_size="$BS" --num_blocks="$NB" --parallelism="$P" >"$sp" 2>&1
  wait "$RECV" 2>/dev/null
  CPP=$(grab "$sp" 'Throughput:[[:space:]]*[0-9.]+')

  # ---- Python : launcher (push, prints Throughput (mean) GB/s) ----
  "$PY" --role=launch --sender_numa="$SENDER_NUMA" --receiver_numa="$RECEIVER_NUMA" \
        --num_layers="$LAYERS" --num_shards="$SHARDS" --block_size="$BS" \
        --num_blocks="$NB" --parallelism="$P" --warmup="$WARMUP" --iters="$ITERS" \
        >"$pp" 2>&1
  PYV=$(grab "$pp" 'Throughput \(mean\):[[:space:]]*[0-9.]+')

  RATIO=$(awk -v a="$PYV" -v b="$CPP" 'BEGIN{ if(b>0 && a!="") printf "%.2f", a/b; else print "NA" }')
  printf "%-11s %-6s %-4s | %10s %10s %8s\n" "$BS" "$NB" "$P" "${CPP:-NA}" "${PYV:-NA}" "$RATIO"
  rm -f "$rp" "$sp" "$pp"
done
