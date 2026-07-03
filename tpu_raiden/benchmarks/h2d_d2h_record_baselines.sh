#!/bin/bash
# MUST run on the SAME hardware the gate runs on (v5e-8, runner
# linux-x86-ct5lp-224-8tpu); baselines/floors recorded elsewhere are meaningless.
#
# --record refreshes baseline_d2h/h2d AND floor_d2h/h2d in place; the config
# list, warmup, sigma_k, max_margin and the gate's iters are preserved.
#
# RECORD_ITERS is large on purpose: the floor is median - k*MAD_sigma, and the
# sigma estimate is noisy from few samples, so record from many iterations to
# get a stable floor. The gate itself still runs the smaller `iters` from the
# baselines file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINES="${SCRIPT_DIR}/gating_baselines.json"
BAZEL_CONFIG="${BAZEL_CONFIG:--c opt --config=oss --config=ci}"
RECORD_ITERS="${RECORD_ITERS:-1000}"   # many samples -> stable MAD/floor

cd "${WORKSPACE_DIR}"
echo "Recording baselines+floors (iters=${RECORD_ITERS}) into: ${BASELINES}"
bazel run ${BAZEL_CONFIG} \
  //tpu_raiden/benchmarks:h2d_d2h_benchmark_gating -- \
  --record --iters="${RECORD_ITERS}" --baselines="${BASELINES}"

echo "Done. Review before committing:"
echo "  git diff -- tpu_raiden/benchmarks/gating_baselines.json"














