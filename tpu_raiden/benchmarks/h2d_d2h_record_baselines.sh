#!/bin/bash
# <Apache 2.0 header>
#
# Regenerate gating_baselines.json for the h2d/d2h perf gate.
#
# MUST run on the SAME hardware the gate runs on (v5e-8, runner
# linux-x86-ct5lp-224-8tpu); baselines recorded elsewhere are meaningless.
# --record refreshes ONLY baseline_d2h/baseline_h2d in place -- threshold,
# iters, warmup and the config list are preserved.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINES="${SCRIPT_DIR}/gating_baselines.json"
# Match the gate's build config from benchmark_registry.pbtxt.
BAZEL_CONFIG="${BAZEL_CONFIG:--c opt --config=oss --config=ci}"

cd "${WORKSPACE_DIR}"
echo "Recording baselines into: ${BASELINES}"
bazel run ${BAZEL_CONFIG} \
  //tpu_raiden/benchmarks:h2d_d2h_benchmark_gating -- \
  --record --baselines="${BASELINES}"

echo "Done. Review before committing:"
echo "  git diff -- tpu_raiden/benchmarks/gating_baselines.json"














