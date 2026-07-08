#!/bin/bash
# Record C++ H2H baselines LOCALLY into the source tree, then review + commit.
#
# MUST run on the SAME hardware the gate runs on (v5e-8, runner
# linux-x86-ct5lp-224-8tpu); a baseline recorded elsewhere is meaningless.
#
# This writes measured median GB/s (and the integrity flag) for the two 1MB
# configs into h2h_cpp_baselines.json IN PLACE. The gate then uses each value,
# minus the margin, as the floor.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINES="${SCRIPT_DIR}/h2h_cpp_baselines.json"
BAZEL_CONFIG="${BAZEL_CONFIG:--c opt --config=oss --config=ci}"

cd "${WORKSPACE_DIR}"
echo "Recording C++ H2H baselines into: ${BASELINES}"
bazel run ${BAZEL_CONFIG} \
  //tpu_raiden/benchmarks:h2h_cpp_gate -- \
  --record --baselines="${BASELINES}"

echo "Done. Review before committing:"
echo "  git diff -- tpu_raiden/benchmarks/h2h_cpp_baselines.json"
