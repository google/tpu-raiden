#!/usr/bin/env bash
# Executed by `git bisect run` at each candidate commit.
# Env in: HARNESS (dir holding measure.py + BUILD.inject), THRESH (d2h Gbps cutoff).
# exit 0 = good(fast) | 1 = bad(regressed) | 125 = untestable(skip).
set -uo pipefail
cleanup() { rm -rf tpu_raiden/benchmarks/bisect_run; }
trap cleanup EXIT

mkdir -p tpu_raiden/benchmarks/bisect_run
cp "$HARNESS/measure.py"   tpu_raiden/benchmarks/bisect_run/measure.py
cp "$HARNESS/BUILD.inject" tpu_raiden/benchmarks/bisect_run/BUILD

# Separate output_base so the inner build never collides with the outer bazel.
BZL="bazel --output_base=/tmp/bisect_ob"
if ! $BZL build -c opt --config=oss //tpu_raiden/benchmarks/bisect_run:measure; then
  echo "[skip] build failed at $(git rev-parse --short HEAD)"; exit 125
fi
OUT=$($BZL run -c opt --config=oss //tpu_raiden/benchmarks/bisect_run:measure) \
  || { echo "[skip] run failed"; exit 125; }
echo "$OUT"
D2H=$(echo "$OUT" | sed -n 's/.*d2h=\([0-9.]*\).*/\1/p')
[ -z "$D2H" ] && { echo "[skip] no measurement"; exit 125; }

awk -v v="$D2H" -v t="${THRESH:-760}" 'BEGIN{exit !(v>=t)}'  # good if d2h >= THRESH
