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

"""Same-machine (COLOCATED) A/B distribution probe for the unsaturated canary
configs.

Same-machine is guaranteed by BAP, not by this binary: with `ab_strategy:
COLOCATED` in benchmark_registry_dist.pbtxt (available from BAP commit
e34d5b724ee5daac083881a891d25964081a9f6a onward), BAP provisions ONE runner and
runs the baseline then the experiment sequentially on the SAME host. So this
workload runs ONCE per invocation; BAP runs it twice (baseline + experiment). The
workflow points baseline_ref and experiment_ref at the SAME commit, so there is no
code change: the baseline run is "A", the experiment run is "B", and their delta
is the same-machine noise floor.

Each run writes its raw per-iter samples to $WORKLOAD_ARTIFACTS_DIR/dist_samples.json
(uploaded as that run's Artifacts) and its per-config medians to
TENSORBOARD_OUTPUT_DIR (so ab_analyzer can diff baseline vs experiment). Metric
tags here MUST match the metrics{name:...} entries in the registry.
"""
import json
import os

import numpy as np

from tpu_raiden.benchmarks import perf_core

ITERS = int(os.environ.get('ITERS', '50'))    # samples per run (one run = A or B)
WARMUP = int(os.environ.get('WARMUP', '5'))
SHARD = 2

# The unsaturated / overhead-bound canary configs (realistic KV-cache shape).
KV = [16, 128, 8, 2, 128]
CONFIGS = [
    ('int32 L64 kv', KV, 64, 'int32'),
    ('float32 L64 kv', KV, 64, 'float32'),
]


def main():
  print(f'canary distribution: {len(CONFIGS)} configs x {ITERS} iters '
        f'(warmup {WARMUP}, shard_axis={SHARD}). BAP runs this once as baseline '
        f'and once as experiment, COLOCATED on the same host.\n')
  dump = {}
  for label, shape, layers, dtype in CONFIGS:
    r = perf_core.measure(shape=shape, num_layers=layers, dtype=dtype,
                          shard_axis=SHARD, iters=ITERS, warmup=WARMUP)
    for d in ('d2h', 'h2d'):
      dump[f'{label}|{d}'] = r[f'{d}_gbps_all']   # raw samples for the histogram
    print(f"[{label}]  d2h {np.median(r['d2h_gbps_all']):.1f}  "
          f"h2d {np.median(r['h2d_gbps_all']):.1f} Gbps (median)", flush=True)

  # NOTE: no TensorBoard metrics -- the container image has neither tensorboardX
  # nor torch, so metric emission is impossible here. For this distribution probe
  # we only need the raw samples; we plot from dist_samples.json locally, not from
  # ab_analyzer. (The real ab-gate would need the image to add tensorboardX.)
  out = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out, 'dist_samples.json'), 'w') as f:
    json.dump(dump, f, indent=2)
  n = len(next(iter(dump.values())))
  print(f'\nsaved {n} samples per (config,dir) -> {out}/dist_samples.json')
  print('(BAP uploads this for BOTH the baseline and experiment runs; download '
        'both from the run page -> Artifacts and plot locally)')


if __name__ == '__main__':
  main()
