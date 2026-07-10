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

"""Same-machine A/B *distribution* probe for the unsaturated canary configs.

Purpose: produce the data for the "same-machine A/B distribution" plot, using
BAP's DOCUMENTED same-machine mechanism (not a hand-rolled trick).

SAME-MACHINE IS GUARANTEED BY BAP, NOT BY THIS BINARY
-----------------------------------------------------
Per the BAP onboarding doc:
  - "By default, BAP's A/B mode generates parallel jobs that run the baseline and
     experiment workloads on separate physical machines."
  - With `ab_strategy: COLOCATED` (set in benchmark_registry_dist.pbtxt): "the
     platform provisions a single runner and executes baseline followed by
     experiment sequentially on the same host."
So this workload runs ONCE per invocation; BAP (ab_mode: true + COLOCATED) runs it
twice -- baseline then experiment -- on the SAME host. In the workflow we point
baseline_ref and experiment_ref at the SAME commit, so there is no code change:
the two same-host runs are the "A" and "B" of the distribution, and their delta
is the same-machine noise floor.

Each run writes its raw per-iter samples to $WORKLOAD_ARTIFACTS_DIR/dist_samples.json,
which BAP uploads as that run's Artifacts (baseline vs experiment separately).
Download both and plot locally.
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


def _tb_label(dtype, layers):
  # Clean tag (no spaces / pipes) so it is a valid TensorBoard tag AND matches the
  # metrics{name:...} entries in benchmark_registry_dist.pbtxt exactly.
  return f'{dtype}_L{layers}_kv'


def _write_tb(medians):
  """Per-config median -> TENSORBOARD_OUTPUT_DIR so BAP's ab_analyzer can diff the
  baseline run against the experiment run (same host, COLOCATED)."""
  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tbdir:
    print('TENSORBOARD_OUTPUT_DIR not set; ab_analyzer will have no metrics.')
    return
  try:
    import tensorboardX  # pylint: disable=g-import-not-at-top
    w = tensorboardX.SummaryWriter(log_dir=tbdir)
  except ImportError:
    import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
    w = tut.SummaryWriter(log_dir=tbdir)
  for tag, val in medians.items():
    w.add_scalar(tag, val, global_step=0)
  w.close()
  print(f'wrote {len(medians)} TB metrics -> {tbdir}')


def main():
  print(f'canary distribution: {len(CONFIGS)} configs x {ITERS} iters '
        f'(warmup {WARMUP}, shard_axis={SHARD}). BAP runs this once as baseline '
        f'and once as experiment, COLOCATED on the same host.\n')
  dump = {}
  medians = {}
  for label, shape, layers, dtype in CONFIGS:
    r = perf_core.measure(shape=shape, num_layers=layers, dtype=dtype,
                          shard_axis=SHARD, iters=ITERS, warmup=WARMUP)
    tb = _tb_label(dtype, layers)
    for d in ('d2h', 'h2d'):
      vals = r[f'{d}_gbps_all']
      dump[f'{label}|{d}'] = vals              # raw samples for the histogram
      medians[f'{tb}/{d}_gbps'] = float(np.median(vals))  # median for ab_analyzer
    print(f"[{label}]  d2h {np.median(r['d2h_gbps_all']):.1f}  "
          f"h2d {np.median(r['h2d_gbps_all']):.1f} Gbps (median)", flush=True)

  _write_tb(medians)  # median per config -> ab_analyzer compares baseline vs experiment
  out = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out, 'dist_samples.json'), 'w') as f:
    json.dump(dump, f, indent=2)             # every raw sample -> the histogram
  n = len(next(iter(dump.values())))
  print(f'\nsaved {n} samples per (config,dir) -> {out}/dist_samples.json')
  print('(BAP uploads this for BOTH the baseline and experiment runs; download '
        'both from the run page -> Artifacts and plot locally)')


if __name__ == '__main__':
  main()
