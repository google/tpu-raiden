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

"""Run all 10 micro-benchmark configs x N iters and report the distribution.

Prints per-config median/mean/std/CV%/min/max for d2h & h2d, and saves EVERY
raw per-iter sample to dist_samples.json + dist_samples.csv (for histograms).
CURRENT checkout only, so it runs in BAP like the roofline sweep.

The result files land in $WORKLOAD_ARTIFACTS_DIR (a temp dir BAP sets on the
runner and uploads as the run's Artifacts) -- they are NOT created on your local
machine. Download them from the workflow run page -> Artifacts.

Edit ITERS below to change the sample count.
"""
import json
import os
import sys

import numpy as np

from tpu_raiden.benchmarks import perf_core

ITERS = int(os.environ.get('ITERS', '200'))   # <-- change the sample count here
WARMUP = int(os.environ.get('WARMUP', '5'))
SHARD = 2

KV = [16, 128, 8, 2, 128]      # KV-cache-shaped (small block)
LG = [8, 128, 1024, 128]       # large synthetic shape (big block)
CONFIGS = [
    ('int32 L1 kv', KV, 1, 'int32'),
    ('int32 L64 kv', KV, 64, 'int32'),
    ('int32 L128 kv', KV, 128, 'int32'),
    ('float32 L64 kv', KV, 64, 'float32'),
    ('bf16 L64 kv', KV, 64, 'bfloat16'),
    ('fp8 L64 kv', KV, 64, 'float8_e4m3fn'),
    ('int32 L1 lg', LG, 1, 'int32'),
    ('float32 L1 lg', LG, 1, 'float32'),
    ('bf16 L1 lg', LG, 1, 'bfloat16'),
    ('fp8 L1 lg', LG, 1, 'float8_e4m3fn'),
]


def _stats(a):
  a = np.asarray(a, float)
  m = a.mean()
  return dict(median=float(np.median(a)), mean=float(m), std=float(a.std(ddof=1)),
              cv=float(a.std(ddof=1) / m * 100), lo=float(a.min()), hi=float(a.max()))


def main():
  print(f'distribution sweep: {len(CONFIGS)} configs x {ITERS} iters '
        f'(warmup {WARMUP}, shard_axis={SHARD})\n')
  hdr = (f"{'config':16}{'dir':4}{'median':>8}{'mean':>8}{'std':>7}"
         f"{'CV%':>6}{'min':>8}{'max':>8}")
  print(hdr)
  print('-' * len(hdr))
  dump = {}
  for label, shape, layers, dtype in CONFIGS:
    r = perf_core.measure(shape=shape, num_layers=layers, dtype=dtype,
                          shard_axis=SHARD, iters=ITERS, warmup=WARMUP)
    for d in ('d2h', 'h2d'):
      s = _stats(r[f'{d}_gbps_all'])
      print(f"{label:16}{d:4}{s['median']:8.1f}{s['mean']:8.1f}{s['std']:7.1f}"
            f"{s['cv']:6.1f}{s['lo']:8.1f}{s['hi']:8.1f}", flush=True)
      dump[f'{label}|{d}'] = r[f'{d}_gbps_all']
    print()

  # ---- save EVERY raw sample (to the artifacts dir BAP uploads) ----
  out = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out, 'dist_samples.json'), 'w') as f:
    json.dump(dump, f, indent=2)
  with open(os.path.join(out, 'dist_samples.csv'), 'w') as f:
    f.write('config,dir,iter,gbps\n')
    for key, vals in dump.items():
      cfg, d = key.split('|')
      for i, v in enumerate(vals):
        f.write(f'{cfg},{d},{i},{v:.3f}\n')
  n = len(next(iter(dump.values())))
  print(f'\nsaved ALL {n} samples/config -> {out}/dist_samples.json  &  dist_samples.csv')
  print('(download these from the workflow run page -> Artifacts)')


if __name__ == '__main__':
  main()
