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

"""Characterizes same-machine drift for the unsaturated (canary) KV configs.

Answers the question the post-submit A/B test cannot answer about itself: with
NO code change at all, how far apart do two measurements on one host land? That
spread is the noise floor, and the A/B regression threshold has to sit above its
tail or the test cries wolf.

Deliberately NOT a BAP ab_mode workload. ab_mode yields exactly one delta per CI
run, so characterizing a tail would take dozens of runs over days. This instead
measures the same (unchanged) code in a loop on one host and records the whole
time series, from which every quantity of interest is derived offline:

  * delta vs time gap  -- compare rounds separated by lag k; the wall-clock gap
    between consecutive rounds is one measurement, the same order as the gap
    BAP's COLOCATED mode puts between baseline and experiment.
  * delta vs iters     -- each round keeps its raw per-iteration samples, so a
    smaller `iters` is simulated by resampling instead of re-running.

Always exits 0: this characterizes, it does not gate.
"""

import json
import os
import statistics
import sys
import time

from absl import app
from absl import flags

from tpu_raiden.benchmarks import bap_metrics
from tpu_raiden.benchmarks import perf_core

_ROUNDS = flags.DEFINE_integer(
    'rounds', 20,
    'Measurement rounds per config. Each round is one point in the time series; '
    'deltas at every lag are derived from it offline, so N rounds give O(N) '
    'samples per lag rather than the single delta an ab_mode run would yield.')
_ITERS = flags.DEFINE_integer(
    'iters', 50,
    'Timed iterations per measurement. Record HIGH: the raw samples are kept, so '
    'the offline analysis can subsample down to simulate any smaller iters, but '
    'it can never invent samples that were not taken.')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Untimed iterations before each measurement.')
_DUMP = flags.DEFINE_string(
    'dump', 'canary_drift.json',
    'Raw time series (medians + every per-iteration sample). Written into '
    'WORKLOAD_ARTIFACTS_DIR on BAP so the workflow uploads it.')

# The overhead-bound KV shapes: small blocks, so per-op software cost dominates
# and the PCIe/DMA ceiling does not pin the number down. That is exactly why they
# drift, why the static-floor presubmit gate excludes them, and why they need a
# same-machine A/B instead.
_KV_SHAPE = [16, 128, 8, 2, 128]
_SHARD_AXIS = 2  # _KV_SHAPE[2] == 8, divisible by the 8 chips on a v5e node.
_CONFIGS = [
    ('int32', 64),
    ('float32', 64),
]


def _label(dtype, num_layers):
  return f'{dtype}_L{num_layers}_kv'


def main(_):
  rounds, iters, warmup = _ROUNDS.value, _ITERS.value, _WARMUP.value
  print(f'[drift] {len(_CONFIGS)} config(s) x {rounds} round(s), '
        f'iters={iters} warmup={warmup}', flush=True)

  series = []  # one entry per (round, config)
  t_start = time.time()
  for rnd in range(rounds):
    for dtype, num_layers in _CONFIGS:
      label = _label(dtype, num_layers)
      t0 = time.time()
      r = perf_core.measure(shape=_KV_SHAPE, num_layers=num_layers, dtype=dtype,
                            shard_axis=_SHARD_AXIS, iters=iters, warmup=warmup)
      series.append({
          'round': rnd,
          'label': label,
          'dtype': dtype,
          'num_layers': num_layers,
          # Wall clock is the point of this probe: drift is a function of time,
          # so every round is stamped and the offline analysis reads gaps off it.
          't_start': t0,
          't_end': time.time(),
          'd2h_gbps': r['d2h_gbps'],
          'h2d_gbps': r['h2d_gbps'],
          'd2h_gbps_all': r['d2h_gbps_all'],
          'h2d_gbps_all': r['h2d_gbps_all'],
      })
      print(f'[drift] round {rnd + 1}/{rounds} {label:<16} '
            f'd2h {r["d2h_gbps"]:7.1f}  h2d {r["h2d_gbps"]:7.1f} Gbps  '
            f'({time.time() - t0:.0f}s)', flush=True)

  # Raw series -> artifact. Everything downstream is computed from this file, so
  # dump the per-iteration samples, not just the medians.
  out = _DUMP.value
  adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  if adir:
    out = os.path.join(adir, os.path.basename(out))
  with open(out, 'w') as f:
    json.dump({'rounds': rounds, 'iters': iters, 'warmup': warmup,
               'shape': _KV_SHAPE, 'shard_axis': _SHARD_AXIS,
               'series': series}, f, indent=2)
  print(f'\n[drift] wrote {len(series)} measurements -> {out}', flush=True)

  # Hand BAP the whole per-round series rather than one summary number, so the
  # dashboard shows the spread this probe exists to expose.
  scalars = {}
  for dtype, num_layers in _CONFIGS:
    label = _label(dtype, num_layers)
    rows = [s for s in series if s['label'] == label]
    scalars[f'{label}/d2h_gbps'] = [s['d2h_gbps'] for s in rows]
    scalars[f'{label}/h2d_gbps'] = [s['h2d_gbps'] for s in rows]
  bap_metrics.emit(scalars)

  # Console preview of the headline number: the round-to-round spread at lag 1,
  # which is the closest single stand-in for what a COLOCATED A/B would see.
  print(f'\nlag-1 |delta| (the noise a same-machine A/B would face), '
        f'iters={iters}:\n')
  print(f'{"config":<18}{"dir":<5}{"median":>9}{"p95":>9}{"max":>9}')
  print('-' * 50)
  for dtype, num_layers in _CONFIGS:
    label = _label(dtype, num_layers)
    rows = [s for s in series if s['label'] == label]
    for d in ('d2h', 'h2d'):
      xs = [s[f'{d}_gbps'] for s in rows]
      deltas = [abs(b - a) / a * 100 for a, b in zip(xs, xs[1:]) if a > 0]
      if not deltas:
        continue
      deltas.sort()
      p95 = deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))]
      print(f'{label:<18}{d:<5}{statistics.median(deltas):8.2f}%'
            f'{p95:8.2f}%{max(deltas):8.2f}%')
  print('\n(full lag / iters sweeps come from the dumped series, offline)')

  # Characterization only -- never fail the build.
  sys.exit(0)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
