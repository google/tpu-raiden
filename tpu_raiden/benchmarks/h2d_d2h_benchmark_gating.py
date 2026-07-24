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

import json
import os
import subprocess
import sys

from absl import app
from absl import flags
import numpy as np

from tpu_raiden.benchmarks import bap_metrics
from tpu_raiden.benchmarks import perf_core

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to h2d_d2h_gating_baselines.json. Default: alongside this binary.')
_RECORD = flags.DEFINE_bool(
    'record', False,
    'Re-measure and OVERWRITE baselines/floors instead of gating.')
_ITERS = flags.DEFINE_integer(
    'iters', None,
    'Override iters from the baselines file. Use a large value when recording '
    '(the floor depends on a MAD/sigma estimate, which needs many samples to be '
    'stable); the gate itself runs the smaller value in the baselines file.')

# The perf floor (NOT the correctness check) can be bypassed per-PR by putting
# one of these tags in the CL description / commit message.
_SKIP_TAGS = ('[skip-perf-gate]', '[skip-h2d-d2h-gating]')


def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2d_d2h_gating_baselines.json')


def _opted_out():
  """True if the HEAD commit message asks to skip the perf floor for this PR."""
  try:
    msg = subprocess.run(['git', 'log', '-1', '--format=%B'],
                         capture_output=True, text=True).stdout.lower()
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  for tag in _SKIP_TAGS:
    if tag in msg:
      return tag
  return None


def _core_floor(samples, k):
  """Per-config gate floor: median - k robust-sigmas (MAD-based).

      floor = median - k * MAD_sigma

  MAD_sigma = 1.4826 * median(|x - median|) is an outlier-resistant standard
  deviation, so the low tail does not drag the bound down.
  """
  x = np.asarray(samples, float)
  med = np.median(x)
  sigma = 1.4826 * np.median(np.abs(x - med))
  return float(med - k * sigma)


def _write_tb_metrics(results):
  """Log per-config throughput to TENSORBOARD_OUTPUT_DIR so BAP ingests it.
  Each tag MUST have a matching metrics{name:...} in benchmark_registry.pbtxt."""
  scalars = {}
  for c, r in results:
    label = f"{c['dtype']}_L{c['num_layers']}_{'x'.join(map(str, c['shape']))}"
    scalars[f'{label}/d2h_gbps'] = r['d2h_gbps']
    scalars[f'{label}/h2d_gbps'] = r['h2d_gbps']
  bap_metrics.emit(scalars)


def main(_):
  path = _baselines_path()
  with open(path) as f:
    cfg = json.load(f)
  sigma_k = float(cfg.get('sigma_k', 3.5))       # #robust-sigmas below median
  iters = int(cfg.get('iters', 20))
  if _ITERS.value is not None:
    iters = _ITERS.value
  warmup = int(cfg.get('warmup', 3))
  configs = cfg['configs']

  # NOTE: the correctness check (a d2h->h2d round-trip byte-equality assertion)
  # runs INSIDE perf_core.measure(), before any floor comparison. A corrupt or
  # no-op transfer raises there and fails the run regardless of the skip tag or
  # the "enforce" switch below -- correctness is never bypassable.
  results = []
  for c in configs:
    r = perf_core.measure(shape=c['shape'], num_layers=c['num_layers'],
                          dtype=c['dtype'], shard_axis=c.get('shard_axis', 2),
                          iters=iters, warmup=warmup)
    results.append((c, r))
    print(f"[measured] {c['dtype']} L{c['num_layers']} "
          f"{'x'.join(map(str, c['shape']))}  "
          f"d2h {r['d2h_gbps']:.1f}  h2d {r['h2d_gbps']:.1f} Gbps")

  # --- record mode: overwrite baselines + floors, no gating ---
  if _RECORD.value:
    for c, r in results:
      c['baseline_d2h'] = round(r['d2h_gbps'], 1)
      c['baseline_h2d'] = round(r['h2d_gbps'], 1)
      c['floor_d2h'] = round(_core_floor(r['d2h_gbps_all'], sigma_k), 1)
      c['floor_h2d'] = round(_core_floor(r['h2d_gbps_all'], sigma_k), 1)
    out_path = path
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      out_path = os.path.join(adir, 'h2d_d2h_gating_baselines.json')
    with open(out_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'Recorded {len(results)} baselines+floors '
          f'(iters={iters}, sigma_k={sigma_k}) -> {out_path}')
    return

  # emit per-config throughput to TB for the BAP dashboard (gate mode only)
  _write_tb_metrics(results)

  # --- gate mode: compare the median of `iters` runs against the recorded floor ---
  fails = 0
  print(f'\nperf gate: median of {iters} iters vs per-config floor '
        f'(median - {sigma_k} robust-sigmas / MAD)\n')
  print(f"{'config':30}{'dir':4}{'baseline':>9}{'floor':>9}{'median':>9}"
        f"{'drop':>7}  verdict")
  for c, r in results:
    label = f"{c['dtype']} L{c['num_layers']} {'x'.join(map(str, c['shape']))}"
    for d in ('d2h', 'h2d'):
      base = float(c[f'baseline_{d}'])
      floor = float(c[f'floor_{d}'])
      med = r[f'{d}_gbps']
      ok = med >= floor
      fails += not ok
      print(f"{label:30}{d:4}{base:9.1f}{floor:9.1f}{med:9.1f}"
            f"{(base-med)/base*100:6.1f}%  {'PASS' if ok else 'FAIL <-- REGRESSION'}")

  print()
  if not fails:
    print(f'GATE PASS: all {len(results)} configs at/above their floor')
    return

  # A perf-floor regression. It is blocking unless the maintainer switched the
  # gate to report-only ("enforce": false) or the author opted out via a tag.
  msg = f'GATE FAIL: {fails} direction(s) below the floor'
  enforce = bool(cfg.get('enforce', True))
  opt = _opted_out()
  if not enforce:
    print(msg + '   [report-only: "enforce": false in baselines, NOT blocking]')
    return
  if opt:
    print(msg + f'   [report-only: {opt} in commit message, NOT blocking]')
    return
  print(msg + f'   (to bypass, add {_SKIP_TAGS[0]} to your CL description)',
        file=sys.stderr)
  sys.exit(1)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))