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


import csv
import json
import os
import sys

from absl import app
from absl import flags
import numpy as np

from tpu_raiden.benchmarks import perf_core

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to h2d_d2h_gating_baselines.json. Default: alongside this binary (runfiles).')
_RECORD = flags.DEFINE_bool(
    'record', False,
    'Re-measure and OVERWRITE baselines/floors instead of gating.')
_ITERS = flags.DEFINE_integer(
    'iters', None,
    'Override iters from the baselines file. Use a large value when recording '
    '(the floor depends on a MAD/sigma estimate, which needs many samples to be '
    'stable); the gate itself runs the smaller value in gating_baselines.json.')
_DUMP = flags.DEFINE_string(
    'dump', None,
    'CSV to write EVERY raw per-iteration sample (config,dir,iter,gbps) for your '
    'own analysis. On BAP it is redirected into WORKLOAD_ARTIFACTS_DIR so the '
    'workflow uploads it as a downloadable artifact. Use --iters=500 for 500.')
_ANALYZE = flags.DEFINE_bool(
    'analyze', False,
    'Dump raw samples (--dump) + print, then exit 0. No record, no gate.')


def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2d_d2h_gating_baselines.json')


def _core_floor(samples, k, max_margin):
  """Per-config gate floor: lower edge of the normal core, capped at max_margin.

      floor = max(median - k * MAD_sigma,  median * (1 - max_margin))

  MAD_sigma = 1.4826 * median(|x - median|) is an outlier-resistant standard
  deviation, so the low tail does not drag the bound down. The cap keeps a noisy
  config from ending up looser than a flat max_margin (e.g. fp32 L1).
  """
  x = np.asarray(samples, float)
  med = np.median(x)
  sigma = 1.4826 * np.median(np.abs(x - med))
  core = med - k * sigma                      # normal-core lower bound
  cap = med * (1.0 - max_margin)              # never looser than max_margin
  return float(max(core, cap))


def _write_tb_metrics(results):
  """Log per-config throughput to TENSORBOARD_OUTPUT_DIR so BAP ingests it.
  Each tag MUST have a matching metrics{name:...} in benchmark_registry.pbtxt."""
  tblog_dir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tblog_dir:
    print('TENSORBOARD_OUTPUT_DIR not set; skipping TB metrics.')
    return
  try:
    try:
      import tensorboardX  # pylint: disable=g-import-not-at-top
      w = tensorboardX.SummaryWriter(log_dir=tblog_dir)
    except ImportError:
      import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
      w = tut.SummaryWriter(log_dir=tblog_dir)
    for c, r in results:
      label = f"{c['dtype']}_L{c['num_layers']}_{'x'.join(map(str, c['shape']))}"
      w.add_scalar(f'{label}/d2h_gbps', r['d2h_gbps'], global_step=0)
      w.add_scalar(f'{label}/h2d_gbps', r['h2d_gbps'], global_step=0)
    w.close()
    print(f'Wrote TB metrics for {len(results)} configs -> {tblog_dir}')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: TB metric write failed: {e}', file=sys.stderr)


def main(_):
  path = _baselines_path()
  with open(path) as f:
    cfg = json.load(f)
  sigma_k = float(cfg.get('sigma_k', 3.5))       # #robust-sigmas below median
  max_margin = float(cfg.get('max_margin', 0.03))  # floor never looser than this
  iters = int(cfg.get('iters', 20))
  if _ITERS.value is not None:
    iters = _ITERS.value
  warmup = int(cfg.get('warmup', 3))
  configs = cfg['configs']

  results = []
  for c in configs:
    r = perf_core.measure(shape=c['shape'], num_layers=c['num_layers'],
                          dtype=c['dtype'], shard_axis=c.get('shard_axis', 2),
                          iters=iters, warmup=warmup)
    results.append((c, r))
    print(f"[measured] {c['dtype']} L{c['num_layers']} "
          f"{'x'.join(map(str, c['shape']))}  "
          f"d2h {r['d2h_gbps']:.1f}  h2d {r['h2d_gbps']:.1f} Gbps")

  # --- dump raw per-iteration samples for offline analysis ---
  if _DUMP.value:
    dump_path = _DUMP.value
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:  # on BAP, land it where the workflow uploads artifacts from
      dump_path = os.path.join(adir, os.path.basename(dump_path))
    with open(dump_path, 'w', newline='') as f:
      w = csv.writer(f)
      w.writerow(['config', 'dir', 'iter', 'gbps'])
      for c, r in results:
        label = f"{c['dtype']}_L{c['num_layers']}_{'x'.join(map(str, c['shape']))}"
        for d in ('d2h', 'h2d'):
          for it, v in enumerate(r[f'{d}_gbps_all']):
            w.writerow([label, d, it, f'{float(v):.4f}'])
    print(f'Wrote raw samples ({iters} iters x {len(results)} configs x 2 dirs) '
          f'-> {dump_path}')

  if _ANALYZE.value:
    print('analyze mode: raw samples dumped, no record/gate. Done.')
    return

  # --- record mode: overwrite baselines + floors, no gating ---
  if _RECORD.value:
    for c, r in results:
      c['baseline_d2h'] = round(r['d2h_gbps'], 1)
      c['baseline_h2d'] = round(r['h2d_gbps'], 1)
      c['floor_d2h'] = round(_core_floor(r['d2h_gbps_all'], sigma_k, max_margin), 1)
      c['floor_h2d'] = round(_core_floor(r['h2d_gbps_all'], sigma_k, max_margin), 1)
    # In CI, BAP sets WORKLOAD_ARTIFACTS_DIR and uploads its contents, so write
    # the recorded baselines there to get them back out; locally, overwrite path.
    out_path = path
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      out_path = os.path.join(adir, 'h2d_d2h_gating_baselines.json')
    with open(out_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'Recorded {len(results)} baselines+floors '
          f'(iters={iters}, sigma_k={sigma_k}, cap={max_margin*100:.0f}%) '
          f'-> {out_path}')
    return

  # emit per-config throughput to TB for the BAP dashboard (gate mode only)
  _write_tb_metrics(results)

  # --- gate mode: compare the median of `iters` runs against the recorded floor ---
  fails = 0
  print(f'\nperf gate: median of {iters} iters vs per-config normal-core floor '
        f'(median - {sigma_k} MADsigma, capped at {max_margin*100:.0f}%)\n')
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
  if fails:
    print(f'GATE FAIL: {fails} direction(s) below the normal-core floor',
          file=sys.stderr)
    sys.exit(1)
  print(f'GATE PASS: all {len(results)} configs at/above their normal-core floor')


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
