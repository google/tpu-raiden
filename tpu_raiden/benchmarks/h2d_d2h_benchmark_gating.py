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
import sys

from absl import app
from absl import flags

from tpu_raiden.benchmarks import perf_core

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to gating_baselines.json. Default: alongside this binary (runfiles).')
_RECORD = flags.DEFINE_bool(
    'record', False,
    'Re-measure and OVERWRITE baselines instead of gating (auto-update hook).')
_ITERS = flags.DEFINE_integer(
    'iters', None,
    'Override iters from the baselines file (e.g. more samples when recording). '
    'Default: use the value in gating_baselines.json.')

def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'gating_baselines.json')

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
  thr = float(cfg.get('threshold', 0.03))
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

   # --- record mode: overwrite baselines, no gating ---
  if _RECORD.value:
    for c, r in results:
      c['baseline_d2h'] = round(r['d2h_gbps'], 1)
      c['baseline_h2d'] = round(r['h2d_gbps'], 1)
    # In CI, BAP sets WORKLOAD_ARTIFACTS_DIR and uploads its contents as an
    # artifact, so write the recorded baselines there to get them back out.
    # Locally (env unset) this falls back to the --baselines path, unchanged.
    out_path = path
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      out_path = os.path.join(adir, 'gating_baselines.json')
    with open(out_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'Recorded {len(results)} baselines -> {out_path}')
    return

  # emit per-config throughput to TB for the BAP dashboard (gate mode only)
  _write_tb_metrics(results)

  # --- gate mode ---
  fails = 0
  print(f'\nperf gate  threshold={thr*100:.0f}%  '
        f'(fail if median < {(1-thr)*100:.0f}% of baseline)\n')
  print(f"{'config':30}{'dir':4}{'baseline':>9}{'median':>9}{'drop':>7}  verdict")
  for c, r in results:
    label = f"{c['dtype']} L{c['num_layers']} {'x'.join(map(str, c['shape']))}"
    for d in ('d2h', 'h2d'):
      base = float(c[f'baseline_{d}'])
      med = r[f'{d}_gbps']
      floor = (1 - thr) * base
      ok = med >= floor
      fails += not ok
      print(f"{label:30}{d:4}{base:9.1f}{med:9.1f}"
            f"{(base-med)/base*100:6.1f}%  {'PASS' if ok else 'FAIL <-- REGRESSION'}")

  print()
  if fails:
    print(f'GATE FAIL: {fails} direction(s) regressed > {thr*100:.0f}%',
          file=sys.stderr)
    sys.exit(1)
  print(f'GATE PASS: all {len(results)} configs within {thr*100:.0f}%')


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
