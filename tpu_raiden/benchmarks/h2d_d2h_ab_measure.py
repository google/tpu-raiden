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

"""Measure-only workload for BAP ab_mode (A/B regression check).

BAP runs this once on the baseline ref and once on the experiment ref, then its
ab_analyzer compares the emitted metrics and posts a PR comment / fails on
regression. So this binary must NOT gate itself: it only measures every config
and writes per-config d2h/h2d throughput to TENSORBOARD_OUTPUT_DIR, where BAP
ingests it. Metric names here MUST match the metrics{name:...} entries in
benchmark_registry_ab.pbtxt.

Config list is read from gating_baselines.json (shapes only; the baseline/floor
numbers in it are ignored here -- ab_mode does the comparison).
"""
import json
import os
import sys

from absl import app
from absl import flags

from tpu_raiden.benchmarks import perf_core

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to gating_baselines.json; default alongside this binary.')
_ITERS = flags.DEFINE_integer('iters', 20, 'Iters per config; median reported.')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Warmup iters per config.')


def _label(c):
  return f"{c['dtype']}_L{c['num_layers']}_{'x'.join(map(str, c['shape']))}"


def _write_tb(results):
  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tbdir:
    print('TENSORBOARD_OUTPUT_DIR not set; nothing for BAP to ingest.',
          file=sys.stderr)
    return
  try:
    import tensorboardX  # pylint: disable=g-import-not-at-top
    w = tensorboardX.SummaryWriter(log_dir=tbdir)
  except ImportError:
    import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
    w = tut.SummaryWriter(log_dir=tbdir)
  for label, r in results:
    w.add_scalar(f'{label}/d2h_gbps', r['d2h_gbps'], global_step=0)
    w.add_scalar(f'{label}/h2d_gbps', r['h2d_gbps'], global_step=0)
  w.close()
  print(f'Wrote {len(results)} configs -> {tbdir}')


def main(_):
  path = _BASELINES.value or os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'gating_baselines.json')
  with open(path) as f:
    cfg = json.load(f)
  results = []
  for c in cfg['configs']:
    r = perf_core.measure(shape=c['shape'], num_layers=c['num_layers'],
                          dtype=c['dtype'], shard_axis=c.get('shard_axis', 2),
                          iters=_ITERS.value, warmup=_WARMUP.value)
    results.append((_label(c), r))
    print(f"[measured] {_label(c)}  d2h {r['d2h_gbps']:.1f}  "
          f"h2d {r['h2d_gbps']:.1f} Gbps")
  _write_tb(results)  # BAP's ab_analyzer diffs these across baseline vs experiment


if __name__ == '__main__':
  app.run(main, flags_parser=lambda a: flags.FLAGS(a, known_only=True))
