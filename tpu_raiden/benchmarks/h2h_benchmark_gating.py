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

"""H2H (single-host cross-socket, PUSH) perf gate for the pbtxt/BAP path.

Mirrors h2d_d2h_benchmark_gating.py, but the transport is reused from
h2h_perf_test_oss (the same PUSH benchmark used by run_h2h_compare.sh), so BAP
runs ONE bazel target, on the TPU runner (2 NUMA nodes), no raw `container:`.

Per config it drives the h2h launcher (which spawns sender+receiver subprocesses
of THIS binary via --role), reads the sender's raw_h2h_results.json, then either
records baselines (--record) or gates the median GB/s against
h2h_gating_baselines.json. Each config's median is emitted as
"<label>/h2h_gbs" to TENSORBOARD_OUTPUT_DIR; every tag MUST have a matching
metrics{name:...} in benchmark_registry.pbtxt for BAP to ingest it.

NOTE(integrity): the PUSH receiver's --verify is best-effort/heuristic (dst block
ids are allocated dynamically), so this gate covers BANDWIDTH only. An
authoritative data-integrity gate should come from the pull-mode verify (exact
np.array_equal per layer) or the C++ runner's byte-for-byte check once its
non-zero-exit-on-mismatch is fixed. See the handoff notes.
"""

import json
import os
import sys
import tempfile

from absl import app
from absl import flags
import numpy as np  # noqa: F401  (kept for parity / future stat helpers)

# Reuse the PUSH transport. Importing it also registers the shared flags
# (--role, --num_blocks, --block_size, --num_layers, --num_shards,
#  --parallelism, --iters, --warmup, --sender_numa, --receiver_numa, --verify).
from tpu_raiden.benchmarks import h2h_perf_test_oss as h2h

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to h2h_gating_baselines.json. Default: alongside this binary (runfiles).')
_RECORD = flags.DEFINE_bool(
    'record', False,
    'Re-measure and OVERWRITE baselines instead of gating (auto-update hook).')


def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2h_gating_baselines.json')


def _label(c):
  return f"{c['block_size']}B_x{c['num_blocks']}_P{c.get('parallelism', 1)}"


def _apply_flags(c, iters, warmup):
  """Set the shared h2h flags for this config before launching."""
  f = flags.FLAGS
  f.num_blocks = int(c['num_blocks'])
  f.block_size = int(c['block_size'])
  f.num_layers = int(c.get('num_layers', 32))
  f.num_shards = int(c.get('num_shards', 1))
  f.parallelism = int(c.get('parallelism', 1))
  f.iters = int(iters)
  f.warmup = int(warmup)


def _measure_one(c, iters, warmup):
  """Run one config end-to-end via the h2h launcher; return its result dict.

  The launcher spawns sender+receiver as subprocesses of THIS binary; the sender
  writes raw_h2h_results.json into WORKLOAD_ARTIFACTS_DIR, which we point at a
  private temp dir per config so results never collide. TENSORBOARD_OUTPUT_DIR is
  hidden from the sub-run so only this gate emits the (deduplicated) BAP metrics.
  """
  _apply_flags(c, iters, warmup)
  artdir = tempfile.mkdtemp(prefix='h2h_art_')
  saved = {k: os.environ.get(k)
           for k in ('WORKLOAD_ARTIFACTS_DIR', 'TENSORBOARD_OUTPUT_DIR')}
  os.environ['WORKLOAD_ARTIFACTS_DIR'] = artdir
  os.environ.pop('TENSORBOARD_OUTPUT_DIR', None)
  try:
    rc = 0
    try:
      h2h.run_launcher()  # returns on success; sys.exit(rc) on failure
    except SystemExit as e:
      rc = e.code or 0
    if rc:
      raise RuntimeError(f'h2h transfer failed (rc={rc}) for {_label(c)}')
    with open(os.path.join(artdir, 'raw_h2h_results.json')) as fh:
      return json.load(fh)
  finally:
    for k, v in saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v


def _write_tb(results):
  """Emit <label>/h2h_gbs per config so BAP ingests it."""
  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tbdir:
    print('TENSORBOARD_OUTPUT_DIR not set; skipping TB metrics.')
    return
  try:
    try:
      import tensorboardX  # pylint: disable=g-import-not-at-top
      w = tensorboardX.SummaryWriter(log_dir=tbdir)
    except ImportError:
      import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
      w = tut.SummaryWriter(log_dir=tbdir)
    for c, r in results:
      w.add_scalar(f'{_label(c)}/h2h_gbs', r['throughput_gbs_median'], global_step=0)
    w.close()
    print(f'Wrote TB metrics for {len(results)} configs -> {tbdir}')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: TB metric write failed: {e}', file=sys.stderr)


def _run_gate():
  path = _baselines_path()
  with open(path) as f:
    cfg = json.load(f)
  thr = float(cfg.get('threshold', 0.03))
  iters = int(cfg.get('iters', 50))
  warmup = int(cfg.get('warmup', 3))
  configs = cfg['configs']

  results = []
  for c in configs:
    r = _measure_one(c, iters, warmup)
    results.append((c, r))
    med = float(r['throughput_gbs_median'])
    print(f'[measured] {_label(c):22} {med:8.3f} GB/s ({med*8:7.1f} Gbps)')

  # --- record mode: overwrite baselines, no gating ---
  if _RECORD.value:
    for c, r in results:
      c['baseline_gbs'] = round(float(r['throughput_gbs_median']), 3)
    with open(path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'Recorded {len(results)} baselines -> {path}')
    return

  _write_tb(results)

  # --- gate mode: median GB/s vs (1 - threshold) * baseline ---
  fails = 0
  print(f'\nH2H perf gate  threshold={thr*100:.0f}%  '
        f'(fail if median < {(1-thr)*100:.0f}% of baseline)\n')
  print(f"{'config':22}{'baseline':>10}{'median':>10}{'drop':>8}  verdict")
  for c, r in results:
    base = float(c.get('baseline_gbs', 0.0))
    med = float(r['throughput_gbs_median'])
    floor = (1 - thr) * base
    ok = med >= floor
    fails += not ok
    drop = (base - med) / base * 100 if base else 0.0
    print(f"{_label(c):22}{base:10.3f}{med:10.3f}{drop:7.1f}%  "
          f"{'PASS' if ok else 'FAIL <-- REGRESSION'}")

  if not any(float(c.get('baseline_gbs', 0.0)) > 0 for c in configs):
    print('\nNOTE: all baselines are 0.0 -> gate is a no-op. Run with --record on '
          'the TPU runner first, commit h2h_gating_baselines.json, then the gate '
          'becomes active.', file=sys.stderr)

  print()
  if fails:
    print(f'GATE FAIL: {fails} config(s) regressed > {thr*100:.0f}%', file=sys.stderr)
    sys.exit(1)
  print(f'GATE PASS: all {len(results)} configs within {thr*100:.0f}%')


def main(argv):
  # Sender/receiver subprocesses (spawned by h2h.run_launcher as `sys.argv[0]
  # --role=...`) re-enter here; hand them straight to the transport.
  if flags.FLAGS.role in ('sender', 'receiver'):
    return h2h.main(argv)
  _run_gate()


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
