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

Same gate shape as h2d_d2h_benchmark_gating.py (normal-core floor, capped at
max_margin), but the transport is reused from h2h_perf_test_oss, so BAP runs ONE
bazel target on the TPU runner (2 NUMA nodes) with no raw `container:` block.

Per config it drives the h2h launcher (which spawns sender+receiver subprocesses
of THIS binary via --role), reads the sender's raw_h2h_results.json, then either
records baselines+floors (--record) or gates the median GB/s against
h2h_gating_baselines.json. Each config's median is emitted as "<label>/h2h_gbs"
to TENSORBOARD_OUTPUT_DIR; every tag MUST have a matching metrics{name:...} in
benchmark_registry.pbtxt for BAP to ingest it.

Two hard gates per config:
  * INTEGRITY: an authoritative byte-for-byte check (the sender does one untimed
    verify write with a known src->dst map; the receiver np.array_equal-compares
    every written block to the deterministic source fill). Corruption fails the
    gate regardless of bandwidth.
  * BANDWIDTH: median GB/s must stay at/above the recorded normal-core floor
    (capped at max_margin).
"""

import json
import os
import sys
import tempfile

from absl import app
from absl import flags
import numpy as np

# Reuse the PUSH transport. Importing it also registers the shared flags
# (--role, --num_blocks, --block_size, --num_layers, --num_shards,
#  --parallelism, --iters, --warmup, --sender_numa, --receiver_numa, --verify).
from tpu_raiden.benchmarks import h2h_perf_test_oss as h2h

_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Path to h2h_gating_baselines.json. Default: alongside this binary (runfiles).')
_RECORD = flags.DEFINE_bool(
    'record', False,
    'Re-measure and OVERWRITE baselines/floors instead of gating.')
_ITERS = flags.DEFINE_integer(
    'iters_override', None,
    'Override iters from the baselines file. Use a large value when recording '
    '(the floor depends on a MAD/sigma estimate, which needs many samples to be '
    'stable); the gate itself runs the smaller value in h2h_gating_baselines.json.')


def _baselines_path():
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2h_gating_baselines.json')


def _core_floor(samples, k, max_margin):
  """Per-config gate floor: lower edge of the normal core, capped at max_margin.

      floor = max(median - k * MAD_sigma,  median * (1 - max_margin))

  MAD_sigma = 1.4826 * median(|x - median|) is an outlier-resistant standard
  deviation, so the low tail does not drag the bound down. The cap keeps a noisy
  config from ending up looser than a flat max_margin.
  """
  x = np.asarray(samples, float)
  med = np.median(x)
  sigma = 1.4826 * np.median(np.abs(x - med))
  core = med - k * sigma
  cap = med * (1.0 - max_margin)
  return float(max(core, cap))


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
  f.verify = True  # authoritative byte-for-byte integrity check every config


def _measure_one(c, iters, warmup):
  """Run one config end-to-end via the h2h launcher; return {gbs, gbs_all}.

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
      h2h.run_launcher()  # returns on success; sys.exit(8)=corrupt, other=failed
    except SystemExit as e:
      rc = e.code or 0
    # Bandwidth (raw_h2h_results.json) is written BEFORE the verify verdict, so it
    # exists even when integrity fails (rc==8). Only a real transfer failure leaves
    # no results -> that's the one case we raise.
    res_path = os.path.join(artdir, 'raw_h2h_results.json')
    if not os.path.exists(res_path):
      raise RuntimeError(f'h2h transfer produced no results (rc={rc}) for {_label(c)}')
    with open(res_path) as fh:
      raw = json.load(fh)
    ver = None
    ver_path = os.path.join(artdir, 'verify.json')
    if os.path.exists(ver_path):
      with open(ver_path) as fh:
        ver = json.load(fh)
    return {'gbs': float(raw['throughput_gbs_median']),
            'gbs_all': list(raw['h2h_gbs_all']),
            'integrity_ok': bool(ver and ver.get('ok', False)),
            'verify': ver}
  finally:
    for k, v in saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v


def _write_tb_metrics(results):
  """Log per-config throughput to TENSORBOARD_OUTPUT_DIR so BAP ingests it.
  Each tag MUST have a matching metrics{name:...} in benchmark_registry.pbtxt."""
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
      w.add_scalar(f'{_label(c)}/h2h_gbs', r['gbs'], global_step=0)
    w.close()
    print(f'Wrote TB metrics for {len(results)} configs -> {tbdir}')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: TB metric write failed: {e}', file=sys.stderr)


def _run_gate():
  path = _baselines_path()
  with open(path) as f:
    cfg = json.load(f)
  sigma_k = float(cfg.get('sigma_k', 3.5))
  max_margin = float(cfg.get('max_margin', 0.03))
  iters = int(cfg.get('iters', 50))
  if _ITERS.value is not None:
    iters = _ITERS.value
  warmup = int(cfg.get('warmup', 3))
  configs = cfg['configs']

  results = []
  for c in configs:
    r = _measure_one(c, iters, warmup)
    results.append((c, r))
    print(f'[measured] {_label(c):22} {r["gbs"]:8.3f} GB/s '
          f'({r["gbs"]*8:7.1f} Gbps)  integrity='
          f'{"OK" if r["integrity_ok"] else "CORRUPT"}')

  # --- record mode: overwrite baselines + floors, no gating ---
  if _RECORD.value:
    bad = [_label(c) for c, r in results if not r['integrity_ok']]
    if bad:
      print(f'WARNING: recording baselines despite integrity failures on: {bad}',
            file=sys.stderr)
    for c, r in results:
      c['baseline_gbs'] = round(r['gbs'], 3)
      c['floor_gbs'] = round(_core_floor(r['gbs_all'], sigma_k, max_margin), 3)
    out_path = path
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      out_path = os.path.join(adir, os.path.basename(path))
    with open(out_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'Recorded {len(results)} baselines+floors '
          f'(iters={iters}, sigma_k={sigma_k}, cap={max_margin*100:.0f}%) '
          f'-> {out_path}')
    return

  _write_tb_metrics(results)

  # --- gate mode: (1) authoritative integrity, hard, and (2) bandwidth floor ---
  bw_fail = 0
  integ_fail = 0
  print(f'\nH2H gate: authoritative byte-integrity (hard) + median of {iters} iters '
        f'vs normal-core floor (median - {sigma_k} MADsigma, cap {max_margin*100:.0f}%)\n')
  print(f"{'config':22}{'baseline':>10}{'floor':>10}{'median':>10}{'drop':>7}"
        f"{'integ':>9}  verdict")
  for c, r in results:
    base = float(c.get('baseline_gbs', 0.0))
    floor = float(c.get('floor_gbs', 0.0))
    med = r['gbs']
    bw_ok = med >= floor
    integ_ok = bool(r['integrity_ok'])
    bw_fail += not bw_ok
    integ_fail += not integ_ok
    drop = (base - med) / base * 100 if base else 0.0
    verdict = 'PASS' if (bw_ok and integ_ok) else 'FAIL <-- ' + (
        'CORRUPT' if not integ_ok else 'REGRESSION')
    print(f"{_label(c):22}{base:10.3f}{floor:10.3f}{med:10.3f}{drop:6.1f}%"
          f"{'OK' if integ_ok else 'CORRUPT':>9}  {verdict}")

  if not any(float(c.get('floor_gbs', 0.0)) > 0 for c in configs):
    print('\nNOTE: all floors are 0.0 -> BANDWIDTH gate is a no-op (integrity still '
          'enforced). Run with --record on the TPU runner first, commit '
          'h2h_gating_baselines.json, then the bandwidth gate becomes active.',
          file=sys.stderr)

  print()
  if integ_fail:
    print(f'GATE FAIL: {integ_fail} config(s) FAILED data integrity (corruption)',
          file=sys.stderr)
  if bw_fail:
    print(f'GATE FAIL: {bw_fail} config(s) below the normal-core floor',
          file=sys.stderr)
  if integ_fail or bw_fail:
    sys.exit(1)
  print(f'GATE PASS: all {len(results)} configs integrity-OK and at/above floor')


def main(argv):
  # Sender/receiver subprocesses (spawned by h2h.run_launcher as `sys.argv[0]
  # --role=...`) re-enter here; hand them straight to the transport.
  if flags.FLAGS.role in ('sender', 'receiver'):
    return h2h.main(argv)
  _run_gate()


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
