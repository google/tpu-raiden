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

"""C++-only H2H record/gate driver.

A THIN orchestrator: it only spawns the C++ h2h_benchmark_runner (sender +
receiver) over loopback (cross-socket) and parses its stdout. It never imports
the JAX bindings, so it needs NO change to tpu_raiden_jax_module.cc.

Per config it captures:
  * throughput: derived from the C++ runner's p50 latency (median GB/s), and
  * integrity: the C++ receiver's own "Data integrity verification PASSED/FAILED".

Modes:
  --record : run each config, write {throughput, integrity} to --baselines.
  (gate)   : read --baselines; FAIL if any config reports integrity FAILED or
             throughput below floor (baseline * (1 - --margin)).

Metrics are emitted to TENSORBOARD_OUTPUT_DIR so BAP ingests them; every tag has
a matching metrics{name:...} in the registry pbtxt.
"""

import json
import os
import re
import subprocess
import sys
import time

from absl import app
from absl import flags

_SENDER_NUMA = flags.DEFINE_integer('sender_numa', 0, 'NUMA node for the sender.')
_RECEIVER_NUMA = flags.DEFINE_integer('receiver_numa', 1,
                                      'NUMA node for the receiver.')
_TIMEOUT_S = flags.DEFINE_integer('timeout_s', 300, 'Per-process hard timeout.')
_RECORD = flags.DEFINE_bool('record', False,
                            'Record baselines to --baselines instead of gating.')
_BASELINES = flags.DEFINE_string(
    'baselines', 'tpu_raiden/benchmarks/h2h_cpp_baselines.json',
    'Baselines file: read for gating, written for --record.')
_MARGIN = flags.DEFINE_float('margin', 0.03,
                             'Allowed fractional drop below baseline (floor).')
_CONTROL_PORT = flags.DEFINE_integer('control_port', 9099,
                                     'Base control port for the C++ handshake.')

_LAYERS = 32   # C++ kNumLayers (fixed in the runner)
_SHARDS = 1    # C++ kNumShards (fixed in the runner)

# (block_size_bytes, num_blocks, parallelism). Kept to 1MB so sender+receiver do
# not exhaust host memory (16MB/128MB OOM the pod and drop BAP's :50051 channel).
_CONFIGS = [
    (1048576, 64, 1),
    (1048576, 64, 8),
]

_CPP_P50_RE = re.compile(r'p50:\s*([0-9.]+)')          # "p50:   X ms"
_CPP_MEAN_RE = re.compile(r'Throughput:\s*([0-9.]+)')  # fallback (mean) GB/s
_INTEG_PASS = 'Data integrity verification PASSED'
_INTEG_FAIL = 'Data integrity verification FAILED'


def _label(bs, nb, p):
  return f'{bs}B_x{nb}_P{p}'


def _locate(basename):
  """Find the C++ runner binary in the bazel runfiles by basename."""
  root = os.environ.get('RUNFILES_DIR') or os.path.dirname(os.path.abspath(__file__))
  for dirpath, _, files in os.walk(root):
    if basename in files:
      cand = os.path.join(dirpath, basename)
      if os.access(cand, os.X_OK):
        return cand
  raise FileNotFoundError(f'could not locate {basename} under {root}')


def _run(cmd, timeout):
  """Run a subprocess to completion; return (rc, combined_output)."""
  env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
  try:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, env=env)
    return p.returncode, p.stdout.decode('utf-8', 'replace')
  except subprocess.TimeoutExpired as e:
    out = e.output.decode('utf-8', 'replace') if e.output else ''
    return 124, out + f'\n[timeout after {timeout}s]'


def _run_cpp(cc, bs, nb, p, port):
  """Spawn the C++ receiver (bg) + sender (timed) and return (gbs, integrity_ok).

  gbs is the median throughput derived from the sender's p50 latency, or -1.0 on
  failure. integrity_ok is True only if the receiver printed the PASSED line.
  """
  base = [cc, '--data_interface=lo', f'--peer_control_port={port}',
          f'--block_size={bs}', f'--num_blocks={nb}', f'--parallelism={p}']
  # Receiver: capture its stdout so we can read the integrity verdict.
  recv = subprocess.Popen(
      base + ['--role=receiver', f'--numa_node={_RECEIVER_NUMA.value}'],
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
      env={**os.environ, 'PYTHONUNBUFFERED': '1'})
  time.sleep(3)  # let the control server bind
  rc, sender_out = _run(
      base + ['--role=sender', '--peer_control_ip=127.0.0.1',
              f'--numa_node={_SENDER_NUMA.value}'],
      _TIMEOUT_S.value)
  try:
    recv.terminate()
    recv_out = recv.communicate(timeout=15)[0].decode('utf-8', 'replace')
  except Exception:  # pylint: disable=broad-exception-caught
    recv.kill()
    recv_out = ''

  integrity_ok = (_INTEG_PASS in recv_out) and (_INTEG_FAIL not in recv_out)

  gbs = -1.0
  m = _CPP_P50_RE.search(sender_out or '')
  if m and float(m.group(1)) > 0:
    total_bytes = _LAYERS * _SHARDS * nb * bs
    gbs = (total_bytes / 1e9) / (float(m.group(1)) / 1000.0)
  else:
    m2 = _CPP_MEAN_RE.search(sender_out or '')
    if m2:
      gbs = float(m2.group(1))

  if gbs < 0 or not integrity_ok:
    print(f'[cpp] {_label(bs, nb, p)} FAILED (rc={rc}, integrity_ok={integrity_ok});'
          f' sender output:\n{sender_out}\n--- receiver tail ---\n{recv_out[-2000:]}',
          flush=True)
  return gbs, integrity_ok


def _emit(tag, value):
  """Emit a scalar to TENSORBOARD_OUTPUT_DIR so BAP ingests it (best-effort)."""
  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tbdir:
    return
  try:
    try:
      import tensorboardX  # pylint: disable=g-import-not-at-top
      w = tensorboardX.SummaryWriter(log_dir=tbdir)
    except ImportError:
      import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
      w = tut.SummaryWriter(log_dir=tbdir)
    w.add_scalar(tag, value, global_step=0)
    w.close()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: TB write failed for {tag}: {e}', file=sys.stderr)


def main(_):
  cc = _locate('h2h_benchmark_runner')
  print(f'[cpp-gate] C++ runner: {cc}', flush=True)

  results = {}
  for i, (bs, nb, p) in enumerate(_CONFIGS):
    label = _label(bs, nb, p)
    port = _CONTROL_PORT.value + i
    print(f'[cpp-gate] ({i + 1}/{len(_CONFIGS)}) {label}: running C++ ...',
          flush=True)
    gbs, integ = _run_cpp(cc, bs, nb, p, port)
    results[label] = {'gbs': gbs, 'integrity': integ}
    print(f'[measured] {label:<22} {gbs:8.3f} GB/s  integrity={"OK" if integ else "CORRUPT"}',
          flush=True)
    _emit(f'{label}/cpp_gbs', gbs)

  if _RECORD.value:
    with open(_BASELINES.value, 'w') as f:
      json.dump({'margin': _MARGIN.value, 'configs': results}, f, indent=2)
    print(f'\nRecorded {len(results)} C++ H2H baselines -> {_BASELINES.value}',
          flush=True)
    return

  # Gate mode.
  try:
    with open(_BASELINES.value) as f:
      base = json.load(f)
  except (OSError, ValueError) as e:
    print(f'GATE ERROR: cannot read baselines {_BASELINES.value}: {e}',
          file=sys.stderr)
    sys.exit(1)

  bad = []
  print('\nconfig                  baseline    floor   measured  integ  verdict')
  print('-' * 70)
  for label, r in results.items():
    b = base.get('configs', {}).get(label, {})
    baseline = float(b.get('gbs', 0.0))
    floor = baseline * (1.0 - base.get('margin', _MARGIN.value))
    integ = r['integrity']
    drop = (r['gbs'] < floor) and baseline > 0
    ok = integ and not drop
    verdict = 'PASS' if ok else 'FAIL'
    reason = '' if ok else (' <-- CORRUPT' if not integ else ' <-- SLOW')
    print(f'{label:<22} {baseline:8.3f} {floor:8.3f} {r["gbs"]:8.3f}  '
          f'{"OK" if integ else "CORRUPT":<7} {verdict}{reason}')
    if not ok:
      bad.append(label)

  if bad:
    print(f'\nGATE FAIL: {len(bad)} config(s) failed: {bad}', file=sys.stderr)
    sys.exit(1)
  print('\nGATE PASS: all configs within floor and integrity intact.', flush=True)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
