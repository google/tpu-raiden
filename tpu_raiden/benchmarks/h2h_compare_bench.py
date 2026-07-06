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

"""H2H C++-vs-Python bandwidth comparison as a single BAP workload (pbtxt path).

One bazel target that, per config, runs BOTH the C++ h2h_benchmark_runner and the
Python push h2h_perf_test_oss on the same host over loopback (cross-socket), reads
each side's GB/s, and emits "<label>/cpp_gbs" and "<label>/py_gbs" to
TENSORBOARD_OUTPUT_DIR so BAP ingests them. Every tag MUST have a matching
metrics{name:...} in benchmark_registry_compare.pbtxt.

Both binaries are runfiles data deps (see BUILD), so BAP's `bazel run` builds and
ships them; this runs on the TPU runner via BAP's managed container -- NOT a raw
GitHub `container:` block (which is unreliable on this fleet).

Purpose is VALIDATION: confirm the Python bandwidth agrees with C++. If Python
crashes for a config, its full stdout/stderr is printed (so it shows up in the BAP
log) and py_gbs is recorded as -1.0; the run still finishes and emits what it can.
"""

import os
import re
import subprocess
import sys
import time

from absl import app
from absl import flags

_SENDER_NUMA = flags.DEFINE_integer('sender_numa', 0, 'NUMA node for the sender.')
_RECEIVER_NUMA = flags.DEFINE_integer('receiver_numa', 1, 'NUMA node for the receiver.')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Warmup transfers.')
_ITERS = flags.DEFINE_integer('iters', 20, 'Timed transfers.')
_TIMEOUT_S = flags.DEFINE_integer('timeout_s', 180, 'Per-process hard timeout.')

_LAYERS = 32   # C++ kNumLayers (fixed)
_SHARDS = 1    # C++ kNumShards (fixed)

# (block_size_bytes, num_blocks, parallelism). DEBUG: just one tiny config so the
# job finishes fast (before the flaky container hook dies) and we get the Python
# output to see where it hangs. Add configs back once Python works.
_CONFIGS = [
    (1048576, 64, 1),
]

_CPP_RE = re.compile(r'Throughput:\s*([0-9.]+)')                 # "Throughput: X GB/s (collective)"
_PY_RE = re.compile(r'Throughput \(mean\):\s*([0-9.]+)')          # "Throughput (mean): X GB/s = ..."


def _label(bs, nb, p):
  return f'{bs}B_x{nb}_P{p}'


def _locate(basename):
  """Find a data-dep binary in the runfiles tree by basename (workspace-agnostic)."""
  roots = []
  for env in ('RUNFILES_DIR', 'JAVA_RUNFILES', 'TEST_SRCDIR'):
    if os.environ.get(env):
      roots.append(os.environ[env])
  a0 = os.path.abspath(sys.argv[0])
  roots.append(a0 + '.runfiles')
  roots.append(os.path.dirname(a0))
  roots.append(os.getcwd())
  seen = set()
  for root in roots:
    if not root or root in seen or not os.path.isdir(root):
      continue
    seen.add(root)
    for dirpath, _, files in os.walk(root):
      if basename in files:
        cand = os.path.join(dirpath, basename)
        if os.access(cand, os.X_OK):
          return cand
  raise FileNotFoundError(f'could not locate {basename} in runfiles; tried {roots}')


def _run(cmd, timeout):
  """Run a command, capturing combined stdout/stderr; never raise on non-zero.

  PYTHONUNBUFFERED=1 so the child's [launch]/[sender]/[receiver] prints are NOT
  lost when we kill it on timeout -- that's what tells us where a hang happened.
  """
  env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
  try:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout
  except subprocess.TimeoutExpired as e:
    out = e.output
    if isinstance(out, (bytes, bytearray)):  # on timeout, output may be undecoded bytes
      out = out.decode('utf-8', 'replace')
    return 124, (out or '') + f'\n[timeout after {timeout}s]'


def _run_cpp(cc, bs, nb, p, port):
  """Start the C++ receiver (bg) + sender (timed); return GB/s or -1.0."""
  # Discard the receiver's stdout/stderr: we only need the SENDER's throughput
  # line, and an undrained PIPE here deadlocks once the receiver's log fills it.
  recv = subprocess.Popen(
      [cc, '--role=receiver', '--data_interface=lo', f'--peer_control_port={port}',
       f'--numa_node={_RECEIVER_NUMA.value}', f'--block_size={bs}',
       f'--num_blocks={nb}', f'--parallelism={p}'],
      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  time.sleep(3)  # let the control server bind
  rc, out = _run(
      [cc, '--role=sender', '--peer_control_ip=127.0.0.1', '--data_interface=lo',
       f'--peer_control_port={port}', f'--numa_node={_SENDER_NUMA.value}',
       f'--block_size={bs}', f'--num_blocks={nb}', f'--parallelism={p}'],
      _TIMEOUT_S.value)
  try:
    recv.terminate(); recv.wait(10)
  except Exception:  # pylint: disable=broad-exception-caught
    recv.kill()
  m = _CPP_RE.search(out or '')
  if m:
    return float(m.group(1))
  print(f'[compare] C++ FAILED for {_label(bs, nb, p)} (rc={rc}); output:\n{out}',
        flush=True)
  return -1.0


def _run_py(py_bin, bs, nb, p):
  """Run the Python push launcher (timed); return GB/s or -1.0."""
  rc, out = _run(
      [py_bin, '--role=launch', f'--sender_numa={_SENDER_NUMA.value}',
       f'--receiver_numa={_RECEIVER_NUMA.value}', f'--num_layers={_LAYERS}',
       f'--num_shards={_SHARDS}', f'--block_size={bs}', f'--num_blocks={nb}',
       f'--parallelism={p}', f'--warmup={_WARMUP.value}', f'--iters={_ITERS.value}'],
      _TIMEOUT_S.value)
  m = _PY_RE.search(out or '')
  if m:
    return float(m.group(1))
  # Surface the full Python output so the crash shows up in the BAP log.
  print(f'[compare] PYTHON FAILED for {_label(bs, nb, p)} (rc={rc}); full output:\n'
        f'{"-"*70}\n{out}\n{"-"*70}', flush=True)
  return -1.0


def _emit_tb(results):
  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tbdir:
    print('TENSORBOARD_OUTPUT_DIR not set; skipping TB metrics.', flush=True)
    return
  try:
    try:
      import tensorboardX  # pylint: disable=g-import-not-at-top
      w = tensorboardX.SummaryWriter(log_dir=tbdir)
    except ImportError:
      import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
      w = tut.SummaryWriter(log_dir=tbdir)
    for label, cpp, py in results:
      w.add_scalar(f'{label}/cpp_gbs', cpp, global_step=0)
      w.add_scalar(f'{label}/py_gbs', py, global_step=0)
    w.close()
    print(f'Wrote TB metrics for {len(results)} configs -> {tbdir}', flush=True)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: TB metric write failed: {e}', file=sys.stderr)


def main(_):
  cc = _locate('h2h_benchmark_runner')
  py_bin = _locate('h2h_perf_test_oss')
  print(f'[compare] C++ runner : {cc}', flush=True)
  print(f'[compare] Python push: {py_bin}', flush=True)

  results = []
  port = 9990
  n = len(_CONFIGS)
  for i, (bs, nb, p) in enumerate(_CONFIGS, 1):
    port += 1
    label = _label(bs, nb, p)
    print(f'[compare] ({i}/{n}) {label}: running C++ ...', flush=True)
    cpp = _run_cpp(cc, bs, nb, p, port)
    print(f'[compare] ({i}/{n}) {label}: C++ done ({cpp:.3f} GB/s); running Python ...',
          flush=True)
    py = _run_py(py_bin, bs, nb, p)
    results.append((label, cpp, py))
    print(f'[measured] {label:22} cpp={cpp:8.3f} py={py:8.3f} GB/s', flush=True)

  _emit_tb(results)

  print(f"\n{'config':22}{'cpp_GB/s':>10}{'py_GB/s':>10}{'py/cpp':>9}")
  print('-' * 51)
  py_fail = 0
  for label, cpp, py in results:
    ratio = f'{py / cpp:.2f}' if (cpp > 0 and py > 0) else 'NA'
    py_fail += py <= 0
    print(f'{label:22}{cpp:10.3f}{py:10.3f}{ratio:>9}')
  print()
  if py_fail:
    print(f'NOTE: Python failed on {py_fail}/{len(results)} config(s) -> see the '
          '"PYTHON FAILED" dumps above for the traceback.', flush=True)
  else:
    print('Python produced a bandwidth for every config.', flush=True)
  # Exit 0 so BAP finishes and collects metrics + this log even on Python failure.


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
