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

import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time

from absl import app
from absl import flags

_SENDER_NUMA = flags.DEFINE_integer('sender_numa', 0, 'NUMA node for the sender.')
_RECEIVER_NUMA = flags.DEFINE_integer('receiver_numa', 1,
                                      'NUMA node for the receiver.')
_TIMEOUT_S = flags.DEFINE_integer(
    'timeout_s', 900,
    'Per-process hard timeout. High-parallelism configs contend on a single '
    'loopback and can take several minutes for 50 iterations.')
_RECORD = flags.DEFINE_bool('record', False,
                            'Record baselines to --baselines instead of gating.')
_BASELINES = flags.DEFINE_string(
    'baselines', None,
    'Baselines JSON path. Default: alongside this binary (runfiles).')
_MAX_MARGIN = flags.DEFINE_float(
    'max_margin', 0.03,
    'Floor is never looser than this fractional drop below the median.')
_SIGMA_K = flags.DEFINE_float(
    'sigma_k', 3.5, 'Robust sigmas (MAD) below the median for the gate floor.')
_CONTROL_PORT = flags.DEFINE_integer('control_port', 9099,
                                     'Base control port for the C++ handshake.')
_RUNS_PER_CONFIG = flags.DEFINE_integer(
    'runs_per_config', 1,
    'How many times to relaunch the C++ runner for each config, sequentially. '
    'Each launch is a fresh process contributing `iters` samples, so total raw '
    'samples = runs_per_config * iters. To get more samples, raise --iters (one '
    'process amortizes the handshake/warmup); raise this only when you '
    'specifically want process-to-process spread.')
_ITERS = flags.DEFINE_integer(
    'iters', 50,
    'Timed iterations PER run, passed to the C++ runner as --iterations. One '
    'process emits this many raw H2H_ITER_MS samples. e.g. --iters=500 --runs=1 '
    'gives 500 raw samples in a single handshake.')
_DUMP = flags.DEFINE_string(
    'dump', None,
    'CSV to write every sample for your own analysis (config,run,iter,gbs,'
    'integrity). On BAP it is redirected into WORKLOAD_ARTIFACTS_DIR so the '
    'workflow uploads it as a downloadable artifact.')
_ANALYZE = flags.DEFINE_bool(
    'analyze', False,
    'Collect + --dump + print distribution stats only. No gate, no baseline '
    'write; always exits 0. Use for the data-collection workflow.')

_LAYERS = 32   # C++ kNumLayers (fixed in the runner)
_SHARDS = 1    # C++ kNumShards (fixed in the runner)

# (block_size_bytes, num_blocks, parallelism). Kept to 1MB so sender+receiver do
# not exhaust host memory (16MB/128MB OOM the pod and drop BAP's :50051 channel).
_CONFIGS = [
    (1048576, 64, 1),
    (1048576, 64, 8),
    (1048573, 64, 4),
]

_CPP_P50_RE = re.compile(r'p50:\s*([0-9.]+)')          # "p50:   X ms"
_CPP_P90_RE = re.compile(r'p90:\s*([0-9.]+)')          # "p90:   X ms"
_CPP_P99_RE = re.compile(r'p99:\s*([0-9.]+)')          # "p99:   X ms"
_CPP_MEAN_RE = re.compile(r'Throughput:\s*([0-9.]+)')  # mean GB/s
# OPTIONAL per-iteration raw latency. If the runner prints one line per iter like
# "H2H_ITER_MS <ms>" (a ~3-line print loop over the latency vector it ALREADY
# builds to compute p50/p90/p99), this captures every raw sample -> one run gives
# all 50 points, no re-running. Absent, we fall back to the per-run median.
_CPP_RAW_RE = re.compile(r'H2H_ITER_MS\s+([0-9.]+)')
_INTEG_PASS = 'Data integrity verification PASSED'
_INTEG_FAIL = 'Data integrity verification FAILED'


def _baselines_path():
  """Baselines file: --baselines if given, else alongside this binary (runfiles).

  A CWD-relative path breaks under `bazel run` (CWD is the runfiles tree, not the
  workspace), so default to the copy shipped next to this .py via the BUILD data
  dep -- which is what the gate reads on BAP.
  """
  if _BASELINES.value:
    return _BASELINES.value
  return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'h2h_cpp_baselines.json')


def _label(bs, nb, p):
  return f'{bs}B_x{nb}_P{p}'


def _runfiles_root():
  """Runfiles ROOT, not just this binary's dir. The C++ runner is a data dep in
  a SIBLING tree (_main/examples/microbenchmarks/), so searching from
  dirname(__file__) (=.../benchmarks) misses it. Walk up to the enclosing
  '<name>.runfiles' (covers every repo) or the '_main' workspace root."""
  if os.environ.get('RUNFILES_DIR'):
    return os.environ['RUNFILES_DIR']
  d = os.path.dirname(os.path.abspath(__file__))
  main_root = None
  while d != os.path.dirname(d):
    if d.endswith('.runfiles'):
      return d
    if os.path.basename(d) == '_main':
      main_root = d
    d = os.path.dirname(d)
  return main_root or os.path.dirname(os.path.abspath(__file__))


def _locate(basename):
  """Find the C++ runner binary in the bazel runfiles by basename."""
  root = _runfiles_root()
  for dirpath, _, files in os.walk(root, followlinks=True):
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


def _f(regex, text):
  m = regex.search(text or '')
  return float(m.group(1)) if m else -1.0


def _run_cpp(cc, bs, nb, p, port):
  """Spawn the C++ receiver (bg) + sender (timed) for ONE run.

  Returns a dict: gbs (median throughput from p50 latency), mean_gbs, p50_ms,
  p90_ms, p99_ms, integrity (bool). gbs is -1.0 on failure.
  """
  base = [cc, '--data_interface=lo', f'--peer_control_port={port}',
          f'--block_size={bs}', f'--num_blocks={nb}', f'--parallelism={p}']
  # 50 is the runner's built-in default; only pass --iterations when overriding,
  # so the correctness gate works against a runner that lacks the flag. Custom
  # iters (e.g. analyze --iters=500) does require the runner's --iterations support.
  if _ITERS.value != 50:
    base.append(f'--iterations={_ITERS.value}')
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
  # After the sender finishes, the receiver runs a byte-by-byte integrity check
  # over the whole 2GB buffer, prints PASSED/FAILED, then exits on its own.
  # Terminating it immediately races that check -> false CORRUPT. In record/gate
  # we wait for it to finish and self-exit; in analyze we don't gate on integrity,
  # so skip the (slow) wait and kill it right away.
  if _ANALYZE.value:
    try:
      recv.terminate()
      recv_out = recv.communicate(timeout=15)[0].decode('utf-8', 'replace')
    except Exception:  # pylint: disable=broad-exception-caught
      recv.kill()
      recv_out = ''
  else:
    try:
      recv_out = recv.communicate(timeout=_TIMEOUT_S.value)[0].decode('utf-8', 'replace')
    except subprocess.TimeoutExpired:
      recv.terminate()
      try:
        recv_out = recv.communicate(timeout=15)[0].decode('utf-8', 'replace')
      except Exception:  # pylint: disable=broad-exception-caught
        recv.kill()
        recv_out = ''

  integrity_ok = (_INTEG_PASS in recv_out) and (_INTEG_FAIL not in recv_out)
  p50 = _f(_CPP_P50_RE, sender_out)
  p90 = _f(_CPP_P90_RE, sender_out)
  p99 = _f(_CPP_P99_RE, sender_out)
  mean_gbs = _f(_CPP_MEAN_RE, sender_out)
  total_bytes = _LAYERS * _SHARDS * nb * bs

  # Per-iteration raw samples if the runner emits them; else empty.
  raw_gbs = [(total_bytes / 1e9) / (float(ms) / 1000.0)
             for ms in _CPP_RAW_RE.findall(sender_out or '') if float(ms) > 0]

  gbs = -1.0
  if p50 > 0:
    gbs = (total_bytes / 1e9) / (p50 / 1000.0)
  elif raw_gbs:
    gbs = statistics.median(raw_gbs)
  elif mean_gbs > 0:
    gbs = mean_gbs

  # In analyze mode the receiver is killed before its integrity check finishes,
  # so integrity_ok is expectedly False -- that is NOT a failure here; only a
  # missing throughput reading is. In record/gate, bad integrity IS a failure.
  failed = gbs < 0 or (not _ANALYZE.value and not integrity_ok)
  if failed:
    print(f'[cpp] {_label(bs, nb, p)} FAILED (rc={rc}, gbs={gbs:.3f}, '
          f'integrity_ok={integrity_ok}); sender output:\n{sender_out}\n'
          f'--- receiver tail ---\n{recv_out[-2000:]}', flush=True)
  return {'gbs': gbs, 'mean_gbs': mean_gbs, 'p50_ms': p50, 'p90_ms': p90,
          'p99_ms': p99, 'integrity': integrity_ok, 'raw_gbs': raw_gbs}


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


def _core_floor(samples, k, max_margin):
  """Gate floor: lower edge of the normal core, capped so it is never looser
  than max_margin (mirrors the d2h/h2d _core_floor exactly).

      floor = max(median - k * MAD_sigma,  median * (1 - max_margin))

  MAD_sigma = 1.4826 * median(|x - median|) is an outlier-resistant stddev, so a
  low tail does not drag the bound down; the cap keeps a noisy config from ending
  up looser than a flat max_margin.
  """
  med = statistics.median(samples)
  mad = statistics.median([abs(x - med) for x in samples]) if len(samples) > 1 else 0.0
  sigma = 1.4826 * mad
  core = med - k * sigma
  cap = med * (1.0 - max_margin)
  return max(core, cap)


def _pct(xs, q):
  """q-th percentile (0..100) of non-empty xs via linear interpolation."""
  xs = sorted(xs)
  if len(xs) == 1:
    return xs[0]
  pos = (len(xs) - 1) * (q / 100.0)
  lo = int(pos)
  hi = min(lo + 1, len(xs) - 1)
  return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def main(_):
  cc = _locate('h2h_benchmark_runner')
  runs = max(1, _RUNS_PER_CONFIG.value)
  print(f'[cpp-gate] C++ runner: {cc}  (runs={runs} per config)', flush=True)

  writer = dump = dump_path = None
  if _DUMP.value:
    dump_path = _DUMP.value
    # On BAP, land the CSV in WORKLOAD_ARTIFACTS_DIR so the workflow uploads it.
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      dump_path = os.path.join(adir, os.path.basename(dump_path))
    dump = open(dump_path, 'w', newline='')
    writer = csv.writer(dump)
    # iter = per-iteration index when the runner emits raw H2H_ITER_MS samples,
    # else -1 (the row is that run's median). mean_gbs/p50_ms/p90_ms/p99_ms are
    # the runner's own summary of the 50 internal iters -- available every run,
    # repeated on each raw row so every row is self-describing.
    writer.writerow(['config', 'run', 'iter', 'gbs', 'mean_gbs', 'p50_ms',
                     'p90_ms', 'p99_ms', 'integrity'])

  # results[label] = representative {gbs (median across all samples), integrity}
  results = {}
  for i, (bs, nb, p) in enumerate(_CONFIGS):
    label = _label(bs, nb, p)
    port = _CONTROL_PORT.value + i
    series, integ_all = [], True
    print(f'[cpp-gate] ({i + 1}/{len(_CONFIGS)}) {label}: {runs} run(s) ...',
          flush=True)
    for run in range(runs):
      m = _run_cpp(cc, bs, nb, p, port)
      integ_all = integ_all and m['integrity']
      # Prefer per-iteration raw samples (one run -> 50 points); else the run's
      # single median.
      if m['raw_gbs']:
        samples = [(j, g) for j, g in enumerate(m['raw_gbs'])]
      elif m['gbs'] > 0:
        samples = [(-1, m['gbs'])]
      else:
        samples = []
      series.extend(g for _, g in samples)
      if writer:
        for it, g in samples:
          writer.writerow([label, run, it, f'{g:.4f}', f'{m["mean_gbs"]:.4f}',
                           f'{m["p50_ms"]:.4f}', f'{m["p90_ms"]:.4f}',
                           f'{m["p99_ms"]:.4f}', int(m['integrity'])])
        dump.flush()
      if runs > 1 and (run + 1) % 10 == 0:
        print(f'    {label}: {run + 1}/{runs} runs done', flush=True)

    if series:
      med = _pct(series, 50)
      print(f'[measured] {label:<22} n={len(series):<4} median={med:8.3f}  '
            f'p10={_pct(series, 10):7.3f}  p90={_pct(series, 90):7.3f}  '
            f'min={min(series):7.3f}  max={max(series):7.3f}  '
            f'stdev={statistics.pstdev(series):6.3f} GB/s  '
            f'integrity={"n/a" if _ANALYZE.value else ("OK" if integ_all else "CORRUPT")}',
            flush=True)
    else:
      med = -1.0
      print(f'[measured] {label}: ALL {runs} run(s) failed', flush=True)
    results[label] = {'gbs': med, 'integrity': integ_all, 'samples': series}
    _emit(f'{label}/cpp_gbs', med)

  if dump:
    dump.close()
    print(f'\nWrote samples for {len(_CONFIGS)} config(s) x {runs} run(s) -> '
          f'{dump_path}', flush=True)

  if _ANALYZE.value:
    print('\nanalyze mode: data collected, no gate/baseline. Done.', flush=True)
    return

  if _RECORD.value:
    # baseline = median of all samples; floor = median - k*MADsigma capped at
    # max_margin (same _core_floor as d2h/h2d). Record from MANY samples (large
    # --runs, or the runner's raw H2H_ITER_MS) so the MAD estimate is stable.
    k, cap = _SIGMA_K.value, _MAX_MARGIN.value
    cfg = {'sigma_k': k, 'max_margin': cap, 'configs': {}}
    for label, r in results.items():
      s = r['samples']
      cfg['configs'][label] = {
          'baseline_gbs': round(_pct(s, 50), 3) if s else 0.0,
          'floor_gbs': round(_core_floor(s, k, cap), 3) if s else 0.0,
          'integrity': r['integrity'],
          'n_samples': len(s),
      }
    # On BAP the runfiles tree is read-only, so write into WORKLOAD_ARTIFACTS_DIR
    # (BAP uploads it as a downloadable artifact you then commit); locally, write
    # straight to the file.
    out_path = _baselines_path()
    adir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if adir:
      out_path = os.path.join(adir, 'h2h_cpp_baselines.json')
    with open(out_path, 'w') as f:
      json.dump(cfg, f, indent=2)
    print(f'\nRecorded {len(results)} C++ H2H baselines+floors '
          f'(sigma_k={k}, cap={cap*100:.0f}%) -> {out_path}', flush=True)
    return

  # Gate mode: CORRECTNESS ONLY. On a single machine H2H runs over loopback, so
  # the throughput is not a product metric -- pass/fail is the receiver's
  # byte-integrity check alone. Throughput is printed + emitted for observability
  # but never fails the build, and no baseline/floor is needed (integrity
  # self-checks against the runner's deterministic byte pattern).
  bad = []
  print('\nH2H C++ correctness gate (single-machine loopback; throughput is informational only)\n')
  print('config                    median   integrity  verdict')
  print('-' * 58)
  for label, r in results.items():
    integ = r['integrity']
    print(f'{label:<22} {r["gbs"]:8.3f}  {"OK" if integ else "CORRUPT":<9} '
          f'{"PASS" if integ else "FAIL"}{"" if integ else " <-- DATA CORRUPTION"}')
    if not integ:
      bad.append(label)

  if bad:
    print(f'\nGATE FAIL: byte-integrity failed on {len(bad)} config(s): {bad}',
          file=sys.stderr)
    sys.exit(1)
  print('\nGATE PASS: all configs byte-exact.', flush=True)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
