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

"""Self-contained window sweep: which commit made D2H/H2D worse?"""
import os
import re
import shutil
import subprocess
import sys

# ============================ EDIT THE WINDOW HERE ============================
DAYS = os.environ.get('DAYS', '7')
BRANCH = os.environ.get('BRANCH', 'origin/main')
THRESH_PCT = float(os.environ.get('THRESH_PCT', '5'))
CLONE_URL = os.environ.get('CLONE_URL', 'https://github.com/google/tpu-raiden.git')
# =============================================================================

_OB = '/tmp/probe_ob'
_FULL = '/tmp/full_repo'
_KEEP_RC = '/tmp/keep.bazelrc'
_KEEP_STUB = '/tmp/keep_torch_tpu_stub'
_RUN_PKG = 'tpu_raiden/benchmarks/probe_run'
_STUB_PATH = 'third_party/torch_tpu_stub'

_MEASURE_PY = '''\
import sys
from tpu_raiden.benchmarks import perf_core
r = perf_core.measure(shape=[16, 128, 8, 2, 128], num_layers=64, dtype='float32',
                      shard_axis=2, iters=15, warmup=3)
print(f'BISECT d2h={r["d2h_gbps"]:.1f} h2d={r["h2d_gbps"]:.1f}')
sys.stdout.flush()
'''

_BUILD_INJECT = '''\
load("@rules_python//python:defs.bzl", "py_binary")
py_binary(
    name = "measure",
    testonly = True,
    srcs = ["measure.py"],
    deps = ["//tpu_raiden/benchmarks:perf_core"],
)
'''


def _measure_at(sha, env, cache):
  # runs inside the full clone (cwd == _FULL)
  if sha in cache:
    return cache[sha]
  res = None
  try:
    subprocess.run(['git', 'checkout', '--force', sha], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists('tpu_raiden/benchmarks/BUILD'):
      print(f'  [warn] {sha[:9]}: benchmarks/BUILD missing after checkout', flush=True)
    if os.path.exists(_KEEP_RC):
      shutil.copy(_KEEP_RC, '.bazelrc')
    if os.path.isdir(_KEEP_STUB):
      shutil.copytree(_KEEP_STUB, _STUB_PATH, dirs_exist_ok=True)
    os.makedirs(_RUN_PKG, exist_ok=True)
    with open(os.path.join(_RUN_PKG, 'measure.py'), 'w') as f:
      f.write(_MEASURE_PY)
    with open(os.path.join(_RUN_PKG, 'BUILD'), 'w') as f:
      f.write(_BUILD_INJECT)
    t = '//tpu_raiden/benchmarks/probe_run:measure'
    subprocess.run(['bazel', f'--output_base={_OB}', 'build', '-c', 'opt',
                    '--config=oss', '--config=ci', t], check=True, env=env)
    out = subprocess.run(['bazel', f'--output_base={_OB}', 'run', '-c', 'opt',
                          '--config=oss', '--config=ci', t], check=True, env=env,
                         capture_output=True, text=True).stdout
    m = re.search(r'd2h=([0-9.]+) h2d=([0-9.]+)', out)
    if m:
      res = (float(m.group(1)), float(m.group(2)))
  except subprocess.CalledProcessError:
    res = None
  finally:
    shutil.rmtree(_RUN_PKG, ignore_errors=True)
  cache[sha] = res
  return res


def main():
  env = dict(os.environ)

  # Fresh FULL clone from the public URL (BAP's local checkout is partial, and
  # cloning its local origin gave another partial repo -> missing objects).
  shutil.rmtree(_FULL, ignore_errors=True)
  subprocess.run(['git', 'clone', '--no-single-branch', CLONE_URL, _FULL],
                 check=True, env=env)
  os.chdir(_FULL)

  have_build = os.path.exists('tpu_raiden/benchmarks/BUILD')
  have_rc = os.path.exists('.bazelrc')
  print(f'[diag] fresh clone HEAD: benchmarks/BUILD={have_build} .bazelrc={have_rc}',
        flush=True)

  # Stash today's build infra from the freshly-cloned main HEAD.
  if have_rc:
    shutil.copy('.bazelrc', _KEEP_RC)
  if os.path.isdir(_STUB_PATH):
    shutil.rmtree(_KEEP_STUB, ignore_errors=True)
    shutil.copytree(_STUB_PATH, _KEEP_STUB)

  shas = subprocess.run(
      ['git', 'log', f'--since={DAYS} days ago', '--first-parent', '--reverse',
       '--format=%H', BRANCH],
      check=True, env=env, capture_output=True, text=True).stdout.split()
  print(f'{len(shas)} commits in the last {DAYS} days on {BRANCH}', flush=True)

  cache = {}
  rows = []
  for sha in shas:
    parent = subprocess.run(['git', 'rev-parse', f'{sha}~1'], check=True, env=env,
                            capture_output=True, text=True).stdout.strip()
    cs = _measure_at(sha, env, cache)
    ps = _measure_at(parent, env, cache)
    subj = subprocess.run(['git', 'log', '-1', '--format=%s', sha], env=env,
                          capture_output=True, text=True).stdout.strip()[:44]
    if cs is None or ps is None:
      rows.append((sha[:9], 'skip', '', '', subj))
      continue
    dd = (ps[0] - cs[0]) / ps[0] * 100
    dh = (ps[1] - cs[1]) / ps[1] * 100
    flag = 'REGRESSION <--' if (dd > THRESH_PCT or dh > THRESH_PCT) else ''
    rows.append((sha[:9], f'{cs[0]:.1f}/{cs[1]:.1f}',
                 f'{dd:+.1f}%', f'{dh:+.1f}%', f'{flag}  {subj}'.strip()))
    print(f'  {sha[:9]}  d2h {cs[0]:.1f} ({dd:+.1f}%)  h2d {cs[1]:.1f} ({dh:+.1f}%)'
          f'  {flag}', flush=True)

  hdr = f"{'#':3}{'commit':11}{'d2h/h2d':16}{'d2h_drop':10}{'h2d_drop':10}verdict / subject"
  table = [hdr, '-' * len(hdr)]
  for i, r in enumerate(rows):
    table.append(f'{i:<3}{r[0]:11}{r[1]:16}{r[2]:10}{r[3]:10}{r[4]}')

  first = next((r for r in rows if 'REGRESSION' in r[4]), None)
  bad = [r[0] for r in rows if 'REGRESSION' in r[4]]
  headline = (f'==> FIRST COMMIT THAT REGRESSED: {first[0]}  '
              f'({first[4].split("  ", 1)[-1]})' if first
              else f'==> no commit dropped more than {THRESH_PCT:.0f}% in this window')

  lines = ['', headline, '',
           f'walked {len(rows)} commits oldest -> newest '
           f'(float32 L64, threshold {THRESH_PCT:.0f}%):', ''] + table
  lines += ['', f'all flagged commits (drop > {THRESH_PCT:.0f}%): '
            + (', '.join(bad) or 'none')]
  report = '\n'.join(lines) + '\n'

  out_dir = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out_dir, 'probe_result.txt'), 'w') as f:
    f.write(report)
  print(report)
  sys.exit(0)


if __name__ == '__main__':
  main()
