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

"""Self-contained window sweep: which commit made D2H/H2D worse?

Walks every first-parent commit from the last DAYS on BRANCH, oldest to newest,
measures the float32 L64 d2h/h2d at each commit and its parent on the SAME
machine (one job, each commit built once and cached), and prints a table plus
the first commit that stepped down.

Self-contained on purpose: the per-commit measure.py and its BUILD are embedded
as strings below (no runfiles `data`, which BAP's manifest-mode runfiles do not
expose), and the window is set by the constants right here (BAP's custom_env_vars
is not reliably split into separate vars). To change the window, edit DAYS /
BRANCH / THRESH_PCT below.
"""
import os
import re
import shutil
import subprocess
import sys

# ============================ EDIT THE WINDOW HERE ============================
DAYS = os.environ.get('DAYS', '7')                 # days back to sweep (past week)
BRANCH = os.environ.get('BRANCH', 'origin/main')   # history to walk
THRESH_PCT = float(os.environ.get('THRESH_PCT', '5'))  # drop % that flags a commit
# =============================================================================

_OB = '/tmp/probe_ob'                 # isolated output_base for the inner builds
_RUN = 'tpu_raiden/benchmarks/probe_run'
_KEEP_RC = '/tmp/keep.bazelrc'        # today's .bazelrc, forced onto every commit

# The judge, injected+built at each commit (float32 L64 == the sensitive config).
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
  """Build+measure the L64 config at sha once; cache result (None if broken)."""
  if sha in cache:
    return cache[sha]
  res = None
  try:
    subprocess.run(['git', 'checkout', '--force', sha], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Old commits predate the `oss`/`ci` .bazelrc configs; force today's rc.
    if os.path.exists(_KEEP_RC):
      shutil.copy(_KEEP_RC, '.bazelrc')
    shutil.rmtree(_RUN, ignore_errors=True)
    os.makedirs(_RUN)
    with open(os.path.join(_RUN, 'measure.py'), 'w') as f:
      f.write(_MEASURE_PY)
    with open(os.path.join(_RUN, 'BUILD'), 'w') as f:
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
    shutil.rmtree(_RUN, ignore_errors=True)
  cache[sha] = res
  return res


def main():
  ws = os.environ.get('BUILD_WORKSPACE_DIRECTORY') or os.getcwd()
  os.chdir(ws)
  env = dict(os.environ)

  if os.path.exists('.bazelrc'):
    shutil.copy('.bazelrc', _KEEP_RC)  # stash today's rc for the old commits
  subprocess.run(['git', 'fetch', '--unshallow'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(['git', 'fetch', '--all', '--tags'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
