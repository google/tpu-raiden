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

For each first-parent commit in the last DAYS on BRANCH, extracts that commit's
FULL tree with `git archive | tar` into a clean dir (this sidesteps the sparse /
partial working tree that plain `git checkout` leaves in BAP's checkout),
overlays today's .bazelrc + torch_tpu_stub from BRANCH so old commits build,
injects a tiny measure package, and builds+runs the float32 L64 measurement.
Each unique commit is built once (cached). All on one machine.
"""
import os
import re
import shutil
import subprocess
import sys

# ============================ EDIT THE WINDOW HERE ============================
DAYS = os.environ.get('DAYS', '7')                 # days back to sweep (past week)
BRANCH = os.environ.get('BRANCH', 'origin/main')   # history to walk + infra source
THRESH_PCT = float(os.environ.get('THRESH_PCT', '5'))  # drop % that flags a commit
# =============================================================================

_SRC = '/tmp/probe_src'               # clean extraction dir for the commit tree
_OB = '/tmp/probe_ob'                 # isolated output_base for the inner builds
_RUN_PKG = 'tpu_raiden/benchmarks/probe_run'
_STUB_PATH = 'third_party/torch_tpu_stub'   # referenced by .bazelrc oss config

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
  """Extract sha's tree, build+measure the L64 config; cache (None if broken)."""
  if sha in cache:
    return cache[sha]
  res = None
  try:
    shutil.rmtree(_SRC, ignore_errors=True)
    os.makedirs(_SRC)
    # Full commit tree, independent of the working tree's sparse/partial state.
    subprocess.run(f'git archive {sha} | tar -x -C {_SRC}', shell=True, check=True,
                   env=env, stderr=subprocess.DEVNULL)
    if not os.path.exists(os.path.join(_SRC, 'tpu_raiden/benchmarks/BUILD')):
      print(f'  [warn] {sha[:9]}: benchmarks/BUILD missing in archive', flush=True)
    # Overlay today's build infra from BRANCH (old commits lack the oss config /
    # the stub). Pulled from git objects, not the working tree.
    rc = subprocess.run(['git', 'show', f'{BRANCH}:.bazelrc'], env=env,
                        capture_output=True, text=True)
    if rc.returncode == 0:
      with open(os.path.join(_SRC, '.bazelrc'), 'w') as f:
        f.write(rc.stdout)
    subprocess.run(f'git archive {BRANCH} {_STUB_PATH} | tar -x -C {_SRC}',
                   shell=True, env=env, stderr=subprocess.DEVNULL)
    # Inject the measure package.
    run_dir = os.path.join(_SRC, _RUN_PKG)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'measure.py'), 'w') as f:
      f.write(_MEASURE_PY)
    with open(os.path.join(run_dir, 'BUILD'), 'w') as f:
      f.write(_BUILD_INJECT)
    t = '//tpu_raiden/benchmarks/probe_run:measure'
    subprocess.run(['bazel', f'--output_base={_OB}', 'build', '-c', 'opt',
                    '--config=oss', '--config=ci', t], check=True, env=env, cwd=_SRC)
    out = subprocess.run(['bazel', f'--output_base={_OB}', 'run', '-c', 'opt',
                          '--config=oss', '--config=ci', t], check=True, env=env,
                         cwd=_SRC, capture_output=True, text=True).stdout
    m = re.search(r'd2h=([0-9.]+) h2d=([0-9.]+)', out)
    if m:
      res = (float(m.group(1)), float(m.group(2)))
  except subprocess.CalledProcessError:
    res = None
  finally:
    shutil.rmtree(_SRC, ignore_errors=True)
  cache[sha] = res
  return res


_FULL = '/tmp/full_repo'              # complete clone (BAP's checkout is partial)


def main():
  ws = os.environ.get('BUILD_WORKSPACE_DIRECTORY') or os.getcwd()
  os.chdir(ws)
  env = dict(os.environ)

  # BAP's checkout is shallow+partial: many blob/tree objects are not local, and
  # git archive/show do NOT backfill them, so they yield incomplete trees. Do a
  # full clone of the same remote (reusing whatever URL/creds BAP configured) and
  # run every git operation there.
  url = subprocess.run(['git', 'remote', 'get-url', 'origin'], check=True, env=env,
                       capture_output=True, text=True).stdout.strip()
  shutil.rmtree(_FULL, ignore_errors=True)
  subprocess.run(['git', 'clone', '--no-single-branch', url, _FULL],
                 check=True, env=env)
  os.chdir(_FULL)

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
