# Copyright 2026 Google LLC. Apache-2.0.
"""Window sweep (BAP): which commit made D2H/H2D worse?

BAP's checkout is a partial (filtered) + shallow clone, so historical file
blobs/trees are not local. This first FORCES git to download all objects
(remove the filter + git fetch --refetch + unshallow), verifies the files are
now reachable, then git-archives each commit and builds the float32 L64 measure.
"""
import os
import re
import shutil
import subprocess
import sys

DAYS = os.environ.get('DAYS', '7')
BRANCH = os.environ.get('BRANCH', 'origin/main')
THRESH_PCT = float(os.environ.get('THRESH_PCT', '5'))

_SRC = '/tmp/probe_src'
_OB = '/tmp/probe_ob'
_RUN_PKG = 'tpu_raiden/benchmarks/probe_run'
_STUB_PATH = 'third_party/torch_tpu_stub'
_DN = subprocess.DEVNULL

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
  if sha in cache:
    return cache[sha]
  res = None
  try:
    shutil.rmtree(_SRC, ignore_errors=True)
    os.makedirs(_SRC)
    subprocess.run(f'git archive {sha} | tar -x -C {_SRC}', shell=True, check=True,
                   env=env, stderr=_DN)
    rc = subprocess.run(['git', 'show', f'{BRANCH}:.bazelrc'], env=env,
                        capture_output=True, text=True)
    if rc.returncode == 0:
      with open(os.path.join(_SRC, '.bazelrc'), 'w') as f:
        f.write(rc.stdout)
    subprocess.run(f'git archive {BRANCH} {_STUB_PATH} | tar -x -C {_SRC}',
                   shell=True, env=env, stderr=_DN)
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


def main():
  ws = os.environ.get('BUILD_WORKSPACE_DIRECTORY') or os.getcwd()
  os.chdir(ws)
  env = dict(os.environ)

  # Force git to download ALL objects (BAP clones partially + shallow).
  subprocess.run(['git', 'config', '--unset-all', 'remote.origin.partialclonefilter'],
                 env=env, stdout=_DN, stderr=_DN)
  subprocess.run(['git', 'fetch', '--unshallow'], env=env, stdout=_DN, stderr=_DN)
  subprocess.run(['git', 'fetch', '--refetch', '--tags', 'origin'],
                 env=env, stdout=_DN, stderr=_DN)

  # Decisive check: is the actual file object now present locally?
  probe = subprocess.run(['git', 'cat-file', '-e', f'{BRANCH}:tpu_raiden/benchmarks/BUILD'],
                         env=env)
  ok = probe.returncode == 0
  print(f'[diag] after refetch: {BRANCH}:tpu_raiden/benchmarks/BUILD present={ok}',
        flush=True)
  if not ok:
    print('FATAL: container still cannot obtain file objects. Only BAP can fix '
          'this by doing a full, non-filtered checkout. Stopping.',
          file=sys.stderr, flush=True)
    sys.exit(1)

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
  print('\n' + headline + '\n')
  print(f'walked {len(rows)} commits oldest -> newest '
        f'(float32 L64, threshold {THRESH_PCT:.0f}%):\n')
  print('\n'.join(table))
  print('\nall flagged: ' + (', '.join(bad) or 'none'))
  out_dir = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out_dir, 'probe_result.txt'), 'w') as f:
    f.write('\n'.join([headline, ''] + table) + '\n')


if __name__ == '__main__':
  main()
