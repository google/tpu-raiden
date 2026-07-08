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

"""BAP workload: sweep EVERY commit in a window, each vs its parent, one machine.

Lists the first-parent commits from the last DAYS on BRANCH, measures the
float32 L64 d2h/h2d at each commit and its parent on the SAME machine (one job,
each unique commit built once and cached), and prints a table with the per-commit
drop vs its parent, flagging any that exceed THRESH_PCT.

Env in:
  DAYS        (default 8)          window length in days
  BRANCH      (default origin/main) history to walk
  THRESH_PCT  (default 5.0)        drop percent that flags a commit
Writes $WORKLOAD_ARTIFACTS_DIR/probe_result.txt and the job log.
"""
import os
import re
import shutil
import subprocess
import sys

_OB = '/tmp/probe_ob'
_RUN = 'tpu_raiden/benchmarks/probe_run'
_KEEP_RC = '/tmp/keep.bazelrc'   # today's .bazelrc, forced onto every old commit


def _here():
  return os.path.dirname(os.path.abspath(__file__))


def _measure_at(sha, env, cache):
  """Build+measure the L64 config at sha once; cache result (or None if broken)."""
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
    shutil.copy(os.path.join(_here(), 'measure.py'), os.path.join(_RUN, 'measure.py'))
    shutil.copy(os.path.join(_here(), 'BUILD.inject'), os.path.join(_RUN, 'BUILD'))
    t = '//tpu_raiden/benchmarks/probe_run:measure'
    subprocess.run(['bazel', f'--output_base={_OB}', 'build', '-c', 'opt',
                    '--config=oss', t], check=True, env=env)
    out = subprocess.run(['bazel', f'--output_base={_OB}', 'run', '-c', 'opt',
                          '--config=oss', t], check=True, env=env,
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
  days = os.environ.get('DAYS', '8')
  branch = os.environ.get('BRANCH', 'origin/main')
  thresh = float(os.environ.get('THRESH_PCT', '5.0'))
  env = dict(os.environ)

  if os.path.exists('.bazelrc'):
    shutil.copy('.bazelrc', _KEEP_RC)  # stash today's rc for the old commits
  subprocess.run(['git', 'fetch', '--unshallow'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(['git', 'fetch', '--all', '--tags'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  shas = subprocess.run(
      ['git', 'log', f'--since={days} days ago', '--first-parent', '--reverse',
       '--format=%H', branch],
      check=True, env=env, capture_output=True, text=True).stdout.split()
  print(f'{len(shas)} commits in the last {days} days on {branch}', flush=True)

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
    flag = 'REGRESSION <--' if (dd > thresh or dh > thresh) else ''
    rows.append((sha[:9], f'{cs[0]:.1f}/{cs[1]:.1f}',
                 f'{dd:+.1f}%', f'{dh:+.1f}%', f'{flag}  {subj}'))
    print(f'  {sha[:9]}  d2h {cs[0]:.1f} ({dd:+.1f}%)  h2d {cs[1]:.1f} ({dh:+.1f}%)'
          f'  {flag}', flush=True)

  # oldest -> newest walk; the first row that steps down is the culprit.
  hdr = f"{'#':3}{'commit':11}{'d2h/h2d':16}{'d2h_drop':10}{'h2d_drop':10}verdict / subject"
  table = [hdr, '-' * len(hdr)]
  for i, r in enumerate(rows):
    table.append(f'{i:<3}{r[0]:11}{r[1]:16}{r[2]:10}{r[3]:10}{r[4]}')

  first = next((r for r in rows if 'REGRESSION' in r[4]), None)
  bad = [r[0] for r in rows if 'REGRESSION' in r[4]]
  headline = (f'==> FIRST COMMIT THAT REGRESSED: {first[0]}  ({first[4].split("  ", 1)[-1]})'
              if first else '==> no commit dropped more than '
              f'{thresh:.0f}% in this window')

  lines = ['', headline, '',
           f'walked {len(rows)} commits oldest -> newest '
           f'(float32 L64, threshold {thresh:.0f}%):', ''] + table
  lines += ['', f'all flagged commits (drop > {thresh:.0f}%): '
            + (', '.join(bad) or 'none')]
  report = '\n'.join(lines) + '\n'

  out_dir = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out_dir, 'probe_result.txt'), 'w') as f:
    f.write(report)
  print(report)
  sys.exit(0)


if __name__ == '__main__':
  main()
