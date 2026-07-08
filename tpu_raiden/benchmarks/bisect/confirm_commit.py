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

"""BAP workload: confirm ONE suspect commit (Path A), no git commands for you.

You dispatch confirm_commit.yml, type the suspect SHA in the Actions UI, click
Run. This measures the float32 L64 d2h/h2d at the suspect's PARENT and at the
SUSPECT on the SAME machine (one job), and prints the delta plus a verdict.

Env in:
  SUSPECT_SHA  (required)
  PARENT_SHA   (optional; default <SUSPECT_SHA>~1)
  THRESH_PCT   (optional; default 5.0 -- drop above this counts as REGRESSION)
Writes $WORKLOAD_ARTIFACTS_DIR/confirm_result.txt and the job log.
"""
import os
import re
import shutil
import subprocess
import sys

_OB = '/tmp/confirm_ob'          # isolated output_base for the inner builds
_RUN = 'tpu_raiden/benchmarks/confirm_run'


def _here():
  return os.path.dirname(os.path.abspath(__file__))


def _measure_at(sha, env):
  """Checkout sha, inject the L64 measure, build+run it, return (d2h, h2d)."""
  subprocess.run(['git', 'checkout', '--force', sha], check=True, env=env)
  shutil.rmtree(_RUN, ignore_errors=True)
  os.makedirs(_RUN)
  shutil.copy(os.path.join(_here(), 'measure.py'), os.path.join(_RUN, 'measure.py'))
  shutil.copy(os.path.join(_here(), 'BUILD.inject'), os.path.join(_RUN, 'BUILD'))
  target = '//tpu_raiden/benchmarks/confirm_run:measure'
  subprocess.run(['bazel', f'--output_base={_OB}', 'build', '-c', 'opt',
                  '--config=oss', target], check=True, env=env)
  out = subprocess.run(['bazel', f'--output_base={_OB}', 'run', '-c', 'opt',
                        '--config=oss', target], check=True, env=env,
                       capture_output=True, text=True).stdout
  shutil.rmtree(_RUN, ignore_errors=True)
  m = re.search(r'd2h=([0-9.]+) h2d=([0-9.]+)', out)
  if not m:
    raise RuntimeError(f'no BISECT line from measure at {sha}:\n{out}')
  return float(m.group(1)), float(m.group(2))


def main():
  ws = os.environ.get('BUILD_WORKSPACE_DIRECTORY') or os.getcwd()
  os.chdir(ws)
  suspect = os.environ['SUSPECT_SHA']
  thresh = float(os.environ.get('THRESH_PCT', '5.0'))
  env = dict(os.environ)

  subprocess.run(['git', 'fetch', '--unshallow'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(['git', 'fetch', '--all', '--tags'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  parent = os.environ.get('PARENT_SHA') or subprocess.run(
      ['git', 'rev-parse', f'{suspect}~1'], check=True, env=env,
      capture_output=True, text=True).stdout.strip()

  print(f'=== confirm: parent {parent} vs suspect {suspect} (float32 L64) ===',
        flush=True)
  pd, ph = _measure_at(parent, env)
  sd, sh = _measure_at(suspect, env)

  drop_d = (pd - sd) / pd * 100
  drop_h = (ph - sh) / ph * 100
  culprit = drop_d > thresh or drop_h > thresh
  report = (
      f'suspect {suspect} vs parent {parent}   (threshold {thresh:.1f}%)\n'
      f'  d2h: {pd:.1f} -> {sd:.1f}  ({drop_d:+.1f}%)\n'
      f'  h2d: {ph:.1f} -> {sh:.1f}  ({drop_h:+.1f}%)\n'
      f'  verdict: {"REGRESSION -- this commit is the culprit" if culprit else "no significant change"}\n')
  out_dir = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  with open(os.path.join(out_dir, 'confirm_result.txt'), 'w') as f:
    f.write(report)
  print('\n' + report)
  sys.exit(0)


if __name__ == '__main__':
  main()
