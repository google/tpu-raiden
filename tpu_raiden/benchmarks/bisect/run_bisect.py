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

"""BAP workload: run the WHOLE git-bisect in one job (== one TPU machine).

Env in: GOOD_SHA, BAD_SHA, THRESH (optional, default 760).
Drives `git bisect run run_one.sh`, where run_one.sh builds+measures the float32
L64 config at each candidate commit and returns fast/slow/skip. Writes the
culprit to $WORKLOAD_ARTIFACTS_DIR/bisect_result.txt and the job log.

Pure git orchestration here (no bazel in this process); the per-commit build
happens inside run_one.sh with an isolated --output_base, so the outer bazel that
launched this binary is never disturbed.
"""
import os
import re
import shutil
import stat
import subprocess
import sys


def _here():
  return os.path.dirname(os.path.abspath(__file__))


def main():
  ws = os.environ.get('BUILD_WORKSPACE_DIRECTORY') or os.getcwd()
  os.chdir(ws)
  good = os.environ['GOOD_SHA']
  bad = os.environ['BAD_SHA']
  thresh = os.environ.get('THRESH', '760')

  # Stash the harness outside the tree; bisect checks out commits without bisect/.
  harness = '/tmp/bisect_harness'
  shutil.rmtree(harness, ignore_errors=True)
  os.makedirs(harness)
  for f in ('measure.py', 'BUILD.inject', 'run_one.sh'):
    shutil.copy(os.path.join(_here(), f), os.path.join(harness, f))
  ro = os.path.join(harness, 'run_one.sh')
  os.chmod(ro, os.stat(ro).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

  env = dict(os.environ, HARNESS=harness, THRESH=thresh)
  # Need full history between good..bad.
  subprocess.run(['git', 'fetch', '--unshallow'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  subprocess.run(['git', 'fetch', '--all', '--tags'], env=env,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  print(f'=== bisecting {good} (good) .. {bad} (bad), THRESH={thresh} d2h Gbps ===',
        flush=True)
  subprocess.run(['git', 'bisect', 'start'], check=True, env=env)
  subprocess.run(['git', 'bisect', 'bad', bad], check=True, env=env)
  subprocess.run(['git', 'bisect', 'good', good], check=True, env=env)
  proc = subprocess.run(['git', 'bisect', 'run', ro], env=env,
                        capture_output=True, text=True)
  log = proc.stdout + proc.stderr
  print(log, flush=True)
  subprocess.run(['git', 'bisect', 'reset'], env=env)

  m = re.search(r'([0-9a-f]{7,40}) is the first bad commit', log)
  culprit = m.group(1) if m else None

  out_dir = os.environ.get('WORKLOAD_ARTIFACTS_DIR', '/tmp')
  lines = [f'First bad commit: {culprit or "<not found>"}']
  if culprit:
    show = subprocess.run(['git', 'show', '-s', '--format=%h  %an  %ad%n%s',
                           culprit], capture_output=True, text=True)
    lines += ['', show.stdout.strip()]
  report = '\n'.join(lines) + '\n'
  with open(os.path.join(out_dir, 'bisect_result.txt'), 'w') as f:
    f.write(report)
  print('\n' + report)
  sys.exit(0 if culprit else 1)


if __name__ == '__main__':
  main()
