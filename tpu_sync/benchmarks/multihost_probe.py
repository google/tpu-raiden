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

"""Reports what the multi-host runner looks like from inside a BAP job.

A GitHub Actions job lands on exactly one node, so a real cross-host H2H run
needs a second host obtained some other way -- submitting a JobSet from here is
the only candidate. This probe answers whether that is possible, and collects
the values the driver will need:

  * which interface to pass as --data_interface,
  * whether kubectl exists and the job's identity may create/read a JobSet,
  * NUMA layout and memory, which set the block_size/num_blocks ceiling.

Stdlib only (the py_binary has no deps) and always exits 0: a probe that fails
the job teaches less than one that prints what it found.
"""

import os
import shutil
import subprocess
import sys

_TIMEOUT_S = 60


def _section(title):
  print(f'\n{"=" * 72}\n== {title}\n{"=" * 72}', flush=True)


def _run(cmd, label=None):
  """Runs cmd, prints its output, returns (rc, combined_output) or None."""
  print(f'\n$ {" ".join(cmd)}' + (f'   # {label}' if label else ''), flush=True)
  if not shutil.which(cmd[0]):
    print(f'  [{cmd[0]} is not on PATH]', flush=True)
    return None
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
  except subprocess.TimeoutExpired:
    print(f'  [timed out after {_TIMEOUT_S}s]', flush=True)
    return None
  out = (p.stdout + p.stderr).strip()
  print(out if out else '  [no output]', flush=True)
  return p.returncode, out


def _cat(path):
  print(f'\n$ cat {path}', flush=True)
  try:
    with open(path) as f:
      print(f.read().strip(), flush=True)
    return True
  except OSError as e:
    print(f'  [{e}]', flush=True)
    return False


def main():
  findings = {}

  _section('1. Node identity')
  _run(['hostname', '-f'])
  _run(['uname', '-a'])
  _cat('/etc/os-release')
  # A pod's cgroup path names its container; a plain VM runner's does not.
  _cat('/proc/1/cgroup')

  _section('2. Network interfaces  (--data_interface comes from here)')
  nics = _run(['ip', '-br', 'link'])
  _run(['ip', '-br', 'addr'])
  _run(['ip', 'route'])
  if nics:
    names = [ln.split()[0].split('@')[0] for ln in nics[1].splitlines() if ln.split()]
    findings['nics'] = [n for n in names if n != 'lo']

  _section('3. CPU / NUMA / memory  (sets the block_size x num_blocks ceiling)')
  _run(['lscpu'])
  _run(['numactl', '-H'])
  _run(['free', '-g'])

  _section('4. TPU devices  (H2H does not need them; confirming the node type)')
  _run(['ls', '-l', '/dev/accel0'])
  _run(['bash', '-c', 'ls /dev | grep -iE "accel|vfio" || echo "  [none]"'])

  _section('5. Kubernetes context')
  k8s_env = {k: v for k, v in os.environ.items() if k.startswith('KUBERNETES_')}
  print(f'\nKUBERNETES_* env: {k8s_env or "[none]"}', flush=True)
  sa_dir = '/var/run/secrets/kubernetes.io/serviceaccount'
  has_sa = os.path.isdir(sa_dir)
  print(f'service account dir {sa_dir}: '
        f'{sorted(os.listdir(sa_dir)) if has_sa else "[absent]"}', flush=True)
  findings['in_cluster'] = bool(k8s_env) and has_sa
  kubectl = _run(['kubectl', 'version', '--client', '-o', 'yaml'])
  findings['kubectl'] = kubectl is not None

  _section('6. RBAC  (can this job get a second host?)')
  checks = {
      'create_jobset': ['create', 'jobsets.jobset.x-k8s.io'],
      'get_jobset': ['get', 'jobsets.jobset.x-k8s.io'],
      'delete_jobset': ['delete', 'jobsets.jobset.x-k8s.io'],
      'create_job': ['create', 'jobs.batch'],
      'get_pod': ['get', 'pods'],
      'read_pod_log': ['get', 'pods/log'],
      'list_node': ['list', 'nodes'],
  }
  for name, verb_res in checks.items():
    r = _run(['kubectl', 'auth', 'can-i'] + verb_res, label=name)
    findings[name] = bool(r and r[1].startswith('yes'))

  _section('7. Cluster shape')
  _run(['bash', '-c', 'kubectl get crd 2>&1 | grep -i jobset || echo "  [no jobset CRD visible]"'])
  _run(['kubectl', 'get', 'nodes', '-L',
        'cloud.google.com/gke-tpu-topology,cloud.google.com/gke-tpu-accelerator'])

  _section('VERDICT')
  nic_list = findings.get('nics', [])
  print(f'non-loopback NICs      : {nic_list or "NONE FOUND"}')
  print(f'running inside a pod   : {findings.get("in_cluster")}')
  print(f'kubectl available      : {findings.get("kubectl")}')
  jobset_ok = all(findings.get(k) for k in
                  ('create_jobset', 'get_jobset', 'read_pod_log'))
  print(f'can drive a JobSet     : {jobset_ok}')
  print()
  if jobset_ok:
    print('=> JobSet path is open: the driver submits a 2-replica JobSet and')
    print('   scrapes both pods\' logs.')
  elif findings.get('kubectl'):
    print('=> kubectl works but the identity lacks JobSet rights. Ask for a')
    print('   service account with create/get/delete on jobsets.jobset.x-k8s.io')
    print('   plus get on pods/log, in this cluster.')
  else:
    print('=> No cluster access from the job. A second host cannot be reached')
    print('   from here; multi-host H2H needs a different launch mechanism.')
  print('\nProbe complete (exit 0 regardless of findings).', flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
