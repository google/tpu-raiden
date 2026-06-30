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

"""Single-host cross-NUMA OSS H2H (host-to-host) KV-cache transfer benchmark.

Runs BOTH roles on ONE machine, each pinned to a different NUMA node, so the
host-to-host transfer crosses the inter-socket interconnect (UPI / Infinity
Fabric). Measures cross-NUMA bandwidth (Gbps) and verifies data integrity.

NOTE: loopback traffic does NOT egress a physical NIC, so the number reflects
cross-socket memory bandwidth + kernel net-stack, not wire bandwidth (B-tier).
If the container's cpuset only covers one NUMA node, binding degrades to A-tier
(intra-node loopback) automatically.

Roles (--role):
  launch   : (default) spawn sender (NUMA A) + receiver (NUMA B), supervise, exit.
  sender   : hold source KV cache, notify reads round-by-round.
  receiver : pull the cache via start_read, time warmup+iters, verify, emit metrics.
"""

import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from tpu_raiden.frameworks.jax import _tpu_raiden_jax as kv_cache_manager

_ROLE = flags.DEFINE_enum('role', 'launch', ['launch', 'sender', 'receiver'],
                          'Process role. "launch" = single-host driver.')
_RENDEZVOUS = flags.DEFINE_string('rendezvous', '',
                                  'Shared dir for s.json/r.json (set by launcher).')
_BIND_NUMA = flags.DEFINE_integer('bind_numa', -1,
                                  'Bind this process to NUMA node (cpu+mem). -1 = none.')
_NUM_BLOCKS = flags.DEFINE_integer('num_blocks', 4096, 'Cache blocks to transfer.')
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 2, 'Middle cache dim.')
_NUM_LAYERS = flags.DEFINE_integer('num_layers', 8, 'Number of cache arrays (layers).')
_PARALLELISM = flags.DEFINE_integer('parallelism', 8,
                                    'Parallel TCP streams / transfer slots for H2H.')
_DTYPE = flags.DEFINE_string('dtype', 'bfloat16', 'float32 | bfloat16 | float16.')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Warmup transfers (not timed).')
_ITERS = flags.DEFINE_integer('iters', 20, 'Timed transfers.')
_SENDER_NUMA = flags.DEFINE_integer('sender_numa', 0, 'NUMA node for sender.')
_RECEIVER_NUMA = flags.DEFINE_integer('receiver_numa', 1, 'NUMA node for receiver.')
_LAUNCH_TIMEOUT_S = flags.DEFINE_integer('launch_timeout_s', 1200,
                                         'Hard cap on the whole single-host run.')

_DTYPE_MAP = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16, 'float16': jnp.float16}
_ITEMSIZE = {'float32': 4, 'bfloat16': 2, 'float16': 2}


# ---------------- generic helpers ----------------
def summarize(values):
  a = np.array(values, dtype=float)
  return {'min': float(a.min()), 'p50': float(np.median(a)), 'mean': float(a.mean()),
          'p90': float(np.percentile(a, 90)), 'p99': float(np.percentile(a, 99)),
          'max': float(a.max()),
          'stddev': float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def numa_nodes():
  try:
    return sorted(int(n[4:]) for n in os.listdir('/sys/devices/system/node')
                  if n.startswith('node') and n[4:].isdigit())
  except OSError:
    return []


def _cpus_of_node(node):
  try:
    spec = open(f'/sys/devices/system/node/node{node}/cpulist').read().strip()
  except OSError:
    return set()
  cpus = set()
  for part in spec.split(','):
    if '-' in part:
      a, b = part.split('-')
      cpus.update(range(int(a), int(b) + 1))
    elif part:
      cpus.add(int(part))
  return cpus


def bind_to_numa(node):
  """numactl-free NUMA bind: CPU affinity (intersection with allowed cores) +
  set_mempolicy(MPOL_BIND). Degrades gracefully if the container's cpuset has no
  cores on `node` (e.g. cgroup pinned to one socket)."""
  if node is None or node < 0:
    return
  want = _cpus_of_node(node)
  allowed = os.sched_getaffinity(0)
  cpus = want & allowed
  if not cpus:
    print(f'WARNING: no allowed cores on NUMA node {node} '
          f'(want={len(want)}, allowed={len(allowed)}); container likely pinned '
          'to one node -> B-tier degrades to A-tier.', file=sys.stderr)
  else:
    try:
      os.sched_setaffinity(0, cpus)
    except OSError as e:
      print(f'WARNING: sched_setaffinity failed: {e}', file=sys.stderr)
  try:
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    MPOL_BIND, SYS_set_mempolicy = 2, 238  # SYS_set_mempolicy=238 on x86_64
    nodemask = ctypes.c_ulong(1 << node)
    rc = libc.syscall(SYS_set_mempolicy, MPOL_BIND,
                      ctypes.byref(nodemask), ctypes.c_ulong(64))
    if rc != 0:
      print(f'WARNING: set_mempolicy errno={ctypes.get_errno()}', file=sys.stderr)
    else:
      print(f'[bind] process pinned to NUMA node {node} (mem; cpus={len(cpus)}).')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: NUMA membind unavailable: {e}', file=sys.stderr)


def _wait_until(cond, timeout, what):
  t0 = time.time()
  while not cond():
    if time.time() - t0 > timeout:
      raise TimeoutError(f'timed out after {timeout}s waiting for {what}')
    time.sleep(0.005)


# ---------------- endpoint resolution (IPv4 + IPv6) ----------------
def _primary_endpoint(port):
  """Self-probed primary IP, IPv4 first then IPv6, formatted correctly."""
  for family, probe in ((socket.AF_INET, ('8.8.8.8', 80)),
                        (socket.AF_INET6, ('2001:4860:4860::8888', 80))):
    s = socket.socket(family, socket.SOCK_DGRAM)
    try:
      s.connect(probe)  # no packet sent; just resolves the egress local addr
      ip = s.getsockname()[0]
      return f'[{ip}]:{port}' if family == socket.AF_INET6 else f'{ip}:{port}'
    except OSError:
      continue
    finally:
      s.close()
  return f'127.0.0.1:{port}'


def _resolve_endpoints(manager):
  """Authoritative endpoints the manager bound to (get_local_endpoints returns
  a list of {'endpoint': str, 'shards': [...]}); falls back to a self-probe."""
  try:
    eps = manager.get_local_endpoints()
    out = []
    for e in eps:
      out.append(e if isinstance(e, str) else e.get('endpoint', str(e)))
    out = [e for e in out if e]
    if out:
      return out
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: get_local_endpoints unavailable ({e}); probing.', file=sys.stderr)
  return [_primary_endpoint(manager.local_control_port)]


# ---------------- cache helpers ----------------
def cache_shape():
  return (_NUM_BLOCKS.value, 32, _BLOCK_SIZE.value, 8, 128)


def block_byte_size():
  return int(np.prod(cache_shape()[1:])) * _ITEMSIZE[_DTYPE.value]


def cpu_sharding():
  devices = jax.devices('cpu')
  mesh = jax.sharding.Mesh(np.array(devices).reshape(1, len(devices)), ('data', 'model'))
  spec = jax.sharding.PartitionSpec(None, None, 'model')
  return jax.sharding.NamedSharding(mesh, spec)


def _layer_base(layer):
  """Deterministic float32 source for a layer (identical on both roles)."""
  return (np.arange(np.prod(cache_shape()), dtype=np.float32).reshape(cache_shape())
          + float(layer * 1000.0))


def make_caches(fill):
  sh = cpu_sharding()
  dt = _DTYPE_MAP.get(_DTYPE.value, jnp.float32)
  arrs = []
  for layer in range(_NUM_LAYERS.value):
    base = jnp.asarray(_layer_base(layer), dtype=dt) if fill else jnp.zeros(cache_shape(), dtype=dt)
    arrs.append(jax.device_put(base, sh))
  jax.block_until_ready(arrs)
  return arrs


def _write_json(path, obj):
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(obj, f)
  os.replace(tmp, path)  # atomic


def _read_json(path):
  try:
    with open(path) as f:
      return json.load(f)
  except (OSError, ValueError):
    return None


# ---------------- sender ----------------
def run_sender():
  bind_to_numa(_BIND_NUMA.value)
  rv = _RENDEZVOUS.value
  s_path, r_path = os.path.join(rv, 's.json'), os.path.join(rv, 'r.json')

  arrs = make_caches(fill=True)
  manager = kv_cache_manager.KVCacheManager(
      kv_caches=arrs, local_control_port=0,
      max_blocks=_NUM_BLOCKS.value, num_slots=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True)

  base_uuid = uuid.uuid4().int & 0xFFFFFFFF
  block_ids = list(range(_NUM_BLOCKS.value))
  endpoints = _resolve_endpoints(manager)
  print(f'[sender] endpoints={endpoints}, blocks={len(block_ids)}')

  _write_json(s_path, {'endpoints': endpoints, 'uuid': base_uuid,
                       'block_ids': block_ids, 'reg_round': -1})
  total_rounds = _WARMUP.value + _ITERS.value
  for i in range(total_rounds):
    req_id = f'h2h_{base_uuid}_{i}'
    manager.notify_for_read(req_id, base_uuid, block_ids)            # OSS API
    _write_json(s_path, {'endpoints': endpoints, 'uuid': base_uuid,
                         'block_ids': block_ids, 'reg_round': i})
    _wait_until(lambda i=i: (_read_json(r_path) or {}).get('recv_round', -1) >= i,
                timeout=300, what=f'receiver round {i}')
  print('[sender] all rounds consumed; exiting.')
  os._exit(0)  # kill hanging C++ transfer threads


# ---------------- receiver ----------------
def run_receiver():
  bind_to_numa(_BIND_NUMA.value)
  rv = _RENDEZVOUS.value
  s_path, r_path = os.path.join(rv, 's.json'), os.path.join(rv, 'r.json')

  _wait_until(lambda: _read_json(s_path) is not None, timeout=120, what='sender rendezvous')
  meta = _read_json(s_path)
  endpoints, base_uuid, block_ids = meta['endpoints'], meta['uuid'], meta['block_ids']
  endpoint = endpoints[0]
  print(f'[receiver] peer endpoint={endpoint}, blocks={len(block_ids)}')

  device_arrs = make_caches(fill=False)
  manager = kv_cache_manager.KVCacheManager(
      kv_caches=device_arrs, local_control_port=0,
      max_blocks=_NUM_BLOCKS.value, num_slots=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True)

  total_bytes = _NUM_LAYERS.value * len(block_ids) * block_byte_size()

  def one_read(i):
    req_id = f'h2h_{base_uuid}_{i}'
    _wait_until(lambda i=i: (_read_json(s_path) or {}).get('reg_round', -1) >= i,
                timeout=300, what=f'sender register round {i}')
    t0 = time.perf_counter()
    manager.start_read(req_id=req_id, uuid=base_uuid,               # OSS API
                       remote_endpoint=endpoint,
                       remote_block_ids=block_ids, local_block_ids=block_ids,
                       parallelism=_PARALLELISM.value)              # TCP streams
    while True:
      _done_send, done, failed = manager.complete_read()           # OSS API (poll)
      if req_id in done:
        break
      if req_id in failed:
        raise RuntimeError(f'H2H transfer failed: {req_id}')
      time.sleep(0.001)
    dt = time.perf_counter() - t0
    _write_json(r_path, {'recv_round': i})
    return dt

  for i in range(_WARMUP.value):
    one_read(i)
  times = [one_read(_WARMUP.value + i) for i in range(_ITERS.value)]

  gbps_all = [(total_bytes * 8) / (t * 1e9) for t in times]
  med_gbps = float(np.median(gbps_all))
  print(f'[receiver] cross-NUMA H2H median {med_gbps:.3f} Gbps over {_ITERS.value} iters')

  # data integrity: regenerate expected per-layer (same dtype) and compare exactly
  dt = _DTYPE_MAP.get(_DTYPE.value, jnp.float32)
  for layer, d in enumerate(device_arrs):
    exp = np.asarray(jnp.asarray(_layer_base(layer), dtype=dt))
    if not np.array_equal(np.asarray(d), exp):
      print(f'[receiver] DATA VERIFICATION FAILED on layer {layer}', file=sys.stderr)
      sys.exit(1)
  print('[receiver] data verified, 0% corruption.')

  _emit_metrics(times, gbps_all, total_bytes, med_gbps)


def _emit_metrics(times, gbps_all, total_bytes, med_gbps):
  tb = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if tb:
    try:
      try:
        import tensorboardX  # pylint: disable=g-import-not-at-top
        w = tensorboardX.SummaryWriter(log_dir=tb)
      except ImportError:
        import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
        w = tut.SummaryWriter(log_dir=tb)
      w.add_scalar('h2h_throughput_gbps', med_gbps, global_step=0)
      w.add_scalar('h2h_time_sec', float(np.median(times)), global_step=0)
      w.close()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'WARNING: TB write failed: {e}', file=sys.stderr)
  art = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  if art:
    _write_json(os.path.join(art, 'raw_h2h_results.json'), {
        'h2h_times_sec': times, 'h2h_gbps_all': gbps_all,
        'h2h_gbps_summary': summarize(gbps_all),
        'transferred_bytes_total': total_bytes})


# ---------------- launcher (single-host cross-NUMA driver) ----------------
def _kill(p):
  if p.poll() is None:
    p.terminate()
    try:
      p.wait(10)
    except subprocess.TimeoutExpired:
      p.kill()


def _supervise(sp, rp, timeout_s):
  """Never hang: react when either child finishes/dies; hard timeout backstop."""
  start = time.time()
  while True:
    rrc = rp.poll()
    if rrc is not None:                       # receiver done -> it carries the result
      _kill(sp)
      return rrc
    src = sp.poll()
    if src is not None and src != 0:          # sender died early -> fail fast
      print(f'[launch] sender exited {src} before receiver; aborting.', file=sys.stderr)
      _kill(rp)
      return src
    if time.time() - start > timeout_s:       # backstop: never wait for hours
      print(f'[launch] timeout {timeout_s}s; killing both.', file=sys.stderr)
      _kill(sp)
      _kill(rp)
      return 1
    time.sleep(0.5)


def run_launcher():
  nodes = numa_nodes()
  if len(nodes) < 2:
    print(f'WARNING: <2 NUMA nodes ({nodes}); transfer will be intra-socket (A-tier).')

  rv = tempfile.mkdtemp(prefix='h2h_rv_')
  pass_flags = [f'--rendezvous={rv}', f'--num_blocks={_NUM_BLOCKS.value}',
                f'--block_size={_BLOCK_SIZE.value}', f'--num_layers={_NUM_LAYERS.value}',
                f'--parallelism={_PARALLELISM.value}', f'--dtype={_DTYPE.value}',
                f'--warmup={_WARMUP.value}', f'--iters={_ITERS.value}']

  # REQUIRED: propagate bazel runfiles sys.path, else children can't import absl/jax.
  env = dict(os.environ)
  pp = os.pathsep.join(p for p in sys.path if p)
  env['PYTHONPATH'] = pp + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')

  def spawn(role, node):
    bind = node if len(nodes) >= 2 else -1
    cmd = [sys.executable, sys.argv[0], f'--role={role}', f'--bind_numa={bind}'] + pass_flags
    print('[launch] ' + ' '.join(cmd))
    return subprocess.Popen(cmd, env=env)

  sp = spawn('sender', _SENDER_NUMA.value)
  rp = spawn('receiver', _RECEIVER_NUMA.value)
  rc = _supervise(sp, rp, timeout_s=_LAUNCH_TIMEOUT_S.value)
  shutil.rmtree(rv, ignore_errors=True)
  if rc != 0:
    sys.exit(rc)
  print('[launch] single-host cross-NUMA H2H completed.')


def main(_):
  if _ROLE.value == 'launch':
    run_launcher()
  elif _ROLE.value == 'sender':
    run_sender()
  elif _ROLE.value == 'receiver':
    run_receiver()


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
