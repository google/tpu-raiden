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

Runs BOTH roles on ONE machine, each pinned to a different NUMA node via
numactl, so the host-to-host transfer crosses the inter-socket interconnect
(UPI / Infinity Fabric). Measures cross-NUMA bandwidth (Gbps) and verifies data
integrity. Loopback traffic does NOT egress a physical NIC, so the number
reflects cross-socket memory bandwidth + kernel net-stack, not wire bandwidth.

Roles (--role):
  launch   : (default) spawn sender (NUMA A) + receiver (NUMA B) as subprocesses
             over loopback, wait, propagate the receiver's exit code.
  sender   : hold source KV cache, register reads round-by-round.
  receiver : pull the cache via start_read, time warmup+iters, verify, emit metrics.

API calls into _tpu_raiden_jax are marked `# VERIFY` — confirm against OSS binding.
"""

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
                          'Process role. launch = single-host driver.')
_RENDEZVOUS = flags.DEFINE_string('rendezvous', '',
                                  'Shared dir for s.json/r.json (set by launcher).')
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

_DTYPE_MAP = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16, 'float16': jnp.float16}
_ITEMSIZE = {'float32': 4, 'bfloat16': 2, 'float16': 2}


# ---------- helpers ----------
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


def cache_shape():
  return (_NUM_BLOCKS.value, 32, _BLOCK_SIZE.value, 8, 128)


def block_byte_size():
  return int(np.prod(cache_shape()[1:])) * _ITEMSIZE[_DTYPE.value]


def cpu_sharding():
  devices = jax.devices('cpu')
  mesh = jax.sharding.Mesh(np.array(devices).reshape(1, len(devices)), ('data', 'model'))
  spec = jax.sharding.PartitionSpec(None, None, 'model')
  return jax.sharding.NamedSharding(mesh, spec)


def make_caches(fill):
  """Deterministic per-layer arrays. Both roles regenerate identically, so the
  dtype truncation is identical on both sides -> exact equality regardless of dtype."""
  sh, dt = cpu_sharding(), _DTYPE_MAP.get(_DTYPE.value, jnp.float32)
  arrs = []
  for layer in range(_NUM_LAYERS.value):
    if fill:
      base = (np.arange(np.prod(cache_shape()), dtype=np.float32).reshape(cache_shape())
              + float(layer * 1000.0))
      base = jnp.asarray(base, dtype=dt)
    else:
      base = jnp.zeros(cache_shape(), dtype=dt)
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


# ---------- sender ----------
def run_sender():
  rv = _RENDEZVOUS.value
  s_path, r_path = os.path.join(rv, 's.json'), os.path.join(rv, 'r.json')
  arrs = make_caches(fill=True)
  manager = kv_cache_manager.KVCacheManager(          # VERIFY: transfer ctor signature
      kv_caches=arrs, local_control_port=0,
      max_blocks=_NUM_BLOCKS.value, num_slots=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True)
  base_uuid = uuid.uuid4().int & 0xFFFFFFFF
  block_ids = list(range(_NUM_BLOCKS.value))
  port = manager.local_control_port                    # VERIFY: property name
  host_ip = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6)[0][4][0]
  endpoint = f'[{host_ip}]:{port}'                     # VERIFY: endpoint format (mirror internal)

  _write_json(s_path, {'endpoint': endpoint, 'uuid': base_uuid,
                       'block_ids': block_ids, 'reg_round': -1})
  total_rounds = _WARMUP.value + _ITERS.value
  for i in range(total_rounds):
    req_id = f'h2h_{base_uuid}_{i}'
    manager.register_read(req_id, base_uuid, block_ids)             # VERIFY
    _write_json(s_path, {'endpoint': endpoint, 'uuid': base_uuid,
                         'block_ids': block_ids, 'reg_round': i})
    while (_read_json(r_path) or {}).get('recv_round', -1) < i:     # wait round done
      time.sleep(0.005)
  print('[sender] all rounds consumed by receiver; exiting.')
  os._exit(0)  # kill hanging C++ transfer threads


# ---------- receiver ----------
def run_receiver():
  rv = _RENDEZVOUS.value
  s_path, r_path = os.path.join(rv, 's.json'), os.path.join(rv, 'r.json')
  while _read_json(s_path) is None:
    time.sleep(0.05)
  meta = _read_json(s_path)
  endpoint, base_uuid, block_ids = meta['endpoint'], meta['uuid'], meta['block_ids']

  device_arrs = make_caches(fill=False)
  manager = kv_cache_manager.KVCacheManager(          # VERIFY
      kv_caches=device_arrs, local_control_port=0,
      max_blocks=_NUM_BLOCKS.value, num_slots=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True)

  total_bytes = _NUM_LAYERS.value * len(block_ids) * block_byte_size()

  def one_read(i):
    req_id = f'h2h_{base_uuid}_{i}'
    while (_read_json(s_path) or {}).get('reg_round', -1) < i:      # wait sender registered i
      time.sleep(0.002)
    t0 = time.perf_counter()
    manager.start_read(req_id=req_id, uuid=base_uuid,              # VERIFY
                       remote_endpoint=endpoint,
                       remote_block_ids=block_ids, local_block_ids=block_ids)
    while True:
      _, done, failed = manager.poll_stats()                       # VERIFY
      if req_id in done:
        break
      if req_id in failed:
        raise RuntimeError(f'H2H transfer failed: {req_id}')
      time.sleep(0.001)
    dt = time.perf_counter() - t0
    _write_json(r_path, {'recv_round': i})                         # signal round done
    return dt

  for i in range(_WARMUP.value):
    one_read(i)
  times = [one_read(_WARMUP.value + i) for i in range(_ITERS.value)]

  gbps_all = [(total_bytes * 8) / (t * 1e9) for t in times]
  med_gbps = float(np.median(gbps_all))
  print(f'[receiver] cross-NUMA H2H median {med_gbps:.3f} Gbps over {_ITERS.value} iters')

  # data integrity: regenerate expected (same dtype) and compare exactly
  expected = make_caches(fill=True)
  ok = all(np.array_equal(np.asarray(d), np.asarray(e))
           for d, e in zip(device_arrs, expected))
  if not ok:
    print('[receiver] DATA VERIFICATION FAILED', file=sys.stderr)
    sys.exit(1)
  print('[receiver] data verified, 0% corruption.')

  _emit_metrics(times, gbps_all, total_bytes, med_gbps)


def _emit_metrics(times, gbps_all, total_bytes, med_gbps):
  tb = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if tb:
    try:
      try:
        import tensorboardX
        w = tensorboardX.SummaryWriter(log_dir=tb)
      except ImportError:
        import torch.utils.tensorboard as tut
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


# ---------- launcher (single-host cross-NUMA driver) ----------
def run_launcher():
  nodes = numa_nodes()
  use_numa = len(nodes) >= 2 and shutil.which('numactl') is not None
  if not use_numa:
    print(f'WARNING: cross-NUMA unavailable (nodes={nodes}, '
          f'numactl={shutil.which("numactl")}). Falling back to loopback (A-tier).')

  rv = tempfile.mkdtemp(prefix='h2h_rv_')
  pass_flags = [f'--rendezvous={rv}', f'--num_blocks={_NUM_BLOCKS.value}',
                f'--block_size={_BLOCK_SIZE.value}', f'--num_layers={_NUM_LAYERS.value}',
                f'--parallelism={_PARALLELISM.value}', f'--dtype={_DTYPE.value}',
                f'--warmup={_WARMUP.value}', f'--iters={_ITERS.value}']

  def spawn(role, node):
    cmd = [sys.executable, sys.argv[0], f'--role={role}'] + pass_flags
    if use_numa:
      cmd = ['numactl', f'--cpunodebind={node}', f'--membind={node}'] + cmd
    print('[launch] ' + ' '.join(cmd))
    return subprocess.Popen(cmd)

  sp = spawn('sender', _SENDER_NUMA.value)
  rp = spawn('receiver', _RECEIVER_NUMA.value)
  rc = rp.wait()              # receiver carries the result + exit code
  sp.wait()
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
