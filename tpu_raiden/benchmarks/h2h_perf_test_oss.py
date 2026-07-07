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

"""Single-host cross-socket OSS H2H benchmark, aligned to h2h_benchmark_runner.cc.

PUSH semantics to match the C++ runner exactly:
 * sender (active/timed) calls manager.h2h_write(peer, ids).Await() -> same
   underlying KVCacheManagerBase::H2hWrite the C++ benchmark times.
 * receiver (passive) allocates targets, advertises its endpoint, waits.

Byte accounting mirrors C++: total_bytes = num_layers * num_blocks * block_size,
with block_size in BYTES (C++ kNumLayers=32, kNumShards=1 are flags here so you
can sweep every C++ config 1:1). Reports GB/s (bytes, like C++) AND Gbps (bits),
mean- AND median-based, plus p50/p90/p99 latency.

Single host, sender pinned to one socket, receiver to the other, over loopback:
measures cross-socket (UPI/IF) host-to-host bandwidth. loopback does NOT egress a
NIC, so this is memory+net-stack bound (B-tier), same caveat as the C++ run on
--data_interface=lo.

Roles (--role):
  launch   : (default) spawn sender (NUMA A) + receiver (NUMA B), supervise, exit.
  receiver : passive target; advertise endpoint, wait, (optional) verify.
  sender   : active writer; warmup+iters of h2h_write, time, emit metrics.
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

from absl import app
from absl import flags

# H2H is host-to-host (CPU only); force JAX to the CPU backend BEFORE importing it
# so it never tries to grab the TPU (/dev/vfio), which is busy/claimed on the
# shared runner and would fail init ("Device or resource busy") -- see b/ crash.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Single-host loopback: force the transport to bind 127.0.0.1, like the C++
# runner's --data_interface=lo. Otherwise InitTransportServer() auto-picks the
# real NIC (eth0) and the intra-host transfer over it HANGS (that data path is
# meant for cross-node over physical NICs). Excluding eth0 -- the only NIC on the
# runner -- empties the candidate list, so the manager falls back to 127.0.0.1.
os.environ.setdefault('EXCLUDE_CONTROL_INTERFACE', 'eth0')

# libtpu still loads on this CPU-only run and its background threads try to NUMA-
# pin themselves. bind_to_numa() has (correctly) constrained the process to ONE
# node's cores, so that pinning finds no cross-node cores and spams
# 'tpu_utils.cc: No allowed cores found for pinning on this NUMA node'. It is
# harmless for a host-only H2H test -- raise the C++/libtpu log threshold to ERROR
# so it stays quiet. (setdefault: export TF_CPP_MIN_LOG_LEVEL=0 to see it again.)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TPU_MIN_LOG_LEVEL', '2')
import jax  # noqa: E402
import numpy as np  # noqa: E402

from tpu_raiden.frameworks.jax import _tpu_raiden_jax as kv_cache_manager

_ROLE = flags.DEFINE_enum('role', 'launch', ['launch', 'sender', 'receiver'],
                          'Process role. "launch" = single-host driver.')
_RENDEZVOUS = flags.DEFINE_string('rendezvous', '',
                                  'Shared dir for s.json/r.json (set by launcher).')
_BIND_NUMA = flags.DEFINE_integer('bind_numa', -1,
                                  'Bind this process to NUMA node (cpu+mem). -1 = none.')
# --- knobs matched 1:1 to h2h_benchmark_runner.cc ---
_NUM_BLOCKS = flags.DEFINE_integer('num_blocks', 64, 'Blocks to transfer (C++ --num_blocks).')
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 1024 * 1024,
                                   'BYTES per block (C++ --block_size).')
_NUM_LAYERS = flags.DEFINE_integer('num_layers', 32, 'Layers (C++ kNumLayers=32).')
_NUM_SHARDS = flags.DEFINE_integer('num_shards', 1, 'Shards (C++ kNumShards=1).')
_PARALLELISM = flags.DEFINE_integer('parallelism', 1, 'TCP streams / slots (C++ --parallelism).')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Warmup transfers (C++ kNumWarmup=3).')
_ITERS = flags.DEFINE_integer('iters', 50, 'Timed transfers (C++ kNumIterations=50).')
_SENDER_NUMA = flags.DEFINE_integer('sender_numa', 0, 'NUMA node for sender.')
_RECEIVER_NUMA = flags.DEFINE_integer('receiver_numa', 1, 'NUMA node for receiver.')
_VERIFY = flags.DEFINE_bool('verify', False,
                            'Authoritative data-integrity check: sender does one '
                            'untimed verify write with a known src->dst map; receiver '
                            'byte-compares every written block. Corruption -> non-zero '
                            'exit. Perf numbers are unaffected (verify write is untimed).')
_LAUNCH_TIMEOUT_S = flags.DEFINE_integer('launch_timeout_s', 1200,
                                         'Hard cap on the whole single-host run.')


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
  """CPU affinity (intersection with allowed cores) + set_mempolicy(MPOL_BIND)."""
  if node is None or node < 0:
    return
  want = _cpus_of_node(node)
  allowed = os.sched_getaffinity(0)
  cpus = want & allowed
  if not cpus:
    print(f'WARNING: no allowed cores on NUMA node {node} (want={len(want)}, '
          f'allowed={len(allowed)}); container likely pinned to one node -> '
          'cross-socket degrades to intra-socket.', file=sys.stderr)
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


def report_numa_placement(tag):
  """Hard proof of where this process's pages live -> validates cross-NUMA.

  Parses /proc/self/numa_maps (per-node resident page counts, 'N<node>=<pages>')
  and prints the dominant node. Run after the KV caches are allocated+touched:
  sender pages should sit on its --bind_numa node, receiver on the other, so the
  H2H transfer genuinely crosses the socket boundary.
  """
  try:
    per_node = {}
    with open('/proc/self/numa_maps') as f:
      for line in f:
        for tok in line.split():
          if len(tok) > 2 and tok[0] == 'N':
            key, sep, val = tok.partition('=')
            if sep and key[1:].isdigit() and val.isdigit():
              per_node[int(key[1:])] = per_node.get(int(key[1:]), 0) + int(val)
  except OSError as e:
    print(f'[numa] {tag}: numa_maps unavailable ({e})', file=sys.stderr)
    return
  if not per_node:
    print(f'[numa] {tag}: no page counts in numa_maps')
    return
  total = sum(per_node.values())
  parts = ', '.join(f'node{n}={p}({100 * p // total}%)'
                    for n, p in sorted(per_node.items()))
  dom = max(per_node, key=per_node.get)
  print(f'[numa] {tag}: pages {parts}; dominant=node{dom}')


def _wait_until(cond, timeout, what, soft=False):
  t0 = time.time()
  while not cond():
    if time.time() - t0 > timeout:
      if soft:
        print(f'WARNING: soft timeout after {timeout}s waiting for {what}', file=sys.stderr)
        return False
      raise TimeoutError(f'timed out after {timeout}s waiting for {what}')
    time.sleep(0.005)
  return True


# ---------------- endpoint resolution (IPv4 + IPv6) ----------------
def _primary_endpoint(port):
  for family, probe in ((socket.AF_INET, ('8.8.8.8', 80)),
                        (socket.AF_INET6, ('2001:4860:4860::8888', 80))):
    s = socket.socket(family, socket.SOCK_DGRAM)
    try:
      s.connect(probe)
      ip = s.getsockname()[0]
      return f'[{ip}]:{port}' if family == socket.AF_INET6 else f'{ip}:{port}'
    except OSError:
      continue
    finally:
      s.close()
  return f'127.0.0.1:{port}'


def _resolve_endpoints(manager):
  # The push connects to the block-transport data plane, which listens on
  # manager.local_port() (kernel-assigned). This is EXACTLY what the C++
  # h2h_benchmark_runner advertises (*manager->local_port()), and it works.
  # get_local_endpoints() returns a DIFFERENT port (a control/aggregate endpoint);
  # advertising it made the sender connect to a socket the block-transport accept
  # loop never serves -> the push hung forever. So advertise local_port() directly.
  try:
    port = manager.local_port()
    if port:
      return [_primary_endpoint(port)]
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: local_port() unavailable ({e}); falling back.', file=sys.stderr)
  try:
    eps = manager.get_local_endpoints()
    out = [e if isinstance(e, str) else e.get('endpoint', str(e)) for e in eps]
    out = [e for e in out if e]
    if out:
      return out
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: get_local_endpoints unavailable ({e}); probing.', file=sys.stderr)
  return [_primary_endpoint(manager.local_control_port)]


# ---------------- cache helpers (flat raw bytes, like the C++ runner) ----------------
def cache_shape():
  # (blocks, bytes-per-block) int8 -> per-block bytes == block_size, exactly like C++.
  return (_NUM_BLOCKS.value, _BLOCK_SIZE.value)


def total_bytes():
  # == C++: active_managers(1) * kNumLayers * kNumShards * num_blocks * block_size
  return _NUM_LAYERS.value * _NUM_SHARDS.value * _NUM_BLOCKS.value * _BLOCK_SIZE.value


def _single_device_sharding():
  # num_shards=1 => one host buffer per layer, no cross-device split (matches kNumShards=1).
  dev = jax.devices('cpu')[0]
  return jax.sharding.SingleDeviceSharding(dev)


def _layer_fill(layer):
  n = _NUM_BLOCKS.value * _BLOCK_SIZE.value
  return ((np.arange(n, dtype=np.int64) + layer) % 256).astype(np.int8).reshape(cache_shape())


def make_caches(fill):
  sh = _single_device_sharding()
  arrs = []
  for layer in range(_NUM_LAYERS.value):
    base = _layer_fill(layer) if fill else np.zeros(cache_shape(), dtype=np.int8)
    arrs.append(jax.device_put(base, sh))
  jax.block_until_ready(arrs)
  return arrs


def _new_manager(arrs):
  # NOTE(VERIFY): same constructor the pull version used; h2h_write is a method on it.
  # parallelism is the PUSH stream count (Push() splits blocks into `parallelism`
  # concurrent H2hWriteWorkers). The binding defaults it to 4, and num_slots is a
  # SEPARATE staging-slot knob -- so wiring --parallelism only into num_slots left
  # the push stuck at 4 streams regardless of the flag, which made Python's numbers
  # not track the C++ runner (C++ passes the flag straight to the manager's
  # parallelism). Pass it explicitly so both split into the same number of streams.
  return kv_cache_manager.KVCacheManager(
      kv_caches=arrs, local_control_port=0,
      max_blocks=_NUM_BLOCKS.value, num_slots=_PARALLELISM.value,
      parallelism=_PARALLELISM.value,
      unsafe_skip_buffer_lock=True)


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


# ---------------- sender (active / timed writer, PUSH) ----------------
def run_sender():
  bind_to_numa(_BIND_NUMA.value)
  rv = _RENDEZVOUS.value
  s_path, r_path = os.path.join(rv, 's.json'), os.path.join(rv, 'r.json')

  # wait for the passive receiver to advertise its endpoint
  _wait_until(lambda: (_read_json(r_path) or {}).get('ready', False),
              timeout=120, what='receiver endpoint')
  peer = _read_json(r_path)['endpoints'][0]

  arrs = make_caches(fill=True)
  report_numa_placement(f'sender(bind_numa={_BIND_NUMA.value})')
  manager = _new_manager(arrs)
  block_ids = list(range(_NUM_BLOCKS.value))
  print(f'[sender] peer={peer}, blocks={len(block_ids)}, '
        f'layers={_NUM_LAYERS.value}, block_size={_BLOCK_SIZE.value}B, '
        f'parallelism={_PARALLELISM.value}')

  # Explicit dst = 0..N -> op=6 (write to blocks the receiver already backs with
  # make_caches), exactly like the C++ runner's iota(dst_block_ids). The 2-arg form
  # (op=1) makes the receiver dynamically AllocateBlocks, which FAILS on the JAX
  # manager (no allocatable blocks -> receiver closes the socket -> the sender dies
  # with "Socket closed during readv"). op=6 is the only working push here.
  dst_block_ids = block_ids

  def one_write():
    t0 = time.perf_counter()
    _alloc, fut = manager.h2h_write(peer=peer, src_block_ids=block_ids,
                                    dst_block_ids=dst_block_ids)
    fut.Await()
    return time.perf_counter() - t0

  for _ in range(_WARMUP.value):
    one_write()
  times = [one_write() for _ in range(_ITERS.value)]

  # authoritative integrity: one extra (untimed, so perf is unaffected) write with
  # a known src->dst block map, handed to the receiver so it can byte-compare
  # exactly what landed against the deterministic source fill.
  if _VERIFY.value:
    dst_ids, fut = manager.h2h_write(peer=peer, src_block_ids=block_ids,
                                     dst_block_ids=dst_block_ids)
    fut.Await()
    _write_json(os.path.join(rv, 'verify_map.json'),
                {'src_ids': [int(x) for x in block_ids],
                 'dst_ids': [int(x) for x in dst_ids]})

  _report_and_emit(times)

  # let the receiver finish (optional verify) before we tear everything down
  _write_json(s_path, {'done': True})
  _wait_until(lambda: (_read_json(r_path) or {}).get('done', False),
              timeout=60, what='receiver ack', soft=True)
  os._exit(0)  # kill any lingering transfer threads


def _read_host_block_layer(manager, layer, shard, num_blocks, block_size):
  """Read a layer's received data from the manager's OWN host buffer, the way the
  C++ runner verifies it. Uses read_host_bytes (the copy happens in C++ with an
  explicit length -> safe) instead of get_host_pointer + ctypes (bare address ->
  segfaults) or np.asarray(arrs) (JAX view that doesn't reflect the direct write)."""
  raw = manager.read_host_bytes(layer, shard, num_blocks * block_size)
  return np.frombuffer(raw, dtype=np.int8).reshape((num_blocks, block_size))


# ---------------- receiver (passive target) ----------------
def run_receiver():
  bind_to_numa(_BIND_NUMA.value)
  rv = _RENDEZVOUS.value
  s_path, r_path = os.path.join(rv, 's.json'), os.path.join(rv, 'r.json')

  arrs = make_caches(fill=False)
  report_numa_placement(f'receiver(bind_numa={_BIND_NUMA.value})')
  manager = _new_manager(arrs)
  endpoints = _resolve_endpoints(manager)
  print(f'[receiver] endpoints={endpoints}, blocks={_NUM_BLOCKS.value}')
  _write_json(r_path, {'endpoints': endpoints, 'ready': True})

  # stay alive to absorb the sender's writes; exit when sender signals done
  _wait_until(lambda: (_read_json(s_path) or {}).get('done', False),
              timeout=_LAUNCH_TIMEOUT_S.value, what='sender done')

  if _VERIFY.value:
    # authoritative: the sender handed us the exact src->dst block map from a
    # dedicated (untimed) verify write; byte-compare every written block, per
    # layer, against the deterministic source fill. Any mismatch => corruption.
    _wait_until(lambda: _read_json(os.path.join(rv, 'verify_map.json')) is not None,
                timeout=120, what='verify map')
    m = _read_json(os.path.join(rv, 'verify_map.json'))
    src_ids, dst_ids = m['src_ids'], m['dst_ids']
    mismatched = 0
    for layer in range(_NUM_LAYERS.value):
      # Read the manager's host buffer (where the transport actually wrote) via the
      # length-bounded read_host_bytes binding -- mirrors the C++ runner's
      # GetHostPointer verify. NOT np.asarray(arrs[layer]) (JAX view that misses the
      # direct write) and NOT get_host_pointer+ctypes (bare address -> segfault).
      got = _read_host_block_layer(manager, layer, 0,
                                   _NUM_BLOCKS.value, _BLOCK_SIZE.value)
      exp = _layer_fill(layer)
      if layer == 0 and src_ids:
        s0, d0 = src_ids[0], dst_ids[0]
        print(f'[verify-dbg] L0: got[d={d0}][:8]={got[d0][:8].tolist()} '
              f'exp[s={s0}][:8]={exp[s0][:8].tolist()} '
              f'got_nonzero={int(np.count_nonzero(got))}/{got.size}',
              file=sys.stderr)
      for s_id, d_id in zip(src_ids, dst_ids):
        if not np.array_equal(got[d_id], exp[s_id]):
          mismatched += 1
          break  # one mismatch marks the layer; move on
    ok = mismatched == 0
    verdict = {'ok': bool(ok), 'mismatched_layers': int(mismatched),
               'checked_blocks': len(src_ids), 'layers': _NUM_LAYERS.value}
    _write_json(os.path.join(rv, 'verify.json'), verdict)
    art = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if art:
      _write_json(os.path.join(art, 'verify.json'), verdict)
    print(f'[receiver] verify (authoritative): '
          f'{"OK" if ok else str(mismatched) + " layer(s) MISMATCH"} '
          f'over {len(src_ids)} blocks x {_NUM_LAYERS.value} layers')

  _write_json(r_path, {'endpoints': endpoints, 'ready': True, 'done': True})
  os._exit(0)


# ---------------- reporting (GB/s like C++, plus Gbps) ----------------
def _report_and_emit(times):
  tb = total_bytes()
  t = np.array(times, dtype=float)
  mean_t, med_t = float(t.mean()), float(np.median(t))
  p90_t, p99_t = float(np.percentile(t, 90)), float(np.percentile(t, 99))
  gbs_mean, gbs_med = tb / 1e9 / mean_t, tb / 1e9 / med_t  # GB/s (bytes)
  gbps_mean, gbps_med = gbs_mean * 8, gbs_med * 8  # Gbps (bits)
  gbs_all = [tb / 1e9 / x for x in times]
  gbps_all = [g * 8 for g in gbs_all]

  print('========== H2H PUSH (single-host cross-socket) ==========')
  print(f'Block Size: {_BLOCK_SIZE.value} bytes')
  print(f'Num Blocks: {_NUM_BLOCKS.value}')
  print(f'Layers: {_NUM_LAYERS.value} Shards: {_NUM_SHARDS.value} '
        f'Parallelism: {_PARALLELISM.value}')
  print(f'Total bytes: {tb} ({tb/1e9:.3f} GB) per transfer')
  print(f'Iterations: {_ITERS.value} (warmup {_WARMUP.value})')
  print(f'Latency (ms): mean {mean_t*1e3:.3f} p50 {med_t*1e3:.3f} '
        f'p90 {p90_t*1e3:.3f} p99 {p99_t*1e3:.3f}')
  print(f'Throughput (mean): {gbs_mean:8.3f} GB/s = {gbps_mean:8.3f} Gbps')
  print(f'Throughput (median): {gbs_med:8.3f} GB/s = {gbps_med:8.3f} Gbps')
  # Full distribution across ALL iters (so you can see the spread + tails, not
  # just p50). Percentiles + a compact ASCII histogram of the per-iter GB/s.
  g = np.array(gbs_all, dtype=float)
  pcts = [0, 1, 10, 25, 50, 75, 90, 99, 100]
  qs = np.percentile(g, pcts)
  print(f'Throughput distribution over {len(gbs_all)} iters (GB/s):')
  print('  ' + '  '.join(f'p{p}={q:.2f}' for p, q in zip(pcts, qs)))
  print(f'  mean={g.mean():.2f} std={g.std():.2f} '
        f'min={g.min():.2f} max={g.max():.2f} '
        f'cv={100 * g.std() / g.mean():.1f}%')
  counts, edges = np.histogram(g, bins=12)
  peak = max(int(counts.max()), 1)
  for i in range(len(counts)):
    bar = '#' * int(round(40 * counts[i] / peak))
    print(f'  [{edges[i]:7.2f},{edges[i + 1]:7.2f}) {int(counts[i]):5d} {bar}')
  print('=========================================================')

  tbdir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if tbdir:
    try:
      try:
        import tensorboardX  # pylint: disable=g-import-not-at-top
        w = tensorboardX.SummaryWriter(log_dir=tbdir)
      except ImportError:
        import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
        w = tut.SummaryWriter(log_dir=tbdir)
      w.add_scalar('h2h_throughput_gbs_mean', gbs_mean, global_step=0)
      w.add_scalar('h2h_throughput_gbps_median', gbps_med, global_step=0)
      w.add_scalar('h2h_time_sec', med_t, global_step=0)
      w.close()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'WARNING: TB write failed: {e}', file=sys.stderr)

  art = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  if art:
    _write_json(os.path.join(art, 'raw_h2h_results.json'), {
        'mode': 'push', 'block_size': _BLOCK_SIZE.value,
        'num_blocks': _NUM_BLOCKS.value, 'num_layers': _NUM_LAYERS.value,
        'num_shards': _NUM_SHARDS.value, 'parallelism': _PARALLELISM.value,
        'warmup': _WARMUP.value, 'iters': _ITERS.value,
        'transferred_bytes_total': tb,
        'h2h_times_sec': times,
        'h2h_gbs_all': gbs_all, 'h2h_gbps_all': gbps_all,
        'h2h_gbs_summary': summarize(gbs_all),
        'h2h_gbps_summary': summarize(gbps_all),
        'throughput_gbs_mean': gbs_mean, 'throughput_gbs_median': gbs_med,
    })


# ---------------- launcher (single-host cross-socket driver) ----------------
def _kill(p):
  if p.poll() is None:
    p.terminate()
    try:
      p.wait(10)
    except subprocess.TimeoutExpired:
      p.kill()


def _supervise(timed, passive, timeout_s):
  """Wait for the TIMED side (sender) -> it carries the result; fail fast if the
  passive side dies early; hard timeout backstop so we never hang."""
  start = time.time()
  while True:
    tc = timed.poll()
    if tc is not None:
      _kill(passive)
      return tc
    pc = passive.poll()
    if pc is not None and pc != 0:
      print(f'[launch] passive (receiver) exited {pc} early; aborting.', file=sys.stderr)
      _kill(timed)
      return pc
    if time.time() - start > timeout_s:
      print(f'[launch] timeout {timeout_s}s; killing both.', file=sys.stderr)
      _kill(timed)
      _kill(passive)
      return 1
    time.sleep(0.5)


def run_launcher():
  nodes = numa_nodes()
  if len(nodes) < 2:
    print(f'WARNING: <2 NUMA nodes ({nodes}); transfer will be intra-socket.')

  rv = tempfile.mkdtemp(prefix='h2h_rv_')
  pass_flags = [f'--rendezvous={rv}', f'--num_blocks={_NUM_BLOCKS.value}',
                f'--block_size={_BLOCK_SIZE.value}', f'--num_layers={_NUM_LAYERS.value}',
                f'--num_shards={_NUM_SHARDS.value}', f'--parallelism={_PARALLELISM.value}',
                f'--warmup={_WARMUP.value}', f'--iters={_ITERS.value}',
                f'--verify={_VERIFY.value}']

  # REQUIRED: propagate bazel runfiles sys.path so children can import absl/jax.
  env = dict(os.environ)
  pp = os.pathsep.join(p for p in sys.path if p)
  env['PYTHONPATH'] = pp + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')

  def spawn(role, node):
    bind = node if len(nodes) >= 2 else -1
    cmd = [sys.executable, sys.argv[0], f'--role={role}', f'--bind_numa={bind}'] + pass_flags
    print('[launch] ' + ' '.join(cmd))
    return subprocess.Popen(cmd, env=env)

  # receiver first (it advertises the endpoint the sender writes to)
  rp = spawn('receiver', _RECEIVER_NUMA.value)
  sp = spawn('sender', _SENDER_NUMA.value)
  rc = _supervise(timed=sp, passive=rp, timeout_s=_LAUNCH_TIMEOUT_S.value)

  # authoritative integrity enforcement: read the receiver's verdict and fail the
  # whole run (non-zero exit) on corruption, so a gate driving run_launcher treats
  # it as a hard failure.
  verify_bad = False
  if _VERIFY.value and rc == 0:
    v = _read_json(os.path.join(rv, 'verify.json'))
    if v is None:
      print('[launch] verify requested but receiver produced no verdict; failing.',
            file=sys.stderr)
      verify_bad = True
    elif not v.get('ok', False):
      print(f'[launch] DATA INTEGRITY FAIL: {v}', file=sys.stderr)
      verify_bad = True
    else:
      print(f'[launch] data integrity OK: {v}')

  shutil.rmtree(rv, ignore_errors=True)
  if rc != 0:
    sys.exit(rc)
  if verify_bad:
    sys.exit(8)  # distinct code: transfer succeeded but data was corrupt
  print('[launch] single-host cross-socket H2H (push) completed.')


def main(_):
  if _ROLE.value == 'launch':
    run_launcher()
  elif _ROLE.value == 'sender':
    run_sender()
  elif _ROLE.value == 'receiver':
    run_receiver()


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
