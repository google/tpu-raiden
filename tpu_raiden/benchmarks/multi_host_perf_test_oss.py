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

import ctypes
import gc
import json
import os
import sys
import time

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np

from tpu_raiden.frameworks.jax import _tpu_raiden_jax as kv_cache_manager

# --- flags (dtype/layers match the DMA benchmark's @parameterized groups) ---
_CACHE_SHAPE = flags.DEFINE_string(
    'cache_shape', '16,128,8,2,128',
    'Per-layer shape "num_blocks,d1,...". dim0 = num_blocks (major dim copied). '
    'DMA test_kv_cache uses 16,128,8,2,128; large-shape uses 8,128,1024,128.')
_SHARD_AXIS = flags.DEFINE_integer('shard_axis', 2, 'Array axis sharded across devices.')
_NUM_LAYERS = flags.DEFINE_integer('num_layers', 64, 'Number of cache arrays (layers).')
_DTYPE = flags.DEFINE_string('dtype', 'float32',
                             'float32 | bfloat16 | float16 | int32 | float8_e4m3fn.')
_ITERS = flags.DEFINE_integer('iters', 10, 'Timed iterations (DMA benchmark_runs default 10).')
_WARMUP = flags.DEFINE_integer('warmup', 3, 'Warmup iterations (not timed).')
_LOCK_BUFFERS = flags.DEFINE_bool('lock_buffers', True,
    'Lock host buffers (unsafe_skip_buffer_lock=False). DMA uses locked (True here).')
_NUMA_PIN = flags.DEFINE_bool('numa_pin', False, 'Pin process to one NUMA node (experiment).')
_NUMA_NODE = flags.DEFINE_integer('numa_node', 0, 'NUMA node to pin to when --numa_pin.')

_DTYPE_MAP = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16, 'float16': jnp.float16,
              'int32': jnp.int32, 'float8_e4m3fn': jnp.float8_e4m3fn}
_ITEMSIZE = {'float32': 4, 'bfloat16': 2, 'float16': 2, 'int32': 4, 'float8_e4m3fn': 1}


# ---------------- NUMA (optional experiment knob) ----------------
def _cpus_of_node(node):
  try:
    spec = open(f'/sys/devices/system/node/node{node}/cpulist').read().strip()
  except OSError:
    return set()
  cpus = set()
  for part in spec.split(','):
    if '-' in part:
      a, b = part.split('-'); cpus.update(range(int(a), int(b) + 1))
    elif part:
      cpus.add(int(part))
  return cpus


def bind_to_numa(node):
  want = _cpus_of_node(node)
  allowed = os.sched_getaffinity(0)
  cpus = want & allowed
  if cpus:
    try:
      os.sched_setaffinity(0, cpus)
    except OSError as e:
      print(f'WARNING: sched_setaffinity failed: {e}', file=sys.stderr)
  else:
    print(f'WARNING: no allowed cores on NUMA node {node}.', file=sys.stderr)
  try:
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    nodemask = ctypes.c_ulong(1 << node)
    if libc.syscall(238, 2, ctypes.byref(nodemask), ctypes.c_ulong(64)) == 0:  # set_mempolicy(MPOL_BIND)
      print(f'[numa] pinned to node {node} (mem; cpus={len(cpus)}).')
    else:
      print(f'WARNING: set_mempolicy errno={ctypes.get_errno()}', file=sys.stderr)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: NUMA membind unavailable: {e}', file=sys.stderr)


def summarize(values):
  a = np.array(values, dtype=float)
  return {'min': float(a.min()), 'p50': float(np.median(a)), 'mean': float(a.mean()),
          'p90': float(np.percentile(a, 90)), 'p99': float(np.percentile(a, 99)),
          'max': float(a.max()), 'stddev': float(a.std(ddof=1)) if len(a) > 1 else 0.0}


# ---------------- per-device placement (ported from the DMA benchmark) ----------------
def create_sharded_array(shape, sharding, dtype, is_host=False, is_random=False):
  """Places each shard on its own device (pinned_host if is_host) -> NUMA-local per chip."""
  mesh, spec = sharding.mesh, sharding.spec
  devices = list(mesh.devices.flat)
  shard_shape = list(shape)
  shard_axis = None
  for i, axis in enumerate(spec):
    if axis is not None:
      shard_axis = i; break
  if shard_axis is not None:
    shard_shape[shard_axis] = shape[shard_axis] // len(devices)

  shards = []
  for idx, device in enumerate(devices):
    sd = (jax.sharding.SingleDeviceSharding(device, memory_kind='pinned_host')
          if is_host else jax.sharding.SingleDeviceSharding(device))
    if is_random:
      shard_np = np.random.uniform(0, 1, shard_shape).astype(np.float32)
    elif dtype == jnp.int32:
      start = idx * int(np.prod(shard_shape))
      shard_np = (np.arange(np.prod(shard_shape), dtype=np.int32) + start).reshape(shard_shape)
    else:
      shard_np = np.zeros(shard_shape, dtype=np.float32)
    shards.append(jax.device_put(shard_np, sd).astype(dtype))
  return jax.make_array_from_single_device_arrays(shape, sharding, shards)


def write_tensorboard_metrics(d2h_time, h2d_time, d2h_gbps, h2d_gbps):
  tblog_dir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tblog_dir:
    print('TENSORBOARD_OUTPUT_DIR not set. Skipping TB.')
    return
  try:
    try:
      import tensorboardX  # pylint: disable=g-import-not-at-top
      w = tensorboardX.SummaryWriter(log_dir=tblog_dir)
    except ImportError:
      import torch.utils.tensorboard as tut  # pylint: disable=g-import-not-at-top
      w = tut.SummaryWriter(log_dir=tblog_dir)
    w.add_scalar('d2h_time_sec', d2h_time, global_step=0)
    w.add_scalar('h2d_time_sec', h2d_time, global_step=0)
    w.add_scalar('d2h_throughput_gbps', d2h_gbps, global_step=0)
    w.add_scalar('h2d_throughput_gbps', h2d_gbps, global_step=0)
    w.close()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: TB write failed: {e}', file=sys.stderr)


def main(_):
  if _NUMA_PIN.value:
    bind_to_numa(_NUMA_NODE.value)
  else:
    print('[numa] pinning disabled.')

  devices = jax.devices('tpu')
  if not devices:
    raise RuntimeError('No TPU devices found.')
  num_devices = len(devices)
  print(f'Found {num_devices} TPU devices.')

  shape = tuple(int(x) for x in _CACHE_SHAPE.value.split(','))
  num_blocks = shape[0]                      # major dim copied
  dt = _DTYPE_MAP.get(_DTYPE.value, jnp.float32)
  itemsize = _ITEMSIZE.get(_DTYPE.value, 4)
  print(f'shape={shape}, num_blocks={num_blocks}, layers={_NUM_LAYERS.value}, dtype={_DTYPE.value}')

  # mesh (1, num_devices) ("data","model"); shard `shard_axis` across the device axis
  mesh = jax.sharding.Mesh(np.array(devices).reshape(1, num_devices), ('data', 'model'))
  spec = jax.sharding.PartitionSpec(
      *[('model' if i == _SHARD_AXIS.value else None) for i in range(len(shape))])
  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)

  # per-device sharded TPU source arrays
  src_arrs = [create_sharded_array(shape, tpu_sharding, dt, is_host=False,
                                   is_random=(dt != jnp.int32))
              for _ in range(_NUM_LAYERS.value)]
  jax.block_until_ready(src_arrs)

  # invalidate device shadow copies before each measured d2h (like the DMA benchmark)
  mutate = jax.jit(lambda x: x + jnp.array(1 if x.dtype == jnp.int32 else 0.01, dtype=x.dtype))

  manager = kv_cache_manager.KVCacheManager(
      device_arrays=src_arrs,
      host_blocks_to_allocate=num_blocks,
      unsafe_skip_buffer_lock=not _LOCK_BUFFERS.value)

  # full-major-dim copy (matches DMA: offsets=[0], sizes=[num_blocks])
  offsets, sizes = [0], [num_blocks]
  total_bytes = _NUM_LAYERS.value * int(np.prod(shape)) * itemsize

  def once():
    nonlocal src_arrs
    src_arrs = [mutate(a) for a in src_arrs]
    jax.block_until_ready(src_arrs)
    gc.disable()
    t0 = time.perf_counter()
    manager.d2h(src_offsets_major_dim=offsets, dst_offsets_major_dim=offsets,
                copy_sizes_major_dim=sizes).Await()
    d2h = time.perf_counter() - t0
    gc.enable(); gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    manager.h2d(src_offsets_major_dim=offsets, dst_offsets_major_dim=offsets,
                copy_sizes_major_dim=sizes).Await()
    h2d = time.perf_counter() - t0
    gc.enable(); gc.collect()
    return d2h, h2d

  for _ in range(_WARMUP.value):
    once()
  d2h_times, h2d_times = [], []
  for _ in range(_ITERS.value):
    d, h = once()
    d2h_times.append(d); h2d_times.append(h)

  d2h_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in d2h_times]
  h2d_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in h2d_times]
  d2h_med_t, h2d_med_t = float(np.median(d2h_times)), float(np.median(h2d_times))
  d2h_gbps = (total_bytes * 8) / (d2h_med_t * 1e9)
  h2d_gbps = (total_bytes * 8) / (h2d_med_t * 1e9)
  print(f'D2H median {d2h_med_t*1000:.3f} ms -> {d2h_gbps:.3f} Gbps')
  print(f'H2D median {h2d_med_t*1000:.3f} ms -> {h2d_gbps:.3f} Gbps')

  write_tensorboard_metrics(d2h_med_t, h2d_med_t, d2h_gbps, h2d_gbps)

  art = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
  if art:
    _write = {
        'shape': list(shape), 'num_layers': _NUM_LAYERS.value, 'dtype': _DTYPE.value,
        'lock_buffers': bool(_LOCK_BUFFERS.value), 'numa_pin': bool(_NUMA_PIN.value),
        'transferred_bytes_total': total_bytes,
        'd2h_times_sec': d2h_times, 'h2d_times_sec': h2d_times,
        'd2h_gbps_all': d2h_gbps_all, 'h2d_gbps_all': h2d_gbps_all,
        'd2h_gbps_summary': summarize(d2h_gbps_all),
        'h2d_gbps_summary': summarize(h2d_gbps_all),
    }
    try:
      with open(os.path.join(art, 'raw_perf_results.json'), 'w') as f:
        json.dump(_write, f, indent=2)
      print('Saved raw_perf_results.json')
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'WARNING: artifact write failed: {e}', file=sys.stderr)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
