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

"""Shared measurement core for the h2d/d2h benchmark and its perf gate.

Deliberately flag-free: both the benchmark binary (multi_host_perf_test_oss)
and the gating binary (h2d_d2h_benchmark_gating) import measure() from here, so
neither leaks absl flags into the other and the transfer code is reused, not
duplicated.
"""

import ctypes
import gc
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_raiden.frameworks.jax import _tpu_raiden_jax as kv_cache_manager

DTYPE_MAP = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16, 'float16': jnp.float16,
             'int32': jnp.int32, 'float8_e4m3fn': jnp.float8_e4m3fn}
ITEMSIZE = {'float32': 4, 'bfloat16': 2, 'float16': 2, 'int32': 4, 'float8_e4m3fn': 1}


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


def verify_roundtrip(manager, dev_arrs, num_blocks):
  """One-shot d2h->h2d data-integrity check (ported from V2 verify_device_cache).

  Uses the SAME manager + d2h/h2d as the benchmark, but with disjoint offsets
  and OUTSIDE the timed loop, so it changes neither the measured Gbps nor the
  recorded baselines. It pulls the source-half blocks [0:half] down to host,
  pushes them back up to the *distinct* destination half [half:2*half] on device,
  then asserts the destination now equals the source. The destination half
  starts with different values (see create_sharded_array init), so a no-op,
  misaligned, or corrupting transfer leaves the two halves unequal and fails.
  Raises AssertionError on mismatch -> the gate exits non-zero.
  """
  half = num_blocks // 2
  if half == 0:
    return  # major dim too small to split into src/dst halves; skip check
  # ONE bulk descriptor (copy_sizes=[half]) -- matches once()'s
  # offsets=[0], sizes=[num_blocks], so verify exercises the SAME bulk-copy path
  # that the timed/recorded copy uses, not a per-block scatter/gather.
  # d2h: device blocks [0:half] -> host [0:half]; h2d: host [0:half] -> device [half:2*half].
  manager.d2h(src_offsets_major_dim=[0], dst_offsets_major_dim=[0],
              copy_sizes_major_dim=[half]).Await()
  manager.h2d(src_offsets_major_dim=[0], dst_offsets_major_dim=[half],
              copy_sizes_major_dim=[half]).Await()
  # Read the raw shard buffers (like V2), not an XLA slice, so we observe the
  # bytes the manager actually wrote and never a cached/stale view.
  for li, a in enumerate(dev_arrs):
    for s in a.addressable_shards:
      d = np.asarray(s.data)
      np.testing.assert_array_equal(
          d[half:2 * half], d[0:half],
          err_msg=f'CORRUPTION: layer {li} d2h/h2d round-trip mismatch')

def measure(shape, num_layers, dtype, shard_axis=2, iters=20, warmup=3,
            lock_buffers=True, verify=True):  # verify: run one integrity check before timing
  """Runs the d2h/h2d transfer benchmark for one config and returns a result dict.

  `shape` may be a "a,b,c" string (flag) or a list/tuple (json). Pure compute +
  timing, no flag reads and no file I/O -> reusable from any binary.
  """
  if isinstance(shape, str):
    shape = shape.split(',')
  shape = tuple(int(x) for x in shape)

  devices = jax.devices('tpu')
  if not devices:
    raise RuntimeError('No TPU devices found.')
  num_devices = len(devices)
  num_blocks = shape[0]                      # major dim copied
  dt = DTYPE_MAP.get(dtype, jnp.float32)
  itemsize = ITEMSIZE.get(dtype, 4)

  # mesh (1, num_devices) ("data","model"); shard `shard_axis` across the device axis
  mesh = jax.sharding.Mesh(np.array(devices).reshape(1, num_devices), ('data', 'model'))
  spec = jax.sharding.PartitionSpec(
      *[('model' if i == shard_axis else None) for i in range(len(shape))])
  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)

  src_arrs = [create_sharded_array(shape, tpu_sharding, dt, is_host=False,
                                   is_random=(dt != jnp.int32))
              for _ in range(num_layers)]
  jax.block_until_ready(src_arrs)

  # invalidate device shadow copies before each measured d2h (like the DMA benchmark)
  mutate = jax.jit(lambda x: x + jnp.array(1 if x.dtype == jnp.int32 else 0.01, dtype=x.dtype))

  manager = kv_cache_manager.KVCacheManager(
      device_arrays=src_arrs,
      host_blocks_to_allocate=num_blocks,
      unsafe_skip_buffer_lock=not lock_buffers)

  # Verify the transfer is byte-correct BEFORE timing (and before any mutate), so
  # src_arrs still holds the known init values and the timed loop below is
  # untouched. A corrupt/no-op transfer raises here and fails the gate.
  if verify:
    verify_roundtrip(manager, src_arrs, num_blocks)

  offsets, sizes = [0], [num_blocks]        # full-major-dim copy
  total_bytes = num_layers * int(np.prod(shape)) * itemsize

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

  for _ in range(warmup):
    once()
  d2h_times, h2d_times = [], []
  for _ in range(iters):
    d, h = once()
    d2h_times.append(d); h2d_times.append(h)

  d2h_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in d2h_times]
  h2d_gbps_all = [(total_bytes * 8) / (t * 1e9) for t in h2d_times]
  d2h_med_t, h2d_med_t = float(np.median(d2h_times)), float(np.median(h2d_times))
  return {
      'shape': list(shape), 'num_layers': num_layers, 'dtype': dtype,
      'total_bytes': total_bytes,
      'd2h_times_sec': d2h_times, 'h2d_times_sec': h2d_times,
      'd2h_gbps_all': d2h_gbps_all, 'h2d_gbps_all': h2d_gbps_all,
      'd2h_med_t': d2h_med_t, 'h2d_med_t': h2d_med_t,
      'd2h_gbps': (total_bytes * 8) / (d2h_med_t * 1e9),
      'h2d_gbps': (total_bytes * 8) / (h2d_med_t * 1e9),
      'd2h_gbps_summary': summarize(d2h_gbps_all),
      'h2d_gbps_summary': summarize(h2d_gbps_all),
  }
