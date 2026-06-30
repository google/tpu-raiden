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

import os
import sys
import time
import json
from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from tpu_raiden.frameworks.jax import _tpu_raiden_jax as kv_cache_manager

_NUM_BLOCKS = flags.DEFINE_integer(
    'num_blocks', 512, 'Number of global cache blocks to allocate.'
)
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 2, 'Size of cache blocks.')
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 1, 'Number of transformer layers.'
)
_PARALLELISM = flags.DEFINE_integer('parallelism', 1, 'Concurrent transfer streams.')
_DTYPE = flags.DEFINE_string(
    'dtype',
    'float32',
    'Dataset type for the KV cache array: float32, bfloat16, float16.',
)
_WARMUP = flags.DEFINE_integer('warmup', 5, 'Number of warmup iterations.')
_ITERS = flags.DEFINE_integer('iters', 20, 'Number of benchmark iterations.')


def write_tensorboard_metrics(
    d2h_time_sec: float,
    h2d_time_sec: float,
    d2h_gbps: float,
    h2d_gbps: float,
):
  """Logs local copy CPU-TPU transfer metrics to Tensorboard event logs for BAP."""
  tblog_dir = os.environ.get('TENSORBOARD_OUTPUT_DIR')
  if not tblog_dir:
    print('TENSORBOARD_OUTPUT_DIR is not set. Skipping TensorBoard logging.')
    return

  print(f'Writing metrics to TensorBoard directory: {tblog_dir}')
  try:
    try:
      # pylint: disable=g-import-not-at-top
      import tensorboardX  # pytype: disable=import-error

      writer = tensorboardX.SummaryWriter(log_dir=tblog_dir)
    except ImportError:
      # pylint: disable=g-import-not-at-top
      import torch.utils.tensorboard  # pytype: disable=import-error

      writer = torch.utils.tensorboard.SummaryWriter(log_dir=tblog_dir)

    # Log averages
    writer.add_scalar('d2h_time_sec', d2h_time_sec, global_step=0)
    writer.add_scalar('h2d_time_sec', h2d_time_sec, global_step=0)
    writer.add_scalar('d2h_throughput_gbps', d2h_gbps, global_step=0)
    writer.add_scalar('h2d_throughput_gbps', h2d_gbps, global_step=0)
    writer.close()
    print('Successfully wrote performance metrics to TensorBoard logs.')
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f'WARNING: Failed to write TensorBoard logs: {e}', file=sys.stderr)


def setup_distributed_mesh(devices):
  """Sets up a global JAX Mesh sharding the block count across processes."""
  process_id = jax.process_index()
  num_processes = jax.process_count()
  num_local_devices = len(jax.local_devices())

  print(
      'Initializing JAX Distributed Mesh on Process'
      f' {process_id}/{num_processes}'
  )
  print(f'Local addressable devices seen: {jax.local_devices()}')
  print(f'Global cluster devices seen: {devices}')

  # Reshape mesh to (num_processes, num_local_devices)
  devices_array = np.array(devices).reshape((num_processes, num_local_devices))
  mesh = jax.sharding.Mesh(devices_array, ('host', 'device'))

  # Shard the first dimension (num_blocks) across the host axis!
  spec = jax.sharding.PartitionSpec('host', None, None, None, None)
  tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
  host_sharding = jax.sharding.NamedSharding(
      mesh, spec, memory_kind='pinned_host'
  )

  return tpu_sharding, host_sharding


def verify_device_cache(tpu_cache) -> bool:
  """Verifies local sharded tpu cache destination blocks match source blocks."""
  process_id = jax.process_index()
  print(
      f'[Process {process_id}] Verifying local sharded cache device'
      ' consistency...'
  )
  try:
    for s in tpu_cache.addressable_shards:
      # s.data is a process-local JAX array representing the blocks held by this
      # shard
      local_tpu_data = np.asarray(s.data)
      local_blocks = local_tpu_data.shape[
          0
      ]  # Total blocks held locally by this shard
      half = local_blocks // 2  # Half the blocks

      # Verify destination half matches source half
      np.testing.assert_array_equal(
          local_tpu_data[half:local_blocks],
          local_tpu_data[0:half],
      )
  except AssertionError as exc:
    print(f'[Process {process_id}] Verification FAILED!')
    print(exc)
    return False
  print(
      f'[Process {process_id}] Device consistency verified successfully! 0%'
      ' corruption.'
  )
  return True

def summarize(values):
  a = np.array(values, dtype=float)
  return {
      'min':    float(a.min()),
      'p50':    float(np.median(a)),
      'mean':   float(a.mean()),
      'p90':    float(np.percentile(a, 90)),
      'p99':    float(np.percentile(a, 99)),
      'max':    float(a.max()),   
      'stddev': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
  }

def main(_):
  process_id = jax.process_index()

  devices = jax.devices('tpu')
  tpu_sharding, _ = setup_distributed_mesh(devices)

  # Physical sharding shape: (num_blocks, head_count, 1, head_dim)
  cache_shape = (_NUM_BLOCKS.value, 32, 1, 8, 128)
  print(f'[Process {process_id}] Configured Cache Global Shape: {cache_shape}')

  # 1. Create a single large device array and initialize it with unique values
  print(
      f'[Process {process_id}] Initializing device cache with reference'
      ' sequence...'
  )
  dtype_map = {
      'float32': jnp.float32,
      'bfloat16': jnp.bfloat16,
      'float16': jnp.float16,
  }
  target_dtype = dtype_map.get(_DTYPE.value, jnp.float32)
  print(f'[Process {process_id}] Selected benchmark data type: {_DTYPE.value}')

  caches = []
  for _ in range(_NUM_LAYERS.value):
    base = jnp.arange(np.prod(cache_shape), dtype=target_dtype).reshape(
        cache_shape
    )
    caches.append(jax.device_put(base, tpu_sharding))
  jax.block_until_ready(caches)

  num_processes = jax.process_count()
  local_blocks = (
      _NUM_BLOCKS.value // num_processes
  )  # 512 // 4 = 128 blocks locally
  half_blocks = local_blocks // 2  # 128 // 2 = 64 blocks

  # 2. Create a kv cache manager and let it allocate internal host buffers for
  # local blocks / 2
  print(
      f'[Process {process_id}] Instantiating KVCacheManager with internal host'
      f' buffers for {half_blocks} blocks...'
  )
  manager = kv_cache_manager.KVCacheManager(
      device_arrays=caches, 
      host_blocks_to_allocate=half_blocks,
      unsafe_skip_buffer_lock=True,
      parallelism=_PARALLELISM.value,
  )

  # Calculate data sizes for throughput
  # cache_shape: (num_blocks, head_count, 1, head_dim) -> (num_blocks, 32, 1, 8, 128)
  # Elements per block: 32 * 1 * 8 * 128
  elements_per_block = np.prod(cache_shape[1:])
  dtype_itemsize = jnp.dtype(target_dtype).itemsize
  bytes_per_block = elements_per_block * dtype_itemsize
  num_layers = len(caches)
  num_shards = len(caches[0].addressable_shards)
  transferred_bytes_total = num_layers * num_shards * half_blocks * bytes_per_block

  # 3. Step A: Pull blocks 0:64 from device to internal host blocks 0:64 (D2H)
  print(
      f'[Process {process_id}] Executing D2H offloading (Local Blocks'
      f' 0..{half_blocks} -> Host)...'
  )
  src_offsets = list(range(0, half_blocks))
  dst_offsets = list(range(0, half_blocks))
  sizes = [1] * len(src_offsets)

  # Warmup
  for i in range(_WARMUP.value):
    manager.d2h(
        src_offsets_major_dim=src_offsets,
        dst_offsets_major_dim=dst_offsets,
        copy_sizes_major_dim=sizes,
    ).Await()

  # Benchmark Loop
  d2h_total_time = 0.0
  d2h_times = []
  for _ in range(_ITERS.value):
    start_time = time.perf_counter()
    manager.d2h(
        src_offsets_major_dim=src_offsets,
        dst_offsets_major_dim=dst_offsets,
        copy_sizes_major_dim=sizes,
    ).Await()
    elapsed = time.perf_counter() - start_time
    d2h_times.append(elapsed)
    d2h_total_time += elapsed

  d2h_time_mean = d2h_total_time / _ITERS.value
  d2h_gbps = (transferred_bytes_total * 8) / (d2h_time_mean * 1e9)
  print(
      f'[Process {process_id}] D2H complete. Avg Time:'
      f' {d2h_time_mean:.6f}s. Throughput: {d2h_gbps:.3f} Gbps'
  )
  print(f'[Process {process_id}] D2H Individual times: {d2h_times}')

  # 4. Step B: Push blocks from internal host blocks 0:64 to local TPU blocks
  # 64:128 (H2D)
  print(
      f'[Process {process_id}] Executing H2D reloading (Host -> Local TPU'
      f' Blocks {half_blocks}..{local_blocks})...'
  )
  src_offsets = list(range(0, half_blocks))
  dst_offsets = list(range(half_blocks, local_blocks))
  sizes = [1] * len(src_offsets)

  # Warmup
  for i in range(_WARMUP.value):
    manager.h2d(
        src_offsets_major_dim=src_offsets,
        dst_offsets_major_dim=dst_offsets,
        copy_sizes_major_dim=sizes,
    ).Await()

  # Benchmark Loop
  h2d_total_time = 0.0
  h2d_times = []
  for _ in range(_ITERS.value):
    start_time = time.perf_counter()
    manager.h2d(
        src_offsets_major_dim=src_offsets,
        dst_offsets_major_dim=dst_offsets,
        copy_sizes_major_dim=sizes,
    ).Await()
    elapsed = time.perf_counter() - start_time
    h2d_times.append(elapsed)
    h2d_total_time += elapsed

  h2d_time_mean = h2d_total_time / _ITERS.value
  h2d_gbps = (transferred_bytes_total * 8) / (h2d_time_mean * 1e9)
  print(
      f'[Process {process_id}] H2D complete. Avg Time:'
      f' {h2d_time_mean:.6f}s. Throughput: {h2d_gbps:.3f} Gbps'
  )
  print(f'[Process {process_id}] H2D Individual times: {h2d_times}')

  # 5. Step C: Verify on device that blocks 256:512 match blocks 0:256
  success = all(verify_device_cache(c) for c in caches)
  if not success:
    sys.exit(1)

  print(
      f'[Process {process_id}] E2E Device-to-Device multihost verification'
      ' completed successfully!'
  )

  if jax.process_index() == 0:
    write_tensorboard_metrics(d2h_time_mean, h2d_time_mean, d2h_gbps, h2d_gbps)

    # Save raw times to artifacts directory for detailed analysis
    artifact_dir = os.environ.get('WORKLOAD_ARTIFACTS_DIR')
    if artifact_dir:
      d2h_gbps_all = [(transferred_bytes_total * 8) / (t * 1e9) for t in d2h_times]
      h2d_gbps_all = [(transferred_bytes_total * 8) / (t * 1e9) for t in h2d_times]
      raw_results = {
          'd2h_times_sec': d2h_times,
          'h2d_times_sec': h2d_times,
          'd2h_gbps_all': d2h_gbps_all,           
          'h2d_gbps_all': h2d_gbps_all,
          'd2h_gbps_summary': summarize(d2h_gbps_all),   # ← min/p50/mean/p90/p99/max/stddev
          'h2d_gbps_summary': summarize(h2d_gbps_all),
          'd2h_time_mean': d2h_time_mean,
          'h2d_time_mean': h2d_time_mean,
          'transferred_bytes_total': transferred_bytes_total,
      }
      result_path = os.path.join(artifact_dir, 'raw_perf_results.json')
      try:
        with open(result_path, 'w') as f:
          json.dump(raw_results, f, indent=2)
        print(f'Saved raw performance results to {result_path}')
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'WARNING: Failed to write raw results artifact: {e}', file=sys.stderr)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
