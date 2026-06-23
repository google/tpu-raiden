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

"""Dynamic performance test runner tool for raw memory offloading collectives (D2H and H2D)."""

import os
import sys
import time
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
_DTYPE = flags.DEFINE_string(
    'dtype',
    'float32',
    'Dataset type for the KV cache array: float32, bfloat16, float16.',
)


def write_tensorboard_metrics(d2h_time_sec: float, h2d_time_sec: float):
  """Logs local copy CPU-TPU transfer times to Tensorboard event logs for BAP."""
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

    writer.add_scalar('d2h_time_sec', d2h_time_sec, global_step=0)
    writer.add_scalar('h2d_time_sec', h2d_time_sec, global_step=0)
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

  base = jnp.arange(np.prod(cache_shape), dtype=target_dtype).reshape(
      cache_shape
  )
  tpu_cache = jax.device_put(base, tpu_sharding)
  jax.block_until_ready(tpu_cache)

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
      device_arrays=[tpu_cache],
      host_blocks_to_allocate=half_blocks,
      unsafe_skip_buffer_lock=True,
  )

  # 3. Step A: Pull blocks 0:64 from device to internal host blocks 0:64 (D2H)
  print(
      f'[Process {process_id}] Executing D2H offloading (Local Blocks'
      f' 0..{half_blocks} -> Host)...'
  )
  src_offsets = list(range(0, half_blocks))
  dst_offsets = list(range(0, half_blocks))
  sizes = [1] * len(src_offsets)

  start_time = time.perf_counter()
  manager.d2h(
      src_offsets_major_dim=src_offsets,
      dst_offsets_major_dim=dst_offsets,
      copy_sizes_major_dim=sizes,
  ).Await()
  d2h_time = time.perf_counter() - start_time
  print(
      f'[Process {process_id}] D2H complete in'
      f' {d2h_time:.4f}s.'
  )

  # 4. Step B: Push blocks from internal host blocks 0:64 to local TPU blocks
  # 64:128 (H2D)
  print(
      f'[Process {process_id}] Executing H2D reloading (Host -> Local TPU'
      f' Blocks {half_blocks}..{local_blocks})...'
  )
  src_offsets = list(range(0, half_blocks))
  dst_offsets = list(range(half_blocks, local_blocks))
  sizes = [1] * len(src_offsets)

  start_time = time.perf_counter()
  manager.h2d(
      src_offsets_major_dim=src_offsets,
      dst_offsets_major_dim=dst_offsets,
      copy_sizes_major_dim=sizes,
  ).Await()
  h2d_time = time.perf_counter() - start_time
  print(
      f'[Process {process_id}] H2D complete in'
      f' {h2d_time:.4f}s.'
  )

  # 5. Step C: Verify on device that blocks 256:512 match blocks 0:256
  success = verify_device_cache(tpu_cache)
  if not success:
    sys.exit(1)

  print(
      f'[Process {process_id}] E2E Device-to-Device multihost verification'
      ' completed successfully!'
  )

  if jax.process_index() == 0:
    write_tensorboard_metrics(d2h_time, h2d_time)


if __name__ == '__main__':
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
