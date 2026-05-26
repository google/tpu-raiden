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

"""Tests for mc_jax_ffi, focusing on multi-host sharded KV cache operations.

This test suite verifies the correctness of dynamic device-to-host (D2H) and
host-to-device (H2D) transfers within a sharded KV cache across multiple TPU
hosts using the Raiden FFI.
"""

import os

from absl import app
from absl import flags
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from api.jax import kv_cache_manager_ffi as raiden


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "global_blocks", 512, "Number of global cache blocks to allocate."
)
flags.DEFINE_integer("block_size", 2, "Size of cache blocks.")
flags.DEFINE_integer("layer_count", 4, "Number of transformer layers.")


def setup_distributed_mesh(devices):
  """Sets up a 2D logical sharding Mesh sharded on the device axis."""
  print(f"Discovered addressable PJRT devices: {devices}")
  mesh_shape = (len(devices) // len(devices), len(devices))
  devices_mesh = np.array(devices).reshape(mesh_shape)
  mesh = jax.sharding.Mesh(devices_mesh, ("host", "device"))

  # Shard on device dimension while replicating on host dimension
  tpu_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("device", None, None, None)
  )
  return mesh, tpu_sharding


def run_correctness_test(
    mesh,
    tpu_sharding,
    num_processes,
    num_local_devices,
    process_id,
    block_size=2,
    global_blocks=512,
    layer_count=4,
    head_count=32,
    head_dim=128,
) -> bool:
  """Executes sharded reloading transfers dynamically and verifies closed-loop correctness across device memory."""
  cache_shape = (global_blocks, layer_count, block_size, head_count, head_dim)

  # Format detailed execution metadata clearly for the user
  backend_type = "PJRT (Native TPU VM)"
  if jax.default_backend() == "cpu":
    backend_type = "CPU (Mock System)"

  print("\n=== [JAX Client Execution Metadata] ===")
  print(f"JAX Backend Type: {backend_type}")
  print(f"Total Global TPU Devices: {jax.device_count()}")
  print(f"Addressable Local Devices: {jax.local_device_count()}")
  print(f"Global Sharded Cache Shape: {cache_shape}")
  print("=======================================\n")

  local_blocks = global_blocks // num_processes

  # 1. Allocate single sharded JAX TPU device array (Source and Destination)
  print(f"[Process {process_id}] Initializing single sharded device array...")
  device_data_a = []

  for l in range(layer_count):
    # Cache A: Unique distinct values per layer/block to verify E2E
    base_a = jnp.arange(
        np.prod(cache_shape) // layer_count, dtype=jnp.float32
    ).reshape((global_blocks, block_size, head_count, head_dim)) + (
        l * 10000000
    )
    d_arr_a = jax.device_put(base_a, tpu_sharding)
    device_data_a.append(d_arr_a)

  # 2. Configure sharding metadata offsets representing half of local shard capacity
  local_shard_blocks = local_blocks // len(jax.devices())
  num_chunks = (local_shard_blocks // block_size) // 2

  base_block = process_id * local_blocks
  local_s_off = [x + base_block for x in range(num_chunks)]
  local_d_off = [x + base_block + num_chunks for x in range(num_chunks)]
  local_c_sz = [1] * num_chunks

  # Host-shard all metadata arrays globally using process_allgather
  gathered_s_off = multihost_utils.process_allgather(
      jnp.array(local_s_off, dtype=jnp.int32)
  )
  gathered_d_off = multihost_utils.process_allgather(
      jnp.array(local_d_off, dtype=jnp.int32)
  )
  gathered_c_sz = multihost_utils.process_allgather(
      jnp.array(local_c_sz, dtype=jnp.int32)
  )

  src_offsets = jax.device_put(
      gathered_s_off,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
  )
  dst_offsets = jax.device_put(
      gathered_d_off,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
  )
  copy_sizes = jax.device_put(
      gathered_c_sz,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
  )

  slice_byte_size = (
      head_count * head_dim * 4
  )  # Single block byte size (Float32)

  # Construct sharded device indices cleanly across the dynamic Mesh E2E!
  global_indices = jnp.arange(jax.device_count(), dtype=jnp.int32)
  local_indices = global_indices % num_local_devices
  shard_idx = jax.device_put(
      local_indices,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("device")),
  )

  # 3. Dynamic initialization and synchronization with FFI C++ context
  @jax.jit
  def run_server_init(anchor, s_idx):
    return raiden.init(
        device_array=anchor,
        shard_idx=s_idx,
        mesh=mesh,  # Dynamically distribute the thunk initialization across all local CPU/TPU devices E2E!
        slice_byte_size=slice_byte_size,
        block_size=block_size,
        host_blocks_to_allocate=global_blocks,
        num_layers=layer_count,
    )

  # Assign back to establish a strict data dependency for the compiler!
  device_data_a[0] = run_server_init(device_data_a[0], shard_idx)
  device_data_a[0].block_until_ready()

  # Explicitly block and synchronize sharded metadata arrays before triggering copies
  src_offsets.block_until_ready()
  dst_offsets.block_until_ready()
  copy_sizes.block_until_ready()

  # 4. Execution Stage 1: Trigger D2H transfers (Copy blocks from local_s_off HBM to Host CPU offsets)
  print(
      f"[Process {process_id}] 1. Triggering D2H FFI transfers into Host CPU"
      " buffers..."
  )

  token_d2h_list = []
  for l in range(layer_count):
    device_data_a[l] = raiden.d2h(
        device_data_a[l],
        l,  # Static layer index parameter
        src_offsets,  # Read from local_s_off in TPU HBM
        src_offsets,  # Write to local_s_off in CPU Host buffer
        copy_sizes,
        mesh,
    )
    token_d2h_list.append(device_data_a[l])

  for token in token_d2h_list:
    token.block_until_ready()
  raiden.sync_copies()
  print(f"[Process {process_id}] D2H transfers complete!")

  # 5. Execution Stage 2: Trigger H2D transfers (Copy Host CPU offsets directly back to local_d_off HBM!)
  print(
      f"[Process {process_id}] 2. Triggering H2D FFI transfers back into new"
      " HBM block destinations..."
  )

  token_h2d_list = []
  for l in range(layer_count):
    device_data_a[l] = raiden.h2d(
        device_data_a[l],  # Write back into the same JAX array!
        l,  # Static layer index parameter
        src_offsets,  # Read from local_s_off in CPU Host buffer
        dst_offsets,  # Write to local_d_off in TPU HBM
        copy_sizes,
        mesh,
    )
    token_h2d_list.append(device_data_a[l])

  for token in token_h2d_list:
    token.block_until_ready()
  raiden.sync_copies()
  print(f"[Process {process_id}] H2D transfers complete!")

  # 6. Verification Stage: Closed-loop single array matching!
  print(
      f"[Process {process_id}] 3. Verifying sharded closed-loop correctness..."
  )

  success = True
  for l in range(layer_count):
    # Retrieve local JAX shard view E2E
    # Convert global sharded JAX array to host-local array to safely support multi-host allgather
    local_device_data = multihost_utils.global_array_to_host_local_array(
        device_data_a[l],
        mesh,
        jax.sharding.PartitionSpec("device", None, None, None),
    )
    local_a = np.array(multihost_utils.process_allgather(local_device_data))

    for i in range(num_chunks):
      local_src_idx = i
      local_dst_idx = i + num_chunks

      val_a = local_a[process_id, local_src_idx]
      val_b = local_a[process_id, local_dst_idx]

      try:
        np.testing.assert_allclose(
            val_b,
            val_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg=(
                f"Layer {l} Block {local_s_off[i]}->{local_d_off[i]} mismatch!"
                f" Source val: {val_a[0, 0, 0]}, Destination val:"
                f" {val_b[0, 0, 0]}"
            ),
        )
      except AssertionError as e:
        print(f"[ERROR] {str(e)}")
        success = False

  raiden.destroy_kv_cache()
  return success


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  devices = jax.devices()
  mesh, tpu_sharding = setup_distributed_mesh(devices)

  success = run_correctness_test(
      mesh=mesh,
      tpu_sharding=tpu_sharding,
      num_processes=jax.process_count(),
      num_local_devices=jax.local_device_count(),
      process_id=jax.process_index(),
      block_size=FLAGS.block_size,
      global_blocks=FLAGS.global_blocks,
      layer_count=FLAGS.layer_count,
  )

  if success:
    print("E2E sharded copy correctness verification passed 100%!")
    # The os._exit(0) is intentional and necessary. It bypasses Python's normal
    # shutdown sequence to avoid PJRT/StreamExecutor background thread hangs.
    os._exit(0)  # pylint: disable=protected-access
  else:
    print("E2E correctness verification FAILED!")
    os._exit(1)  # pylint: disable=protected-access


if __name__ == "__main__":
  app.run(main)
