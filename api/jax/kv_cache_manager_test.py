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
import threading
import time

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from api.jax import kv_cache_manager

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

flags.DEFINE_string(
    "device_type",
    "tpu",
    "The JAX device backend platform to run the tests on (e.g. 'tpu', 'cpu',"
    " 'cuda').",
)


class KVCacheManagerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    device_type = flags.FLAGS.device_type
    try:
      self.devices = jax.devices(device_type)
    except RuntimeError as exc:
      raise AssertionError(f"No {device_type} devices found") from exc

    if not self.devices:
      raise AssertionError(f"No {device_type} devices found")

    self.num_devices = len(self.devices)
    num_blocks = 2
    self.shape = (num_blocks, 128, 8, 8, 128)
    self.skip_lock = flags.FLAGS.device_type == "cpu"

  def create_mesh(self, axis_shapes, axis_names):
    try:
      devices = np.array(self.devices)
      num_required_devices = np.prod(axis_shapes)
      if len(devices) < num_required_devices:
        raise AssertionError(
            f"Need {num_required_devices} devices, got {len(devices)}"
        )
      device_array = devices[:num_required_devices].reshape(axis_shapes)
      return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
      self.skipTest("Cannot create mesh.")
      return None

  def setup_shardings(self):
    axis_shapes = (1, self.num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model")
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    return tpu_sharding

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_e2e_shift_copy(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(101)

    n_layers = 2
    # Shape: (num_blocks, ...)
    # We want to copy first half to host, then host to second half.
    # num_blocks = 4. half_blocks = 2.
    test_shape = (4, 128, 8, 8, 128)

    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_arrs)

    # Initialize manager with tpu_arrs and host_blocks_to_allocate = 2 (half of
    # 4 blocks)
    manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_arrs,
        block_size=1,
        host_blocks_to_allocate=2,
        unsafe_skip_buffer_lock=self.skip_lock,
    )

    # Step 1: D2H (TPU blocks 0:2 -> Host blocks 0:2)
    manager.d2h(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    ).Await()

    # Step 2: H2D (Host blocks 0:2 -> TPU blocks 2:4)
    manager.h2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[2],
    ).Await()

    # Verify: TPU blocks 2:4 should match ref_arrs blocks 0:2
    for i in range(n_layers):
      tpu_np = np.asarray(tpu_arrs[i])
      ref_np = np.asarray(ref_arrs[i])
      np.testing.assert_array_equal(tpu_np[2:4], ref_np[0:2])

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_e2e_shift_copy_skip_buffer_lock(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(101)

    n_layers = 2
    test_shape = (4, 128, 8, 8, 128)

    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_arrs)

    manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_arrs,
        block_size=1,
        host_blocks_to_allocate=2,
        unsafe_skip_buffer_lock=True,
    )

    manager.d2h(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    ).Await()

    manager.h2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[2],
    ).Await()

    for i in range(n_layers):
      tpu_np = np.asarray(tpu_arrs[i])
      ref_np = np.asarray(ref_arrs[i])
      np.testing.assert_array_equal(tpu_np[2:4], ref_np[0:2])

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_e2e_shift_copy_partial_chunks(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(303)

    n_layers = 2
    test_shape = (8, 128, 8, 8, 128)
    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_arrs)

    # Host blocks allocated = 5 (enough for our copies, which need 5 slices
    # total)
    manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_arrs,
        block_size=1,
        host_blocks_to_allocate=5,
        unsafe_skip_buffer_lock=self.skip_lock,
    )

    # Copy TPU chunks to Host:
    # TPU slices 1:3 -> Host slices 0:2 (size 2)
    # TPU slices 4:7 -> Host slices 2:5 (size 3)
    src_offsets_d2h = [1, 4]
    dst_offsets_d2h = [0, 2]
    sizes_d2h = [2, 3]
    manager.d2h(
        src_offsets_major_dim=src_offsets_d2h,
        dst_offsets_major_dim=dst_offsets_d2h,
        copy_sizes_major_dim=sizes_d2h,
    ).Await()

    # Copy Host chunks back to TPU at different locations:
    # Host slices 0:2 (containing TPU 1:3) -> TPU slices 2:4
    # Host slices 2:5 (containing TPU 4:7) -> TPU slices 5:8
    src_offsets_h2d = [0, 2]
    dst_offsets_h2d = [2, 5]
    sizes_h2d = [2, 3]
    manager.h2d(
        src_offsets_major_dim=src_offsets_h2d,
        dst_offsets_major_dim=dst_offsets_h2d,
        copy_sizes_major_dim=sizes_h2d,
    ).Await()

    for i in range(n_layers):
      tpu_np = np.asarray(tpu_arrs[i])
      ref_np = np.asarray(ref_arrs[i])
      np.testing.assert_array_equal(tpu_np[2:4], ref_np[1:3])
      np.testing.assert_array_equal(tpu_np[5:8], ref_np[4:7])

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_d2h_auto_allocate(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(505)

    n_layers = 2
    block_size = 2
    test_shape = (8, 128, 8, 8, 128)
    ref_arrs = []
    tpu_src_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      ref_arrs.append(base)

    for base in ref_arrs:
      tpu_src_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_src_arrs)

    # We allocate 2 blocks in host (capacity 4 slices).
    manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_src_arrs,
        block_size=block_size,
        host_blocks_to_allocate=2,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    # Transfer sub-region chunk of size 2 starting at offset 4 (TPU slices 4:6).
    # Needed blocks = 2 / 2 = 1 block.
    src_offsets = [4]
    sizes = [2]
    block_ids, future = manager.d2h_auto_allocate(
        src_offsets_major_dim=src_offsets,
        copy_sizes_major_dim=sizes,
        entity_id=101,
    )
    self.assertLen(block_ids, 1)
    self.assertEqual(block_ids[0], 0)

    future.Await()

    # Now copy from internal host block 0 (slices 0:2) back to TPU slices 0:2 to
    # verify.
    manager.h2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    ).Await()

    for i in range(n_layers):
      tpu_np = np.asarray(tpu_src_arrs[i])
      ref_np = np.asarray(ref_arrs[i])
      np.testing.assert_array_equal(tpu_np[0:2], ref_np[4:6])

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_h2h_write(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(606)

    n_layers = 2
    block_size = 2
    test_shape = (8, 128, 8, 8, 128)
    ref_arrs = []
    tpu_src_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_src_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_src_arrs)

    tpu_dst_arrs = [
        jax.device_put(jnp.empty(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(tpu_dst_arrs)

    # Spin up destination peer server on kernel ephemeral port 0.
    dst_manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_dst_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    time.sleep(0.05)
    port = dst_manager.local_port()
    self.assertIsNotNone(port)

    # Client source manager
    src_manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_src_arrs,
        block_size=block_size,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    # Populate src_manager's host buffer first
    src_manager.d2h().Await()

    # Push block ID 2 (slices 4:6) to remote peer server.
    peer = f"127.0.0.1:{port}"
    allocated_ids, future = src_manager.h2h_write(
        peer=peer, src_block_ids=[2], entity_id=202
    )
    self.assertLen(allocated_ids, 1)
    self.assertEqual(allocated_ids[0], 0)
    future.Await()

    # To verify, dst_manager received data into its host buffer.
    # We must h2d it to tpu_dst_arrs to verify.
    dst_manager.h2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    ).Await()

    for i in range(n_layers):
      tpu_dst_np = np.asarray(tpu_dst_arrs[i])
      ref_np = np.asarray(ref_arrs[i])
      np.testing.assert_array_equal(tpu_dst_np[0:2], ref_np[4:6])

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_h2h_read(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(707)

    n_layers = 2
    block_size = 2
    test_shape = (8, 128, 8, 8, 128)
    ref_arrs = []
    tpu_src_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_src_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_src_arrs)

    tpu_dst_arrs = [
        jax.device_put(jnp.empty(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(tpu_dst_arrs)

    # Remote server owns pre-populated source host buffers. Bound to dynamic
    # kernel port 0.
    remote_manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_src_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    # Populate remote host buffer
    remote_manager.d2h().Await()
    time.sleep(0.05)
    port = remote_manager.local_port()
    self.assertIsNotNone(port)

    local_manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_dst_arrs,
        block_size=block_size,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )

    peer = f"127.0.0.1:{port}"
    # Pull remote block ID 1 (slices 2:4) into local host memory.
    allocated_ids, future = local_manager.h2h_read(
        peer=peer, src_block_ids=[1], entity_id=303
    )
    self.assertLen(allocated_ids, 1)
    self.assertEqual(allocated_ids[0], 0)
    future.Await()

    # Copy from local host buffer to tpu_dst_arrs to verify
    local_manager.h2d(
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    ).Await()

    for i in range(n_layers):
      tpu_dst_np = np.asarray(tpu_dst_arrs[i])
      ref_np = np.asarray(ref_arrs[i])
      np.testing.assert_array_equal(tpu_dst_np[0:2], ref_np[2:4])

  def test_concurrent_transfer_skip_buffer_lock(self):
    tpu_sharding = self.setup_shardings()
    test_shape = (8, 128, 8, 8, 128)

    # Create host arrays.
    # Blocks 0:4 populated with 1.0
    # Blocks 4:8 populated with 2.0
    init_np = np.zeros(test_shape, dtype=np.float32)
    init_np[0:4] = 1.0
    init_np[4:8] = 2.0

    tpu_arrs = [
        jax.device_put(init_np, tpu_sharding) for _ in range(2)  # 2 layers
    ]
    jax.block_until_ready(tpu_arrs)

    manager = kv_cache_manager.KVCacheManager(
        device_arrays=tpu_arrs,
        block_size=1,
        host_blocks_to_allocate=8,
        unsafe_skip_buffer_lock=True,
    )
    # Populate internal host buffer with 1.0 and 2.0
    manager.d2h().Await()

    stop_event = threading.Event()
    thread1_errors = []

    def thread1_loop():
      try:
        # Thread 1 repeatedly copies blocks 0:4 H2D and D2H
        while not stop_event.is_set():
          manager.h2d(
              src_offsets_major_dim=[0],
              dst_offsets_major_dim=[0],
              copy_sizes_major_dim=[4],
          ).Await()
          manager.d2h(
              src_offsets_major_dim=[0],
              dst_offsets_major_dim=[0],
              copy_sizes_major_dim=[4],
          ).Await()
          time.sleep(0.01)
      except Exception as e:
        thread1_errors.append(e)

    def thread2_run():
      # Thread 2 copies blocks 4:8 H2D once, then signals Thread 1 to stop
      manager.h2d(
          src_offsets_major_dim=[4],
          dst_offsets_major_dim=[4],
          copy_sizes_major_dim=[4],
      ).Await()
      stop_event.set()

    t1 = threading.Thread(target=thread1_loop)
    t2 = threading.Thread(target=thread2_run)

    t1.start()
    t2.start()

    t2.join()
    t1.join()

    # Check if Thread 1 encountered any errors
    if thread1_errors:
      raise thread1_errors[0]

    # Final Verification:
    # TPU blocks 0:4 should have 1.0
    # TPU blocks 4:8 should have 2.0
    for i in range(2):
      tpu_np = np.asarray(tpu_arrs[i])
      np.testing.assert_array_equal(tpu_np[0:4], 1.0)
      np.testing.assert_array_equal(tpu_np[4:8], 2.0)


if __name__ == "__main__":
  absltest.main()
