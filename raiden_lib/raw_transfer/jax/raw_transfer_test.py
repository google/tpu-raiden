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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from raiden_lib.raw_transfer.jax import raw_transfer

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

SUPPORTED_DTYPES = {
    jnp.float8_e4m3fn: "fp8",
    jnp.bfloat16: "bf16",
    jnp.float32: "fp32",
    jnp.int32: "int32",
}


class RawTransferTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      raise AssertionError("No TPU devices found")

    if not self.devices:
      raise AssertionError("No TPU devices found")

    self.num_devices = len(self.devices)
    # Emulate realistic KV cache block dimensions as requested by user
    num_blocks = 12
    self.shape = (num_blocks, 128, 8, 2, 128)

  def create_mesh(self, axis_shapes, axis_names, devices=None):
    try:
      num_required_devices = np.prod(axis_shapes)
      if devices is None:
        devices = self.devices
      devices = np.array(devices)
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
    # KV cache sharding alignment
    axis_shapes = (1, self.num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model")

    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    host_sharding = jax.sharding.NamedSharding(
        mesh, spec, memory_kind="pinned_host"
    )
    return tpu_sharding, host_sharding

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_single_transfers(self, dtype):
    tpu_sharding, host_sharding = self.setup_shardings()
    key = jax.random.key(123)

    if dtype == jnp.int32:
      ref_arr = jnp.arange(np.prod(self.shape), dtype=dtype).reshape(self.shape)
    else:
      ref_arr = jax.random.uniform(key, self.shape, dtype=dtype)

    tpu_arr = jax.device_put(ref_arr, tpu_sharding)
    host_alloc = jax.jit(
        lambda: jnp.zeros(self.shape, dtype=dtype),
        out_shardings=host_sharding,
    )

    # Validate D2H Sync
    dst_d2h = host_alloc()
    jax.block_until_ready([tpu_arr, dst_d2h])
    raw_transfer.transfer_d2h(tpu_arr, dst_d2h)
    np.testing.assert_array_equal(np.asarray(dst_d2h), np.asarray(ref_arr))

    # Validate H2D Sync
    dst_h2d = jax.device_put(jnp.empty(self.shape, dtype=dtype), tpu_sharding)
    jax.block_until_ready([dst_d2h, dst_h2d])
    raw_transfer.transfer_h2d(dst_d2h, dst_h2d)
    np.testing.assert_array_equal(np.asarray(dst_h2d), np.asarray(ref_arr))

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_single_async_transfers(self, dtype):
    tpu_sharding, host_sharding = self.setup_shardings()
    key = jax.random.key(456)

    if dtype == jnp.int32:
      ref_arr = jnp.arange(np.prod(self.shape), dtype=dtype).reshape(self.shape)
    else:
      ref_arr = jax.random.uniform(key, self.shape, dtype=dtype)

    tpu_arr = jax.device_put(ref_arr, tpu_sharding)
    host_alloc = jax.jit(
        lambda: jnp.zeros(self.shape, dtype=dtype),
        out_shardings=host_sharding,
    )

    # Validate D2H Async
    dst_d2h = host_alloc()
    jax.block_until_ready([tpu_arr, dst_d2h])
    raw_transfer.transfer_d2h_async(tpu_arr, dst_d2h).Await()
    np.testing.assert_array_equal(np.asarray(dst_d2h), np.asarray(ref_arr))

    # Validate H2D Async
    dst_h2d = jax.device_put(jnp.empty(self.shape, dtype=dtype), tpu_sharding)
    jax.block_until_ready([dst_d2h, dst_h2d])
    raw_transfer.transfer_h2d_async(dst_d2h, dst_h2d).Await()
    np.testing.assert_array_equal(np.asarray(dst_h2d), np.asarray(ref_arr))

  @parameterized.named_parameters(
      ("sync_fp8", "sync", jnp.float8_e4m3fn),
      ("sync_bf16", "sync", jnp.bfloat16),
      ("sync_fp32", "sync", jnp.float32),
      ("sync_int32", "sync", jnp.int32),
      ("async_opt_fp8", "async_opt", jnp.float8_e4m3fn),
      ("async_opt_bf16", "async_opt", jnp.bfloat16),
      ("async_opt_fp32", "async_opt", jnp.float32),
      ("async_opt_int32", "async_opt", jnp.int32),
      ("async_naive_fp8", "async_naive", jnp.float8_e4m3fn),
      ("async_naive_bf16", "async_naive", jnp.bfloat16),
      ("async_naive_fp32", "async_naive", jnp.float32),
      ("async_naive_int32", "async_naive", jnp.int32),
  )
  def test_batch_transfers(self, mode, dtype):
    tpu_sharding, host_sharding = self.setup_shardings()
    key = jax.random.key(789)

    n_layers = 2
    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(self.shape), dtype=dtype).reshape(self.shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, self.shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_arrs.append(jax.device_put(base, tpu_sharding))

    host_arrs = [
        jax.device_put(np.zeros(self.shape, dtype=dtype), host_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(host_arrs)

    tpu_dst_arrs = [
        jax.device_put(jnp.empty(self.shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(tpu_dst_arrs)

    if mode == "sync":
      raw_transfer.transfer_d2h_batch(tpu_arrs, host_arrs)
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(host_arrs[i]), np.asarray(ref_arrs[i])
        )

      raw_transfer.transfer_h2d_batch(host_arrs, tpu_dst_arrs)
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(tpu_dst_arrs[i]), np.asarray(ref_arrs[i])
        )

    elif mode == "async_opt":
      raw_transfer.transfer_d2h_batch_async(tpu_arrs, host_arrs).Await()
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(host_arrs[i]), np.asarray(ref_arrs[i])
        )

      raw_transfer.transfer_h2d_batch_async(host_arrs, tpu_dst_arrs).Await()
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(tpu_dst_arrs[i]), np.asarray(ref_arrs[i])
        )

    elif mode == "async_naive":
      raw_transfer.transfer_d2h_batch_async_naive(tpu_arrs, host_arrs).Await()
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(host_arrs[i]), np.asarray(ref_arrs[i])
        )

      raw_transfer.transfer_h2d_batch_async_naive(
          host_arrs, tpu_dst_arrs
      ).Await()
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(tpu_dst_arrs[i]), np.asarray(ref_arrs[i])
        )

  def test_single_async_transfers_is_ready(self):
    dtype = jnp.bfloat16
    tpu_sharding, host_sharding = self.setup_shardings()
    key = jax.random.key(456)

    ref_arr = jax.random.uniform(key, self.shape, dtype=dtype)
    tpu_arr = jax.device_put(ref_arr, tpu_sharding)
    host_alloc = jax.jit(
        lambda: jnp.zeros(self.shape, dtype=dtype),
        out_shardings=host_sharding,
    )

    # Validate D2H Async
    dst_d2h = host_alloc()
    jax.block_until_ready([tpu_arr, dst_d2h])
    future_d2h = raw_transfer.transfer_d2h_async(tpu_arr, dst_d2h)

    self.assertIsInstance(future_d2h.IsReady(), bool)
    self.assertTrue(raw_transfer.is_ready(future_d2h) in [True, False])
    self.assertTrue(raw_transfer.is_ready([future_d2h]) in [True, False])

    future_d2h.Await()

    self.assertTrue(future_d2h.IsReady())
    self.assertTrue(raw_transfer.is_ready(future_d2h))
    self.assertTrue(raw_transfer.is_ready([future_d2h]))
    np.testing.assert_array_equal(np.asarray(dst_d2h), np.asarray(ref_arr))

    # Validate H2D Async
    dst_h2d = jax.device_put(jnp.empty(self.shape, dtype=dtype), tpu_sharding)
    jax.block_until_ready([dst_d2h, dst_h2d])
    future_h2d = raw_transfer.transfer_h2d_async(dst_d2h, dst_h2d)

    self.assertIsInstance(future_h2d.IsReady(), bool)
    self.assertTrue(raw_transfer.is_ready(future_h2d) in [True, False])
    self.assertTrue(raw_transfer.is_ready([future_h2d]) in [True, False])

    future_h2d.Await()

    self.assertTrue(future_h2d.IsReady())
    self.assertTrue(raw_transfer.is_ready(future_h2d))
    self.assertTrue(raw_transfer.is_ready([future_h2d]))
    np.testing.assert_array_equal(np.asarray(dst_h2d), np.asarray(ref_arr))

  @parameterized.named_parameters(
      ("async_opt", "async_opt"),
      ("async_naive", "async_naive"),
  )
  def test_batch_transfers_is_ready(self, mode):
    dtype = jnp.bfloat16
    tpu_sharding, host_sharding = self.setup_shardings()
    key = jax.random.key(789)

    n_layers = 2
    ref_arrs = []
    tpu_arrs = []

    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      base = jax.random.uniform(sub_key, self.shape, dtype=dtype)
      ref_arrs.append(base)
      tpu_arrs.append(jax.device_put(base, tpu_sharding))

    host_arrs = [
        jax.device_put(np.zeros(self.shape, dtype=dtype), host_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(host_arrs)

    tpu_dst_arrs = [
        jax.device_put(jnp.empty(self.shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(tpu_dst_arrs)

    if mode == "async_opt":
      future_d2h = raw_transfer.transfer_d2h_batch_async(tpu_arrs, host_arrs)
      self.assertIsInstance(future_d2h.IsReady(), bool)
      self.assertTrue(raw_transfer.is_ready(future_d2h) in [True, False])
      self.assertTrue(raw_transfer.is_ready([future_d2h]) in [True, False])

      future_d2h.Await()
      self.assertTrue(future_d2h.IsReady())
      self.assertTrue(raw_transfer.is_ready(future_d2h))
      self.assertTrue(raw_transfer.is_ready([future_d2h]))
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(host_arrs[i]), np.asarray(ref_arrs[i])
        )

      future_h2d = raw_transfer.transfer_h2d_batch_async(host_arrs, tpu_dst_arrs)
      self.assertIsInstance(future_h2d.IsReady(), bool)
      self.assertTrue(raw_transfer.is_ready(future_h2d) in [True, False])
      self.assertTrue(raw_transfer.is_ready([future_h2d]) in [True, False])

      future_h2d.Await()
      self.assertTrue(future_h2d.IsReady())
      self.assertTrue(raw_transfer.is_ready(future_h2d))
      self.assertTrue(raw_transfer.is_ready([future_h2d]))
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(tpu_dst_arrs[i]), np.asarray(ref_arrs[i])
        )

    elif mode == "async_naive":
      future_d2h = raw_transfer.transfer_d2h_batch_async_naive(tpu_arrs, host_arrs)
      self.assertIsInstance(future_d2h.IsReady(), bool)
      self.assertTrue(raw_transfer.is_ready(future_d2h) in [True, False])
      self.assertTrue(raw_transfer.is_ready([future_d2h]) in [True, False])

      future_d2h.Await()
      self.assertTrue(future_d2h.IsReady())
      self.assertTrue(raw_transfer.is_ready(future_d2h))
      self.assertTrue(raw_transfer.is_ready([future_d2h]))
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(host_arrs[i]), np.asarray(ref_arrs[i])
        )

      future_h2d = raw_transfer.transfer_h2d_batch_async_naive(
          host_arrs, tpu_dst_arrs
      )
      self.assertIsInstance(future_h2d.IsReady(), bool)
      self.assertTrue(raw_transfer.is_ready(future_h2d) in [True, False])
      self.assertTrue(raw_transfer.is_ready([future_h2d]) in [True, False])

      future_h2d.Await()
      self.assertTrue(future_h2d.IsReady())
      self.assertTrue(raw_transfer.is_ready(future_h2d))
      self.assertTrue(raw_transfer.is_ready([future_h2d]))
      for i in range(n_layers):
        np.testing.assert_array_equal(
            np.asarray(tpu_dst_arrs[i]), np.asarray(ref_arrs[i])
        )

  def run_offsets_scenarios(
      self, dtype, is_async=False, is_batch=False, is_naive=False
  ):
    tpu_sharding, host_sharding = self.setup_shardings()
    key1 = jax.random.key(123)
    key2 = jax.random.key(456)

    if dtype == jnp.int32:
      ref_arr1 = jnp.arange(np.prod(self.shape), dtype=dtype).reshape(
          self.shape
      )
      ref_arr2 = ref_arr1 + 1000000
    else:
      ref_arr1 = jax.random.uniform(key1, self.shape, dtype=dtype)
      ref_arr2 = jax.random.uniform(key2, self.shape, dtype=dtype)

    tpu_arrs = [
        jax.device_put(ref_arr1, tpu_sharding),
        jax.device_put(ref_arr2, tpu_sharding),
    ]
    host_alloc = jax.jit(
        lambda: jnp.zeros(self.shape, dtype=dtype),
        out_shardings=host_sharding,
    )

    if is_batch:
      if is_async:
        if is_naive:
          transfer_d2h_fn = raw_transfer.transfer_d2h_batch_async_naive
          transfer_h2d_fn = raw_transfer.transfer_h2d_batch_async_naive
        else:
          transfer_d2h_fn = raw_transfer.transfer_d2h_batch_async
          transfer_h2d_fn = raw_transfer.transfer_h2d_batch_async
      else:
        transfer_d2h_fn = raw_transfer.transfer_d2h_batch
        transfer_h2d_fn = raw_transfer.transfer_h2d_batch
    else:
      if is_async:
        transfer_d2h_fn = raw_transfer.transfer_d2h_async
        transfer_h2d_fn = raw_transfer.transfer_h2d_async
      else:
        transfer_d2h_fn = raw_transfer.transfer_d2h
        transfer_h2d_fn = raw_transfer.transfer_h2d

    def call_d2h(src_list, dst_list, **kwargs):
      if is_batch:
        res = transfer_d2h_fn(src_list, dst_list, **kwargs)
      else:
        res = transfer_d2h_fn(src_list[0], dst_list[0], **kwargs)
      if is_async:
        res.Await()

    def call_h2d(src_list, dst_list, **kwargs):
      if is_batch:
        res = transfer_h2d_fn(src_list, dst_list, **kwargs)
      else:
        res = transfer_h2d_fn(src_list[0], dst_list[0], **kwargs)
      if is_async:
        res.Await()

    num_to_verify = 2 if is_batch else 1

    # Scenario 1: Copy a single block from middle to middle
    # src block 1 -> dst block 2, copy size 1
    dst_d2hs = [host_alloc(), host_alloc()]
    jax.block_until_ready(tpu_arrs + dst_d2hs)
    call_d2h(
        tpu_arrs,
        dst_d2hs,
        src_offsets_major_dim=[1],
        dst_offsets_major_dim=[2],
        copy_sizes_major_dim=[1],
    )
    for dst_d2h, ref_arr in zip(
        dst_d2hs[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      np.testing.assert_array_equal(
          np.asarray(dst_d2h)[2], np.asarray(ref_arr)[1]
      )
      for idx in range(12):
        if idx != 2:
          np.testing.assert_array_equal(
              np.asarray(dst_d2h)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    dst_h2ds = [
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
    ]
    jax.block_until_ready(dst_d2hs + dst_h2ds)
    call_h2d(
        dst_d2hs,
        dst_h2ds,
        src_offsets_major_dim=[2],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[1],
    )
    for dst_h2d, ref_arr in zip(
        dst_h2ds[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      np.testing.assert_array_equal(
          np.asarray(dst_h2d)[0], np.asarray(ref_arr)[1]
      )
      for idx in range(12):
        if idx != 0:
          np.testing.assert_array_equal(
              np.asarray(dst_h2d)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    # Scenario 2: Copy multiple non-contiguous blocks (size 1)
    # src blocks [0, 4] -> dst blocks [1, 5], copy sizes [1, 1]
    dst_d2hs = [host_alloc(), host_alloc()]
    jax.block_until_ready(tpu_arrs + dst_d2hs)
    call_d2h(
        tpu_arrs,
        dst_d2hs,
        src_offsets_major_dim=[0, 4],
        dst_offsets_major_dim=[1, 5],
        copy_sizes_major_dim=[1, 1],
    )
    for dst_d2h, ref_arr in zip(
        dst_d2hs[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      np.testing.assert_array_equal(
          np.asarray(dst_d2h)[1], np.asarray(ref_arr)[0]
      )
      np.testing.assert_array_equal(
          np.asarray(dst_d2h)[5], np.asarray(ref_arr)[4]
      )
      for idx in range(12):
        if idx != 1 and idx != 5:
          np.testing.assert_array_equal(
              np.asarray(dst_d2h)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    dst_h2ds = [
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
    ]
    jax.block_until_ready(dst_d2hs + dst_h2ds)
    call_h2d(
        dst_d2hs,
        dst_h2ds,
        src_offsets_major_dim=[1, 5],
        dst_offsets_major_dim=[0, 4],
        copy_sizes_major_dim=[1, 1],
    )
    for dst_h2d, ref_arr in zip(
        dst_h2ds[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      np.testing.assert_array_equal(
          np.asarray(dst_h2d)[0], np.asarray(ref_arr)[0]
      )
      np.testing.assert_array_equal(
          np.asarray(dst_h2d)[4], np.asarray(ref_arr)[4]
      )
      for idx in range(12):
        if idx != 0 and idx != 4:
          np.testing.assert_array_equal(
              np.asarray(dst_h2d)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    # Scenario 3: Copy multiple contiguous blocks
    # src block 1 -> dst block 0, copy size 2 (copies blocks 1 and 2)
    dst_d2hs = [host_alloc(), host_alloc()]
    jax.block_until_ready(tpu_arrs + dst_d2hs)
    call_d2h(
        tpu_arrs,
        dst_d2hs,
        src_offsets_major_dim=[1],
        dst_offsets_major_dim=[0],
        copy_sizes_major_dim=[2],
    )
    for dst_d2h, ref_arr in zip(
        dst_d2hs[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      np.testing.assert_array_equal(
          np.asarray(dst_d2h)[0], np.asarray(ref_arr)[1]
      )
      np.testing.assert_array_equal(
          np.asarray(dst_d2h)[1], np.asarray(ref_arr)[2]
      )
      for idx in range(12):
        if idx != 0 and idx != 1:
          np.testing.assert_array_equal(
              np.asarray(dst_d2h)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    dst_h2ds = [
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
    ]
    jax.block_until_ready(dst_d2hs + dst_h2ds)
    call_h2d(
        dst_d2hs,
        dst_h2ds,
        src_offsets_major_dim=[0],
        dst_offsets_major_dim=[1],
        copy_sizes_major_dim=[2],
    )
    for dst_h2d, ref_arr in zip(
        dst_h2ds[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      np.testing.assert_array_equal(
          np.asarray(dst_h2d)[1], np.asarray(ref_arr)[1]
      )
      np.testing.assert_array_equal(
          np.asarray(dst_h2d)[2], np.asarray(ref_arr)[2]
      )
      for idx in range(12):
        if idx != 1 and idx != 2:
          np.testing.assert_array_equal(
              np.asarray(dst_h2d)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    # Scenario 4: Copy multiple non-contiguous blocks with copy_size = 3
    # src blocks [0, 6] -> dst blocks [2, 8], copy sizes [3, 3]
    dst_d2hs = [host_alloc(), host_alloc()]
    jax.block_until_ready(tpu_arrs + dst_d2hs)
    call_d2h(
        tpu_arrs,
        dst_d2hs,
        src_offsets_major_dim=[0, 6],
        dst_offsets_major_dim=[2, 8],
        copy_sizes_major_dim=[3, 3],
    )
    for dst_d2h, ref_arr in zip(
        dst_d2hs[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      for i in range(3):
        np.testing.assert_array_equal(
            np.asarray(dst_d2h)[2 + i], np.asarray(ref_arr)[0 + i]
        )
        np.testing.assert_array_equal(
            np.asarray(dst_d2h)[8 + i], np.asarray(ref_arr)[6 + i]
        )
      tiled_dst_indices = {2, 3, 4, 8, 9, 10}
      for idx in range(12):
        if idx not in tiled_dst_indices:
          np.testing.assert_array_equal(
              np.asarray(dst_d2h)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

    dst_h2ds = [
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
        jax.device_put(jnp.zeros(self.shape, dtype=dtype), tpu_sharding),
    ]
    jax.block_until_ready(dst_d2hs + dst_h2ds)
    call_h2d(
        dst_d2hs,
        dst_h2ds,
        src_offsets_major_dim=[2, 8],
        dst_offsets_major_dim=[0, 6],
        copy_sizes_major_dim=[3, 3],
    )
    for dst_h2d, ref_arr in zip(
        dst_h2ds[:num_to_verify], [ref_arr1, ref_arr2][:num_to_verify]
    ):
      for i in range(3):
        np.testing.assert_array_equal(
            np.asarray(dst_h2d)[0 + i], np.asarray(ref_arr)[0 + i]
        )
        np.testing.assert_array_equal(
            np.asarray(dst_h2d)[6 + i], np.asarray(ref_arr)[6 + i]
        )
      tiled_device_indices = {0, 1, 2, 6, 7, 8}
      for idx in range(12):
        if idx not in tiled_device_indices:
          np.testing.assert_array_equal(
              np.asarray(dst_h2d)[idx], np.zeros_like(np.asarray(ref_arr)[idx])
          )

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_single_transfers_with_offsets(self, dtype):
    self.run_offsets_scenarios(dtype, is_async=False, is_batch=False)

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_single_async_transfers_with_offsets(self, dtype):
    self.run_offsets_scenarios(dtype, is_async=True, is_batch=False)

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_batch_transfers_with_offsets(self, dtype):
    self.run_offsets_scenarios(dtype, is_async=False, is_batch=True)

  @parameterized.named_parameters(
      ("fp8", jnp.float8_e4m3fn),
      ("bf16", jnp.bfloat16),
      ("fp32", jnp.float32),
      ("int32", jnp.int32),
  )
  def test_batch_async_transfers_with_offsets(self, dtype):
    self.run_offsets_scenarios(
        dtype, is_async=True, is_batch=True, is_naive=False
    )
    self.run_offsets_scenarios(
        dtype, is_async=True, is_batch=True, is_naive=True
    )


if __name__ == "__main__":
  absltest.main()
