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

from api.jax import disagg_kv_cache_manager

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

flags.DEFINE_string(
    "device_type",
    "tpu",
    "The JAX device backend platform to run the tests on (e.g. 'tpu', 'cpu',"
    " 'cuda').",
)


class DisaggKVCacheManagerTest(parameterized.TestCase):

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
  def test_e2e_disagg_push(self, dtype):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(101)

    n_layers = 2
    block_size = 2
    test_shape = (8, 128, 8, 8, 128)

    # 1. Setup Prefill (Sender) data
    prefill_ref_arrs = []
    prefill_tpu_arrs = []
    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      if dtype == jnp.int32:
        base = (
            jnp.arange(np.prod(test_shape), dtype=dtype).reshape(test_shape) + i
        )
      else:
        base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      prefill_ref_arrs.append(base)
      prefill_tpu_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(prefill_tpu_arrs)

    # 2. Setup Decode (Receiver) empty buffers
    decode_tpu_arrs = [
        jax.device_put(jnp.zeros(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(decode_tpu_arrs)

    # 3. Initialize managers
    prefill_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )

    # Start background threads
    prefill_manager.start()
    decode_manager.start()

    # Wait for ports to be bound
    time.sleep(0.1)
    decode_trans_port = decode_manager.local_port()
    decode_zmq_port = decode_manager.zmq_control_port()

    self.assertIsNotNone(decode_trans_port)
    self.assertNotEqual(decode_zmq_port, 0)

    # Bootstrap connection: Prefill registers Decode peer
    prefill_manager.register_peer(
        "decode", "127.0.0.1", decode_zmq_port, decode_trans_port
    )

    # Events/Callbacks for synchronization
    prefill_done = threading.Event()
    decode_done = threading.Event()
    errors = []

    def prefill_cb(status):
      if status is not None:
        errors.append(f"Prefill error: {status}")
      prefill_done.set()

    def decode_cb(status):
      if status is not None:
        errors.append(f"Decode error: {status}")
      decode_done.set()

    # 4. Submit Decode Receive Request (H2D step)
    decode_manager.submit_request(
        request_id=1001,
        req_type=disagg_kv_cache_manager.DisaggTransferRequestType.DECODE_H2D,
        dst_offsets=[0, 2],
        sizes=[2, 2],
        callback=decode_cb,
    )

    # 5. Submit Prefill Send Request (D2H + H2H Write steps)
    prefill_manager.submit_request(
        request_id=1001,
        req_type=disagg_kv_cache_manager.DisaggTransferRequestType.PREFILL_D2H,
        src_offsets=[4, 6],
        dst_offsets=[0, 2],
        sizes=[2, 2],
        peer="decode",
        callback=prefill_cb,
    )

    # 6. Wait for completion
    if not prefill_done.wait(timeout=10.0):
      self.fail("Prefill transfer timed out")
    if not decode_done.wait(timeout=10.0):
      self.fail("Decode transfer timed out")

    # Stop managers
    prefill_manager.stop()
    decode_manager.stop()

    # Check errors
    if errors:
      self.fail("\n".join(errors))

    # 7. Verify Results
    for i in range(n_layers):
      decode_np = np.asarray(decode_tpu_arrs[i])
      ref_np = np.asarray(prefill_ref_arrs[i])
      np.testing.assert_array_equal(decode_np[0:4], ref_np[4:8])

  def test_e2e_disagg_push_concurrent(self):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(202)
    dtype = jnp.float32

    n_layers = 2
    block_size = 2
    test_shape = (8, 128, 8, 8, 128)

    # 1. Setup Prefill data (two independent streams)
    prefill_ref_arrs1 = []
    prefill_tpu_arrs1 = []
    prefill_ref_arrs2 = []
    prefill_tpu_arrs2 = []

    for i in range(n_layers):
      key1 = jax.random.fold_in(key, i * 2)
      key2 = jax.random.fold_in(key, i * 2 + 1)

      base1 = jax.random.uniform(key1, test_shape, dtype=dtype)
      prefill_ref_arrs1.append(base1)
      prefill_tpu_arrs1.append(jax.device_put(base1, tpu_sharding))

      base2 = jax.random.uniform(key2, test_shape, dtype=dtype)
      prefill_ref_arrs2.append(base2)
      prefill_tpu_arrs2.append(jax.device_put(base2, tpu_sharding))

    jax.block_until_ready(prefill_tpu_arrs1 + prefill_tpu_arrs2)

    # 2. Setup Decode empty buffers (two independent targets)
    decode_tpu_arrs1 = [
        jax.device_put(jnp.zeros(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    decode_tpu_arrs2 = [
        jax.device_put(jnp.zeros(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(decode_tpu_arrs1 + decode_tpu_arrs2)

    # 3. Initialize managers
    prefill_manager1 = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs1,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    prefill_manager2 = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs2,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    decode_manager1 = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs1,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    decode_manager2 = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs2,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=4,
        unsafe_skip_buffer_lock=self.skip_lock,
    )

    prefill_manager1.start()
    prefill_manager2.start()
    decode_manager1.start()
    decode_manager2.start()

    time.sleep(0.1)

    # Bootstrap cross connections
    prefill_manager1.register_peer(
        "decode1",
        "127.0.0.1",
        decode_manager1.zmq_control_port(),
        decode_manager1.local_port(),
    )
    prefill_manager2.register_peer(
        "decode2",
        "127.0.0.1",
        decode_manager2.zmq_control_port(),
        decode_manager2.local_port(),
    )

    # Events for sync
    done_events = [threading.Event() for _ in range(4)]
    errors = []

    def make_cb(idx):
      return lambda status: (
          errors.append(f"Error at {idx}: {status}")
          if status is not None
          else None,
          done_events[idx].set(),
      )

    # 4. Submit Decode Receive Requests
    decode_manager1.submit_request(
        1001,
        disagg_kv_cache_manager.DisaggTransferRequestType.DECODE_H2D,
        dst_offsets=[0, 2],
        sizes=[2, 2],
        callback=make_cb(0),
    )
    decode_manager2.submit_request(
        1002,
        disagg_kv_cache_manager.DisaggTransferRequestType.DECODE_H2D,
        dst_offsets=[0, 2],
        sizes=[2, 2],
        callback=make_cb(1),
    )

    # 5. Submit Prefill Send Requests concurrently!
    prefill_manager1.submit_request(
        1001,
        disagg_kv_cache_manager.DisaggTransferRequestType.PREFILL_D2H,
        src_offsets=[4, 6],
        dst_offsets=[0, 2],
        sizes=[2, 2],
        peer="decode1",
        callback=make_cb(2),
    )
    prefill_manager2.submit_request(
        1002,
        disagg_kv_cache_manager.DisaggTransferRequestType.PREFILL_D2H,
        src_offsets=[4, 6],
        dst_offsets=[0, 2],
        sizes=[2, 2],
        peer="decode2",
        callback=make_cb(3),
    )

    # 6. Wait for all
    for i, ev in enumerate(done_events):
      if not ev.wait(timeout=15.0):
        self.fail(f"Transfer {i} timed out")

    # Stop all
    prefill_manager1.stop()
    prefill_manager2.stop()
    decode_manager1.stop()
    decode_manager2.stop()

    if errors:
      self.fail("\n".join(errors))

    # 7. Verify Results
    for i in range(n_layers):
      dec_np1 = np.asarray(decode_tpu_arrs1[i])
      ref_np1 = np.asarray(prefill_ref_arrs1[i])
      np.testing.assert_array_equal(dec_np1[0:4], ref_np1[4:8])

      dec_np2 = np.asarray(decode_tpu_arrs2[i])
      ref_np2 = np.asarray(prefill_ref_arrs2[i])
      np.testing.assert_array_equal(dec_np2[0:4], ref_np2[4:8])

  def test_e2e_disagg_push_multi_request_concurrent(self):
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(303)
    dtype = jnp.float32
    parallelism = 2

    n_layers = 2
    block_size = 2
    test_shape = (16, 128, 8, 8, 128)

    # 1. Setup Prefill data (16 slices)
    prefill_ref_arrs = []
    prefill_tpu_arrs = []
    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      prefill_ref_arrs.append(base)
      prefill_tpu_arrs.append(jax.device_put(base, tpu_sharding))
    jax.block_until_ready(prefill_tpu_arrs)

    # 2. Setup Decode empty buffers (16 slices)
    decode_tpu_arrs = [
        jax.device_put(jnp.zeros(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(decode_tpu_arrs)

    # 3. Initialize managers with parallelism = 2!
    prefill_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=8,
        unsafe_skip_buffer_lock=self.skip_lock,
        parallelism=parallelism,
    )
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=8,
        unsafe_skip_buffer_lock=self.skip_lock,
        parallelism=parallelism,
    )

    prefill_manager.start()
    decode_manager.start()

    time.sleep(0.1)

    prefill_manager.register_peer(
        "decode",
        "127.0.0.1",
        decode_manager.zmq_control_port(),
        decode_manager.local_port(),
    )

    # Events for sync (2 requests * 2 callbacks = 4 events)
    done_events = [threading.Event() for _ in range(4)]
    errors = []

    def make_cb(idx):
      return lambda status: (
          errors.append(f"Error at {idx}: {status}")
          if status is not None
          else None,
          done_events[idx].set(),
      )

    # 4. Submit Decode Receive Requests (req 1001 and 1002)
    decode_manager.submit_request(
        1001,
        disagg_kv_cache_manager.DisaggTransferRequestType.DECODE_H2D,
        dst_offsets=[0, 2],
        sizes=[2, 2],
        callback=make_cb(0),
    )
    decode_manager.submit_request(
        1002,
        disagg_kv_cache_manager.DisaggTransferRequestType.DECODE_H2D,
        dst_offsets=[4, 6],
        sizes=[2, 2],
        callback=make_cb(1),
    )

    # 5. Submit Prefill Send Requests concurrently!
    prefill_manager.submit_request(
        1001,
        disagg_kv_cache_manager.DisaggTransferRequestType.PREFILL_D2H,
        src_offsets=[4, 6],
        dst_offsets=[0, 2],
        sizes=[2, 2],
        peer="decode",
        callback=make_cb(2),
    )
    prefill_manager.submit_request(
        1002,
        disagg_kv_cache_manager.DisaggTransferRequestType.PREFILL_D2H,
        src_offsets=[8, 10],
        dst_offsets=[4, 6],
        sizes=[2, 2],
        peer="decode",
        callback=make_cb(3),
    )

    # 6. Wait for all
    for i, ev in enumerate(done_events):
      if not ev.wait(timeout=15.0):
        self.fail(f"Transfer {i} timed out")

    # Stop
    prefill_manager.stop()
    decode_manager.stop()

    if errors:
      self.fail("\n".join(errors))

    # 7. Verify Results
    for i in range(n_layers):
      dec_np = np.asarray(decode_tpu_arrs[i])
      ref_np = np.asarray(prefill_ref_arrs[i])

      np.testing.assert_array_equal(dec_np[0:4], ref_np[4:8])
      np.testing.assert_array_equal(dec_np[4:8], ref_np[8:12])


if __name__ == "__main__":
  absltest.main()
