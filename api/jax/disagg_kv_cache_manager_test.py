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
  def test_e2e_disagg_pull(self, dtype):
    """End-to-end PULL transfer via await_pull()/pull().

    The decode initiates the data movement: prefill stages (D2H) and advertises
    readiness via NOTIFY_READY, the decode pulls the blocks (H2H Read) then
    loads them (H2D) and acks with PULL_COMPLETE. Needs BOTH directions
    registered, and both sides pass the same uuid and peer=<other engine>.
    """
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(404)

    n_layers = 2
    block_size = 1  # block (page) size along the major dim; src/dst_offsets and
    # sizes are counts of major-dim slices of the device array.
    test_shape = (8, 128, 8, 8, 128)

    # 1. Setup Prefill (source) data
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

    # 2. Setup Decode (puller) empty buffers
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
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
    )

    prefill_manager.start()
    decode_manager.start()
    time.sleep(0.1)

    # Pull needs BOTH directions registered: prefill -> decode for NOTIFY_READY,
    # decode -> prefill for the H2H Read pull and the PULL_COMPLETE ack.
    prefill_manager.register_peer(
        "decode",
        "127.0.0.1",
        decode_manager.zmq_control_port(),
        decode_manager.local_port(),
    )
    decode_manager.register_peer(
        "prefill",
        "127.0.0.1",
        prefill_manager.zmq_control_port(),
        prefill_manager.local_port(),
    )

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

    # 4. Decode (the puller) sets up its receive first. It specifies only the
    # destination device offsets; its local staging is manager-allocated.
    decode_manager.pull(
        uuid=2001,
        req_id=2001,
        src_offsets=[4, 5, 6, 7],
        dst_offsets=[0, 1, 2, 3],
        sizes=[1, 1, 1, 1],
        peer="prefill",
        callback=decode_cb,
    )

    # 5. Prefill stages and advertises readiness. It specifies only the source
    # device offsets; the host staging blocks are manager-allocated.
    prefill_manager.await_pull(
        uuid=2001,
        req_id=2001,
        src_offsets=[4, 5, 6, 7],
        sizes=[1, 1, 1, 1],
        peer="decode",
        callback=prefill_cb,
    )

    # 6. Wait for completion
    if not prefill_done.wait(timeout=10.0):
      self.fail("Prefill transfer timed out")
    if not decode_done.wait(timeout=10.0):
      self.fail("Decode transfer timed out")

    prefill_manager.stop()
    decode_manager.stop()

    if errors:
      self.fail("\n".join(errors))

    # 7. Verify Results
    for i in range(n_layers):
      decode_np = np.asarray(decode_tpu_arrs[i])
      ref_np = np.asarray(prefill_ref_arrs[i])
      np.testing.assert_array_equal(decode_np[0:4], ref_np[4:8])

  def test_e2e_disagg_pull_noncontiguous(self):
    """PULL into NON-CONTIGUOUS device destination offsets.

    Staging is now manager-allocated, so the caller no longer controls staging
    block ids; instead this drives a gapped DEVICE destination: the decode pulls
    two single-slice chunks into device offsets [0, 2] (block 1 is skipped and
    must stay zero), while the prefill supplies source slices [4] and [6]. This
    keeps coverage that scattered device destinations verify end-to-end. (The
    BlockTransport's non-contiguous Pull coalescing remains covered by the C++
    block_transport_test cases.)
    """
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(505)
    dtype = jnp.float32

    n_layers = 2
    block_size = 1  # block (page) size along the major dim; src/dst_offsets and
    # sizes are counts of major-dim slices of the device array.
    test_shape = (8, 128, 8, 8, 128)

    # 1. Setup Prefill (source) data
    prefill_ref_arrs = []
    prefill_tpu_arrs = []
    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      prefill_ref_arrs.append(base)
      prefill_tpu_arrs.append(jax.device_put(base, tpu_sharding))
    jax.block_until_ready(prefill_tpu_arrs)

    # 2. Setup Decode (puller) empty buffers
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
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
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
    decode_manager.register_peer(
        "prefill",
        "127.0.0.1",
        prefill_manager.zmq_control_port(),
        prefill_manager.local_port(),
    )

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

    # 4. Decode pulls into non-contiguous device offsets [0, 2] (device block 1
    # is skipped and must stay zero). Staging is manager-allocated.
    decode_manager.pull(
        uuid=2002,
        req_id=2002,
        src_offsets=[4, 6],
        dst_offsets=[0, 2],
        sizes=[1, 1],
        peer="prefill",
        callback=decode_cb,
    )

    # 5. Prefill supplies source device slices [4:5] and [6:7]; the host staging
    # blocks are manager-allocated.
    prefill_manager.await_pull(
        uuid=2002,
        req_id=2002,
        src_offsets=[4, 6],
        sizes=[1, 1],
        peer="decode",
        callback=prefill_cb,
    )

    if not prefill_done.wait(timeout=10.0):
      self.fail("Prefill transfer timed out")
    if not decode_done.wait(timeout=10.0):
      self.fail("Decode transfer timed out")

    prefill_manager.stop()
    decode_manager.stop()

    if errors:
      self.fail("\n".join(errors))

    # 7. Verify: decode[0:1] <- prefill[4:5], decode[2:3] <- prefill[6:7], and
    # the skipped block (decode[1:2]) stayed zero.
    for i in range(n_layers):
      dec_np = np.asarray(decode_tpu_arrs[i])
      ref_np = np.asarray(prefill_ref_arrs[i])
      np.testing.assert_array_equal(dec_np[0:1], ref_np[4:5])
      np.testing.assert_array_equal(dec_np[2:3], ref_np[6:7])
      np.testing.assert_array_equal(dec_np[1:2], np.zeros_like(dec_np[1:2]))

  def test_e2e_disagg_pull_multiblock(self):
    """PULL with multi-block chunks (sizes a multiple of block_size).

    Each side gives only its own device region; staging is manager-allocated and
    chunks are expanded into unit blocks internally. block_size=1 and:

      prefill (await_pull): src_offsets=[1, 4], sizes=[2, 3]
      decode  (pull):       dst_offsets=[10, 13], sizes=[2, 3]

    The two sides group the 5 blocks as 2+3 at unrelated offsets, so a correct
    result proves the per-block expansion stays positionally aligned.
    Expect device[1:3] -> decode device[10:12], device[4:7] -> device[13:16].
    """
    tpu_sharding = self.setup_shardings()
    dtype = jnp.float32

    n_layers = 2
    block_size = 1
    test_shape = (16, 128, 8, 8, 128)  # major dim 16 so staging block 8 fits.

    key = jax.random.key(606)
    prefill_ref_arrs = []
    prefill_tpu_arrs = []
    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      prefill_ref_arrs.append(base)
      prefill_tpu_arrs.append(jax.device_put(base, tpu_sharding))
    jax.block_until_ready(prefill_tpu_arrs)

    decode_tpu_arrs = [
        jax.device_put(jnp.zeros(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(decode_tpu_arrs)

    prefill_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
    )
    prefill_manager.start()
    decode_manager.start()
    time.sleep(0.1)

    prefill_manager.register_peer(
        "decode", "127.0.0.1",
        decode_manager.zmq_control_port(), decode_manager.local_port(),
    )
    decode_manager.register_peer(
        "prefill", "127.0.0.1",
        prefill_manager.zmq_control_port(), prefill_manager.local_port(),
    )

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

    decode_manager.pull(
        uuid=2002,
        req_id=2002,
        src_offsets=[1, 4],
        dst_offsets=[10, 13],
        sizes=[2, 3],
        peer="prefill",
        callback=decode_cb,
    )
    prefill_manager.await_pull(
        uuid=2002,
        req_id=2002,
        src_offsets=[1, 4],
        sizes=[2, 3],
        peer="decode",
        callback=prefill_cb,
    )

    if not prefill_done.wait(timeout=10.0):
      self.fail("Prefill transfer timed out")
    if not decode_done.wait(timeout=10.0):
      self.fail("Decode transfer timed out")

    prefill_manager.stop()
    decode_manager.stop()
    if errors:
      self.fail("\n".join(errors))

    for i in range(n_layers):
      dec_np = np.asarray(decode_tpu_arrs[i])
      ref_np = np.asarray(prefill_ref_arrs[i])
      np.testing.assert_array_equal(dec_np[10:12], ref_np[1:3])
      np.testing.assert_array_equal(dec_np[13:16], ref_np[4:7])

  def test_e2e_disagg_pull_multi_request_concurrent(self):
    """Two concurrent pulls on one manager pair with parallelism=2.

    Stresses concurrent producer-side staging auto-allocation (the staging for
    the two requests may be handed out non-sequentially) plus the per-request
    staging Unlock on pull-complete, alongside the concurrent block-allocation /
    NOTIFY path that previously corrupted KV.
    """
    tpu_sharding = self.setup_shardings()
    key = jax.random.key(303)
    dtype = jnp.float32
    parallelism = 2

    n_layers = 2
    block_size = 1  # block (page) size along the major dim; src/dst_offsets and
    # sizes are counts of major-dim slices of the device array.
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

    # 3. Initialize managers with parallelism = 2 on BOTH axes: per-transfer TCP
    # streams (transport_parallelism) and concurrent H2H workers
    # (worker_parallelism). This stresses the concurrent block-allocation /
    # NOTIFY path that previously corrupted KV.
    prefill_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
        transport_parallelism=parallelism,
        worker_parallelism=parallelism,
    )
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs,
        block_size=block_size,
        local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock,
        transport_parallelism=parallelism,
        worker_parallelism=parallelism,
    )

    prefill_manager.start()
    decode_manager.start()

    time.sleep(0.1)

    # Pull mode needs bidirectional peer registration.
    prefill_manager.register_peer(
        "decode",
        "127.0.0.1",
        decode_manager.zmq_control_port(),
        decode_manager.local_port(),
    )
    decode_manager.register_peer(
        "prefill",
        "127.0.0.1",
        prefill_manager.zmq_control_port(),
        prefill_manager.local_port(),
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

    # 4. Decode (puller) sets up both receives; local staging is auto-allocated.
    decode_manager.pull(
        uuid=1001,
        req_id=1001,
        src_offsets=[4, 5, 6, 7],
        dst_offsets=[0, 1, 2, 3],
        sizes=[1, 1, 1, 1],
        peer="prefill",
        callback=make_cb(0),
    )
    decode_manager.pull(
        uuid=1002,
        req_id=1002,
        src_offsets=[8, 9, 10, 11],
        dst_offsets=[4, 5, 6, 7],
        sizes=[1, 1, 1, 1],
        peer="prefill",
        callback=make_cb(1),
    )

    # 5. Prefill stages both sources concurrently; host staging is auto-allocated
    # by the manager (the two requests' blocks may be handed out non-sequentially).
    prefill_manager.await_pull(
        uuid=1001,
        req_id=1001,
        src_offsets=[4, 5, 6, 7],
        sizes=[1, 1, 1, 1],
        peer="decode",
        callback=make_cb(2),
    )
    prefill_manager.await_pull(
        uuid=1002,
        req_id=1002,
        src_offsets=[8, 9, 10, 11],
        sizes=[1, 1, 1, 1],
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

  def test_e2e_disagg_pull_src_mismatch_rejected(self):
    """A pull() whose src_offsets disagree with the producer's await_pull is

    rejected at the handshake (identity validation): the consumer's callback
    fires with a non-None error message and no data is written into the decode
    buffer.
    """
    tpu_sharding = self.setup_shardings()
    dtype = jnp.float32
    n_layers = 2
    block_size = 1
    test_shape = (8, 128, 8, 8, 128)

    key = jax.random.key(707)
    prefill_tpu_arrs = []
    for i in range(n_layers):
      base = jax.random.uniform(jax.random.fold_in(key, i), test_shape,
                                dtype=dtype)
      prefill_tpu_arrs.append(jax.device_put(base, tpu_sharding))
    decode_tpu_arrs = [
        jax.device_put(jnp.zeros(test_shape, dtype=dtype), tpu_sharding)
        for _ in range(n_layers)
    ]
    jax.block_until_ready(prefill_tpu_arrs + decode_tpu_arrs)

    prefill_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=prefill_tpu_arrs, block_size=block_size, local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock)
    decode_manager = disagg_kv_cache_manager.DisaggKVCacheManager(
        device_arrays=decode_tpu_arrs, block_size=block_size, local_port=0,
        host_blocks_to_allocate=test_shape[0] // block_size,
        unsafe_skip_buffer_lock=self.skip_lock)
    prefill_manager.start()
    decode_manager.start()
    time.sleep(0.1)
    prefill_manager.register_peer(
        "decode", "127.0.0.1", decode_manager.zmq_control_port(),
        decode_manager.local_port())
    decode_manager.register_peer(
        "prefill", "127.0.0.1", prefill_manager.zmq_control_port(),
        prefill_manager.local_port())

    decode_done = threading.Event()
    decode_status = []

    def decode_cb(status):
      decode_status.append(status)
      decode_done.set()

    # Consumer asks for src [0,1,2,3]; producer staged src [4,5,6,7]. Same uuid.
    decode_manager.pull(
        uuid=909, req_id=909, src_offsets=[0, 1, 2, 3],
        dst_offsets=[0, 1, 2, 3], sizes=[1, 1, 1, 1], peer="prefill",
        callback=decode_cb)
    prefill_manager.await_pull(
        uuid=909, req_id=909, src_offsets=[4, 5, 6, 7],
        sizes=[1, 1, 1, 1], peer="decode", callback=lambda s: None)

    # The callback must fire with a non-None error message (validation rejected
    # the pull), and no data may be written into the decode buffer.
    self.assertTrue(decode_done.wait(timeout=10.0),
                    "decode callback should fire on rejection")
    prefill_manager.stop()
    decode_manager.stop()

    self.assertEqual(len(decode_status), 1)
    self.assertIsNotNone(decode_status[0],
                         "src mismatch should report an error to the callback")
    self.assertIn("source region", str(decode_status[0]))
    for arr in decode_tpu_arrs:
      np.testing.assert_array_equal(np.asarray(arr)[0:4],
                                    np.zeros_like(np.asarray(arr)[0:4]))


if __name__ == "__main__":
  absltest.main()
