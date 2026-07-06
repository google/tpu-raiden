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
import subprocess
import time
import unittest

if not os.path.exists("/dev/accel0"):
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

from absl.testing import absltest
import portpicker

from google3.pyglib import resources
from tpu_raiden.api.jax import kv_cache_store
from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager
import jax
import jax.numpy as jnp
import numpy as np

# Global variables for subprocesses
_orchestrator_process = None
_registry_process = None
_orchestrator_port = None
_registry_port = None


def setUpModule():
  global _orchestrator_process, _registry_process
  global _orchestrator_port, _registry_port

  _orchestrator_port = portpicker.pick_unused_port()
  _registry_port = portpicker.pick_unused_port()

  orchestrator_binary = resources.GetResourceFilename(
      "google3/third_party/tpu_raiden/tpu_raiden/kv_cache/raiden_orchestrator_main"
  )
  registry_binary = resources.GetResourceFilename(
      "google3/third_party/tpu_raiden/tpu_raiden/kv_cache/global_registry/global_registry_server"
  )

  print(f"Starting Orchestrator on port {_orchestrator_port}")
  _orchestrator_process = subprocess.Popen([
      orchestrator_binary,
      f"--port={_orchestrator_port}",
      "--bind_ip=0.0.0.0",
      "--alsologtostderr",
  ])

  print(f"Starting Registry on port {_registry_port}")
  _registry_process = subprocess.Popen([
      registry_binary,
      f"--port={_registry_port}",
      "--alsologtostderr",
  ])

  # Give them some time to start
  time.sleep(2)


def tearDownModule():
  global _orchestrator_process, _registry_process
  if _orchestrator_process:
    _orchestrator_process.terminate()
    _orchestrator_process.wait()
  if _registry_process:
    _registry_process.terminate()
    _registry_process.wait()


class KVCacheStoreTest(absltest.TestCase):

  def _make_sharding(self, devices):
    import jax.experimental.mesh_utils as mesh_utils

    n = len(devices)
    if n >= 4:
      device_mesh = mesh_utils.create_device_mesh((2, n // 2), devices)
    else:
      device_mesh = mesh_utils.create_device_mesh((1, n), devices)
    mesh = jax.sharding.Mesh(device_mesh, ("x", "y"))
    return jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, None, "x", "y", None)
    )

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except Exception:
      self.devices = jax.devices()

    self.num_devices = len(self.devices)
    self.sharding = self._make_sharding(self.devices)
    self.mesh = self.sharding.mesh
    print(f"DEBUG_DEVICES: {self.devices}")
    print(f"DEBUG_SHARDING: {self.sharding}")

  def _require_tpu(self):
    if self.devices[0].platform != "tpu":
      self.skipTest("This test requires TPU hardware")

  def _make_device_array(self, np_array, sharding=None):
    if sharding is None:
      sharding = self.sharding
    return jax.device_put(jnp.array(np_array), sharding).block_until_ready()

  def test_basic_tests(self):
    print("TEST_BASIC_DEVICES:", jax.devices())
    print("TEST_BASIC_DEVICE_COUNT:", jax.device_count())
    print("TEST_BASIC_MESH_DEVICES:", self.mesh.devices)
    print("TEST_BASIC_MESH_SHAPE:", self.mesh.shape)
    controller = kv_cache_store.KVCacheStore(capacity=20)
    self.assertEqual(controller.capacity(), 20)

    hashes = [b"6001", b"6002"]
    slices = [
        kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0),
        kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0),
    ]

    # 1. Insert
    self.assertTrue(controller.insert(hashes, slices, True)[0])
    self.assertFalse(
        controller.insert(hashes, slices, True)[0]
    )  # Already exists

    # 2. Lookup with a partial miss at the end
    hashes_with_miss = [b"6001", b"6002", b"6003"]
    lookup_res = controller.lookup(hashes_with_miss)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], b"6001")
    self.assertEqual(lookup_res[0][1].raiden_id.job_name, "inference_server")
    self.assertEqual(lookup_res[0][1].raiden_id.job_replica_id, "0")

    # Lookup with an early miss
    hashes_early_miss = [b"6001", b"6003", b"6002"]
    lookup_res_early = controller.lookup(hashes_early_miss)
    self.assertLen(lookup_res_early, 1)
    self.assertEqual(lookup_res_early[0][0], b"6001")

    # 3. Delete
    controller.delete(hashes, slices)
    self.assertTrue(
        controller.insert(hashes, slices, True)[0]
    )  # Successful again

  def test_pin_and_release(self):
    controller = kv_cache_store.KVCacheStore(capacity=2)

    hashes = [b"7001", b"7002"]
    slices = [
        kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0),
        kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0),
    ]

    self.assertTrue(controller.insert(hashes, slices, True)[0])

    # Pin both
    self.assertTrue(controller.pin(hashes))

    # Inserting a third element should fail to evict because both items are
    # pinned.
    hash_3 = [b"7003"]
    slice_3 = [kv_cache_store.RaidenId("inference_server", "2", "kv_cache", 0)]
    controller.insert(hash_3, slice_3, True)

    # Release 7001
    controller.release([b"7001"])

    # Now inserting a fourth element (7004) should successfully evict 7001
    hash_4 = [b"7004"]
    slice_4 = [kv_cache_store.RaidenId("inference_server", "3", "kv_cache", 0)]
    controller.insert(hash_4, slice_4, True)

    self.assertEmpty(controller.lookup([b"7001", b"7002"]))
    self.assertLen(controller.lookup([b"7002"]), 1)

  def test_partial_pin_rollback(self):
    controller = kv_cache_store.KVCacheStore(capacity=2)

    hashes = [b"8001", b"8002"]
    slices = [
        kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0),
        kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0),
    ]
    self.assertTrue(controller.insert(hashes, slices, True)[0])

    # Attempt to pin a sequence with a missing hash (8003).
    self.assertFalse(controller.pin([b"8001", b"8002", b"8003"]))

    # Now inserting two new items (8004, 8005) should successfully evict 8001
    # and 8002 because their pins were completely rolled back!
    self.assertTrue(
        controller.insert(
            [b"8004", b"8005"],
            [
                kv_cache_store.RaidenId("inference_server", "2", "kv_cache", 0),
                kv_cache_store.RaidenId("inference_server", "3", "kv_cache", 0),
            ],
            True,
        )[0]
    )

    self.assertEmpty(controller.lookup([b"8001", b"8002"]))
    self.assertLen(controller.lookup([b"8004", b"8005"]), 2)

  def test_large_and_arbitrary_length_hashes(self):
    controller = kv_cache_store.KVCacheStore(capacity=5)

    # Test both high-bit 8-byte hash and a very long arbitrary length hash
    large_hash = b"\xff" * 8
    long_hash = b"a" * 100
    hashes = [large_hash, long_hash]
    slices = [
        kv_cache_store.RaidenId("inference_server", "0", "kv_cache", 0),
        kv_cache_store.RaidenId("inference_server", "1", "kv_cache", 0),
    ]

    self.assertTrue(controller.insert(hashes, slices, True)[0])

    lookup_res = controller.lookup(hashes)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], large_hash)
    self.assertEqual(lookup_res[1][0], long_hash)

  def test_global_lookup_case1_local_hit(self):
    # Case 1: Full local hit, no global hit.
    # We don't need a registry server for this because it shouldn't be queried.
    controller = kv_cache_store.KVCacheStore(capacity=20)
    hashes = [b"local_only"]
    slices = [
        kv_cache_store.RaidenId("local_job", "0", "kv_cache", 0),
    ]
    self.assertTrue(controller.insert(hashes, slices, True)[0])

    res = controller.lookup(hashes, enable_global=True)
    self.assertLen(res, 1)
    self.assertEqual(res[0][0], b"local_only")
    self.assertEqual(res[0][1].raiden_id.job_name, "local_job")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 0)

  def test_global_lookup_case2_and_3_mocked(self):
    # We mock _impl to simulate Case 2 and Case 3 because we don't
    # have a running registry server in Python tests.
    controller = kv_cache_store.KVCacheStore(capacity=20)

    # Create a mock for the C++ impl
    mock_impl = unittest.mock.MagicMock()
    controller._impl = mock_impl

    # Case 2: Both local and global have the same hit, but we return local.
    local_id = kv_cache_store._impl.RaidenBlockID(
        kv_cache_store._impl.RaidenId("local_job", "0", "kv_cache", 1)
    )
    mock_impl.lookup.return_value = [(b"shared_hash", local_id)]

    res = controller.lookup([b"shared_hash"], enable_global=True)
    self.assertLen(res, 1)
    self.assertEqual(res[0][0], b"shared_hash")
    self.assertEqual(res[0][1].raiden_id.job_name, "local_job")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 1)
    mock_impl.lookup.assert_called_with([b"shared_hash"], True)

    # Case 3: No local hit, only global hits.
    remote_id1 = kv_cache_store._impl.RaidenBlockID(
        kv_cache_store._impl.RaidenId("10.0.0.1:1234", "0", "kv_cache", 42)
    )
    remote_id2 = kv_cache_store._impl.RaidenBlockID(
        kv_cache_store._impl.RaidenId("10.0.0.2:1234", "0", "kv_cache", 43)
    )
    mock_impl.lookup.return_value = [
        (b"global_1", remote_id1),
        (b"global_2", remote_id2),
    ]

    res = controller.lookup([b"global_1", b"global_2"], enable_global=True)
    self.assertLen(res, 2)
    self.assertEqual(res[0][0], b"global_1")
    self.assertEqual(res[0][1].raiden_id.job_name, "10.0.0.1:1234")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 42)
    self.assertEqual(res[1][0], b"global_2")
    self.assertEqual(res[1][1].raiden_id.job_name, "10.0.0.2:1234")
    self.assertEqual(res[1][1].raiden_id.data_replica_idx, 43)
    mock_impl.lookup.assert_called_with([b"global_1", b"global_2"], True)

  def test_global_lookup_error_ignored(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=20, global_registry_address="invalid.address:12345"
    )
    hashes = [b"9001"]
    # Should not fail, just return empty because the registry is down
    res = controller.lookup(hashes, enable_global=True)
    self.assertEmpty(res)

  def test_insert_and_pin_release_and_delete(self):
    controller = kv_cache_store.KVCacheStore(capacity=2)

    local_hashes = [b"local_1", b"local_2"]
    local_slices = [
        kv_cache_store.RaidenBlockID(
            kv_cache_store.RaidenId("local_job", "0", "kv_cache", 0),
            -1,
            kv_cache_store.BlockStatus.HOST,
        ),
        kv_cache_store.RaidenBlockID(
            kv_cache_store.RaidenId("local_job", "0", "kv_cache", 1),
            -1,
            kv_cache_store.BlockStatus.HOST,
        ),
    ]
    self.assertTrue(controller.insert(local_hashes, local_slices, True)[0])

    remote_hashes = [b"remote_1", b"remote_2"]
    remote_slices = [
        kv_cache_store.RaidenBlockID(
            kv_cache_store.RaidenId("remote_job", "0", "kv_cache", 0),
            -1,
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            kv_cache_store.RaidenId("remote_job", "0", "kv_cache", 1),
            -1,
            kv_cache_store.BlockStatus.REMOTE,
        ),
    ]
    success, evicted = controller.insert_and_pin(
        remote_hashes, remote_slices, True
    )
    self.assertTrue(success)
    self.assertLen(evicted, 2)
    self.assertEmpty(controller.lookup([b"local_1"]))

    del_count, rem_evicted = controller.release_and_delete(
        remote_hashes, evicted
    )
    self.assertEqual(del_count, 2)
    self.assertEmpty(rem_evicted)
    self.assertLen(controller.lookup([b"local_1", b"local_2"]), 2)

  def test_fetch_remote_basic(self):
    config = kv_cache_store.RemoteFetchConfig()
    config.orchestrator_address = f"localhost:{_orchestrator_port}"
    config.controller_port = 0  # Ephemeral
    config.local_worker_port = 0  # Ephemeral
    config.bytes_per_block = 1024
    config.num_shards = 1

    controller = kv_cache_store.KVCacheStore(capacity=2, remote_config=config)

    remote_hashes = [b"remote_1"]
    remote_slices = [
        kv_cache_store.RaidenBlockID(
            kv_cache_store.RaidenId("remote_job", "0", "kv_cache", 0),
            5,  # host_block_id
            kv_cache_store.BlockStatus.REMOTE,
        )
    ]
    self.assertTrue(controller.insert(remote_hashes, remote_slices, True)[0])

    # Call fetch_remote
    futures = controller.fetch_remote(remote_hashes)
    self.assertIn(b"remote_1", futures)
    future = futures[b"remote_1"]
    self.assertFalse(future.IsDone())

    # Poll status
    done, failed, pending = controller.poll_fetch_remote_status()
    self.assertEmpty(done)
    self.assertEmpty(failed)
    self.assertEqual(pending, [b"remote_1"])

  def test_e2e_remote_fetch_cpu(self):
    """Tests end-to-end remote fetch on CPU."""
    self._require_tpu()
    # Enable VLOG for debugging
    # We can't easily set absl flags in python if they are C++ flags, but we can try environment variables
    # or just rely on the fact that they might be logged to stderr anyway.
    # Let's try setting environment variables for logging if possible.

    # 1. Setup Ports
    sender_listener_port = portpicker.pick_unused_port()
    receiver_listener_port = portpicker.pick_unused_port()
    sender_controller_port = portpicker.pick_unused_port()
    receiver_controller_port = portpicker.pick_unused_port()

    print(f"Sender Listener Port: {sender_listener_port}")
    print(f"Receiver Listener Port: {receiver_listener_port}")
    print(f"Sender Controller Port: {sender_controller_port}")
    print(f"Receiver Controller Port: {receiver_controller_port}")

    # 2. Setup JAX arrays (CPU)
    # Using small shape for testing
    shape = (1, 128, 8, 8, 128)
    sender_kv_caches = [
        self._make_device_array(np.ones(shape, dtype=np.float32))
    ]
    receiver_kv_caches = [
        self._make_device_array(np.zeros(shape, dtype=np.float32))
    ]

    sender_mgr = KVCacheManager(
        kv_caches=sender_kv_caches,
        node_id=0,
        local_control_port=0,  # Ephemeral for Manager's own internal server
        max_blocks=4,
        num_slots=2,
        listener_port=sender_listener_port,
        listener_controller_port=sender_controller_port,
    )
    self.assertTrue(sender_mgr.is_listener_active)

    receiver_mgr = KVCacheManager(
        kv_caches=receiver_kv_caches,
        node_id=0,
        local_control_port=0,  # Ephemeral for Manager's own internal server
        max_blocks=4,
        num_slots=2,
        listener_port=receiver_listener_port,
        listener_controller_port=receiver_controller_port,
    )
    self.assertTrue(receiver_mgr.is_listener_active)

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = self.num_devices
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # 4. Instantiate KVCacheStores (Control Plane)
    sender_config = kv_cache_store.RemoteFetchConfig()
    sender_config.orchestrator_address = f"localhost:{_orchestrator_port}"
    sender_config.controller_port = sender_controller_port
    sender_config.local_worker_port = sender_listener_port
    sender_config.bytes_per_block = local_bytes_per_block
    sender_config.num_shards = self.num_devices
    sender_config.num_listeners = 1

    sender_id = kv_cache_store.RaidenId("sender_job", "0", "kv_cache", 0)
    sender_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=sender_id,
        remote_config=sender_config,
    )

    receiver_config = kv_cache_store.RemoteFetchConfig()
    receiver_config.orchestrator_address = f"localhost:{_orchestrator_port}"
    receiver_config.controller_port = receiver_controller_port
    receiver_config.local_worker_port = receiver_listener_port
    receiver_config.bytes_per_block = local_bytes_per_block
    receiver_config.num_shards = self.num_devices
    receiver_config.num_listeners = 1

    receiver_id = kv_cache_store.RaidenId("receiver_job", "0", "kv_cache", 0)
    receiver_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=receiver_id,
        remote_config=receiver_config,
    )

    # Give them some time to register with orchestrator
    time.sleep(1)

    # 5. Prepare Data
    hash_1 = b"hash_1"
    # Sender has block 0 locally
    sender_slices = [
        kv_cache_store.RaidenBlockID(
            sender_id,
            0,  # host_block_id
            kv_cache_store.BlockStatus.HOST,
        )
    ]
    self.assertTrue(sender_store.insert([hash_1], sender_slices, True)[0])

    # NOTE: We simulate global registry lookup here by inserting REMOTE status directly.
    # When write / delete through in KVCacheStore is implemented, we can test with a real global registry.
    # Receiver knows block is remote at sender
    receiver_slices = [
        kv_cache_store.RaidenBlockID(
            sender_id,
            0,  # host_block_id at sender
            kv_cache_store.BlockStatus.REMOTE,
        )
    ]
    # We need to insert it into receiver store so fetch_remote can look it up?
    # Wait, fetch_remote takes hashes, and looks up in local store to find where it is.
    self.assertTrue(receiver_store.insert([hash_1], receiver_slices, True)[0])

    # 6. Trigger Fetch
    futures = receiver_store.fetch_remote([hash_1])
    self.assertIn(hash_1, futures)
    future = futures[hash_1]

    # 7. Wait for completion
    # Await might block indefinitely if something is wrong, so let's use a timeout loop with poll
    max_wait = 10
    start_time = time.time()
    completed = False
    while time.time() - start_time < max_wait:
      done, failed, pending = receiver_store.poll_fetch_remote_status()
      if hash_1 in done:
        completed = True
        break
      if hash_1 in failed:
        self.fail(f"Fetch failed for {hash_1}")
      time.sleep(0.5)

    self.assertTrue(completed, "Fetch timed out")
    self.assertTrue(future.IsDone())

    # TODO: Verify data content if possible, but for now just success is good.

  def test_queue_flow_embedded(self):
    self._require_tpu()
    print("test_queue_flow_embedded starting...")
    """Verifies high-level Fetch API with embedded controller, including multiple blocks, layers, and shards."""
    sender_listener_port = portpicker.pick_unused_port()
    receiver_listener_port = portpicker.pick_unused_port()
    sender_controller_port = portpicker.pick_unused_port()
    receiver_controller_port = portpicker.pick_unused_port()

    num_blocks = 4
    num_layers = 2
    shape = (num_blocks, 128, 8, 8, 128)

    sender_kv_caches = []
    receiver_kv_caches = []

    for l in range(num_layers):
      # Fill with non-zero values
      sender_data = np.full(shape, fill_value=float(l + 1), dtype=np.float32)
      sender_kv_caches.append(self._make_device_array(sender_data))
      print(f"Layer {l} sender_data sample: {sender_data[2, 0, 0, 0, :5]}")

      # Use jnp.zeros to create device buffers
      buf = np.zeros(shape, dtype=np.float32)
      receiver_kv_caches.append(self._make_device_array(buf))
      print(f"Layer {l} receiver buffer id before: {id(buf)}")

    # shape is (num_blocks, 128, 8, 8, 128)
    # slice_byte_size in C++ is GetMajorSliceByteSize, which is size of one block.
    # For shape (4, 128, 8, 8, 128) of float32, block size is 128 * 8 * 8 * 128 * 4 = 4194304 bytes.
    bytes_per_block = 128 * 8 * 8 * 128 * 4

    sender_mgr = KVCacheManager(
        kv_caches=sender_kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=sender_listener_port,
        listener_controller_port=sender_controller_port,
    )

    receiver_mgr = KVCacheManager(
        kv_caches=receiver_kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=receiver_listener_port,
        listener_controller_port=receiver_controller_port,
    )

    # Keep sender_mgr alive
    print(f"Sender manager alive: {sender_mgr is not None}")

    sender_raiden_id = kv_cache_store.RaidenId("job1", "0", "data1", 0)
    receiver_raiden_id = kv_cache_store.RaidenId("job2", "0", "data2", 0)

    local_shards_per_listener = self.num_devices
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    sender_remote_config = kv_cache_store.RemoteFetchConfig()
    sender_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    sender_remote_config.controller_port = sender_controller_port
    sender_remote_config.local_worker_port = sender_listener_port
    sender_remote_config.bytes_per_block = local_bytes_per_block
    sender_remote_config.num_shards = self.num_devices

    receiver_remote_config = kv_cache_store.RemoteFetchConfig()
    receiver_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    receiver_remote_config.controller_port = receiver_controller_port
    receiver_remote_config.local_worker_port = receiver_listener_port
    receiver_remote_config.bytes_per_block = local_bytes_per_block
    receiver_remote_config.num_shards = self.num_devices

    sender_store = kv_cache_store.KVCacheStore(
        capacity=100,
        global_registry_address="",
        raiden_id=sender_raiden_id,
        remote_config=sender_remote_config,
    )

    receiver_store = kv_cache_store.KVCacheStore(
        capacity=100,
        global_registry_address="",
        raiden_id=receiver_raiden_id,
        remote_config=receiver_remote_config,
    )

    # Insert blocks into sender (HOST status)
    print("Inserting into sender...")
    for l in range(num_layers):
      print(
          f"Layer {l} sender buffer id before insert: {id(sender_kv_caches[l])}"
      )

    sender_store.insert(
        [b"hash1", b"hash2"],
        [
            kv_cache_store.RaidenBlockID(
                sender_raiden_id, 2, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                sender_raiden_id, 3, kv_cache_store.BlockStatus.HOST
            ),
        ],
        on_host=True,
    )
    print("Inserted into sender.")
    for l in range(num_layers):
      print(
          f"Layer {l} sender buffer id after insert: {id(sender_kv_caches[l])}"
      )

    # NOTE: We simulate global registry lookup here by inserting REMOTE status directly.
    # When write / delete through in KVCacheStore is implemented, we can test with a real global registry.
    # Insert placeholders into receiver (REMOTE status)
    receiver_store.insert(
        [b"hash1", b"hash2"],
        [
            kv_cache_store.RaidenBlockID(
                sender_raiden_id, 2, kv_cache_store.BlockStatus.REMOTE
            ),
            kv_cache_store.RaidenBlockID(
                sender_raiden_id, 3, kv_cache_store.BlockStatus.REMOTE
            ),
        ],
        on_host=True,
    )

    # Manual D2H on sender to populate staging buffer
    print("Running manual D2H on sender...")
    sender_mgr.d2h([2, 3], [2, 3]).wait()
    print("Manual D2H on sender completed.")

    # Trigger Remote Fetch
    print("Triggering Remote Fetch...")
    receiver_store.fetch_remote([b"hash1", b"hash2"])
    print("Remote Fetch Triggered.")

    # Wait for completion
    done = False
    for i in range(50):
      done_recving, failed_recving, pending_recving = (
          receiver_store.poll_fetch_remote_status()
      )
      print(
          f"Poll {i}: done_recving={done_recving},"
          f" failed_recving={failed_recving}, pending_recving={pending_recving}"
      )
      if failed_recving:
        self.fail(f"Transfer failed: {failed_recving}")
      if done_recving:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Receiver did not finish transfer in time")

    # Manual H2D on receiver to populate JAX buffers
    print("Running manual H2D on receiver...")
    print("Calling receiver_mgr.h2d")
    receiver_mgr.h2d([2, 3], [2, 3]).wait()
    print("receiver_mgr.h2d finished")
    print("Manual H2D on receiver completed.")

    # Give time for H2D copies to complete in background if needed
    time.sleep(1)

    for l in range(num_layers):
      print(f"Layer {l} receiver buffer id after: {id(receiver_kv_caches[l])}")

    # Verify data
    for l in range(num_layers):
      receiver_data = np.asarray(receiver_kv_caches[l])
      expected_data = np.zeros(shape, dtype=np.float32)
      expected_data[2] = l + 1
      expected_data[3] = l + 1

      np.testing.assert_array_equal(receiver_data[2], expected_data[2])
      np.testing.assert_array_equal(receiver_data[3], expected_data[3])

      # [0] and [1] should remain zero
      np.testing.assert_array_equal(receiver_data[0], expected_data[0])
      np.testing.assert_array_equal(receiver_data[1], expected_data[1])
      print(f"Layer {l} verification passed! Bytes are exact.")

    # Verify Status Upgrade in Receiver Store
    # Verify Status Upgrade in Receiver Store
    results = receiver_store.lookup([b"hash1", b"hash2"])
    self.assertEqual(len(results), 2)
    for _, block in results:
      self.assertEqual(block.status, kv_cache_store.BlockStatus.HOST)

  def test_queue_flow_multi_listener_end_to_end(self):
    """Verifies remote fetch flow where both src and dst have multiple listeners."""
    self._require_tpu()
    # Robust port picking to avoid derived port collisions
    all_picked_ports = set()

    def pick_non_colliding_base(size):
      import socket

      while True:
        base = portpicker.pick_unused_port()
        derived_ports = [base + i for i in range(size)]
        if not any(p in all_picked_ports for p in derived_ports):
          bound_sockets = []
          success = True
          for p in derived_ports:
            try:
              s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
              s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
              s.bind(("localhost", p))
              bound_sockets.append(s)
            except socket.error:
              success = False
              break
          for s in bound_sockets:
            s.close()
          if success:
            for p in derived_ports:
              all_picked_ports.add(p)
            return base

    def pick_non_colliding_single():
      while True:
        port = portpicker.pick_unused_port()
        if port not in all_picked_ports:
          all_picked_ports.add(port)
          return port

    sender_listener_port_base = pick_non_colliding_base(2)
    receiver_listener_port_base = pick_non_colliding_base(2)

    port1 = sender_listener_port_base
    port2 = port1 + 1  # Consecutive

    receiver_port_base = receiver_listener_port_base

    sender_controller_port = pick_non_colliding_single()
    receiver_controller_port = pick_non_colliding_single()

    num_blocks = 10
    num_layers = 1
    shape = (num_blocks, 128, 8, 8, 128)
    bytes_per_block = 128 * 8 * 8 * 128 * 4

    # Sender Data
    sender_data_a = np.full(shape, fill_value=1.0, dtype=np.float32)
    sender_data_b = np.full(shape, fill_value=2.0, dtype=np.float32)

    if self.num_devices > 1:
      mid = self.num_devices // 2
      devices_a = self.devices[:mid]
      devices_b = self.devices[mid:]
      sharding_a = self._make_sharding(devices_a)
      sharding_b = self._make_sharding(devices_b)

      sender_buf_a = self._make_device_array(sender_data_a, sharding_a)
      sender_buf_b = self._make_device_array(sender_data_b, sharding_b)
      receiver_buf_a = self._make_device_array(
          np.zeros(shape, dtype=np.float32), sharding_a
      )
      receiver_buf_b = self._make_device_array(
          np.zeros(shape, dtype=np.float32), sharding_b
      )

      local_shards_per_listener = mid
      num_shards_config = self.num_devices
    else:
      sender_buf_a = self._make_device_array(sender_data_a)
      sender_buf_b = self._make_device_array(sender_data_b)
      receiver_buf_a = self._make_device_array(
          np.zeros(shape, dtype=np.float32)
      )
      receiver_buf_b = self._make_device_array(
          np.zeros(shape, dtype=np.float32)
      )
      local_shards_per_listener = 1
      num_shards_config = 2

    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # Sender Manager A (Shard 0)
    sender_mgr_a = KVCacheManager(
        kv_caches=[sender_buf_a],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=port1,
        listener_controller_port=sender_controller_port,
    )

    # Sender Manager B (Shard 1)
    sender_mgr_b = KVCacheManager(
        kv_caches=[sender_buf_b],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=port2,
        listener_controller_port=sender_controller_port,
    )

    # Receiver Manager A (Shard 0)
    receiver_mgr_a = KVCacheManager(
        kv_caches=[receiver_buf_a],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=receiver_port_base,
        listener_controller_port=receiver_controller_port,
    )

    # Receiver Manager B (Shard 1)
    receiver_mgr_b = KVCacheManager(
        kv_caches=[receiver_buf_b],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=receiver_port_base + 1,
        listener_controller_port=receiver_controller_port,
    )

    sender_id = kv_cache_store.RaidenId("sender_multi_job", "0", "kv_cache", 0)
    receiver_id = kv_cache_store.RaidenId("receiver_multi_job", "0", "data2", 0)

    sender_remote_config = kv_cache_store.RemoteFetchConfig()
    sender_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    sender_remote_config.controller_port = sender_controller_port
    sender_remote_config.local_worker_port = port1  # Base
    sender_remote_config.bytes_per_block = local_bytes_per_block
    sender_remote_config.num_shards = num_shards_config  # Total shards
    sender_remote_config.num_listeners = 2

    receiver_remote_config = kv_cache_store.RemoteFetchConfig()
    receiver_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    receiver_remote_config.controller_port = receiver_controller_port
    receiver_remote_config.local_worker_port = receiver_port_base  # Base
    receiver_remote_config.bytes_per_block = local_bytes_per_block
    receiver_remote_config.num_shards = num_shards_config
    receiver_remote_config.num_listeners = 2

    sender_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=sender_id,
        remote_config=sender_remote_config,
    )

    receiver_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=receiver_id,
        remote_config=receiver_remote_config,
    )

    # Insert block 1 into sender
    sender_store.insert(
        [b"remote_hash1"],
        [
            kv_cache_store.RaidenBlockID(
                sender_id, 1, kv_cache_store.BlockStatus.HOST
            )
        ],
        on_host=True,
    )

    # Manual D2H on sender managers
    sender_mgr_a.d2h([1], [1]).wait()
    sender_mgr_b.d2h([1], [1]).wait()

    # Simulate Global Registry Lookup Result and Prepare InsertAndPin
    allocated_blocks = [5]

    pin_hashes = [b"remote_hash1"]
    pin_slices = [
        kv_cache_store.RaidenBlockID(
            sender_id,
            allocated_blocks[0],
            kv_cache_store.BlockStatus.REMOTE,
        ),
    ]

    success, evicted = receiver_store.insert_and_pin(
        pin_hashes, pin_slices, True
    )
    self.assertTrue(success)
    self.assertEmpty(evicted)

    # Fetch Remote
    futures = receiver_store.fetch_remote(pin_hashes)
    self.assertEqual(len(futures), 1)

    # Wait for completion
    done = False
    for i in range(50):
      done_recving, failed_recving, pending_recving = (
          receiver_store.poll_fetch_remote_status()
      )
      print(
          f"Poll {i}: done_recving={done_recving},"
          f" failed_recving={failed_recving}, pending_recving={pending_recving}"
      )
      if failed_recving:
        self.fail(f"Transfer failed: {failed_recving}")
      if len(done_recving) == 1:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Receiver did not finish transfer in time")

    # Manual H2D on receiver managers
    receiver_mgr_a.h2d(allocated_blocks, allocated_blocks).wait()
    receiver_mgr_b.h2d(allocated_blocks, allocated_blocks).wait()

    # Verify Data
    rec_data_a = np.asarray(receiver_buf_a)
    np.testing.assert_array_equal(
        rec_data_a[allocated_blocks[0]], sender_data_a[1]
    )

    rec_data_b = np.asarray(receiver_buf_b)
    np.testing.assert_array_equal(
        rec_data_b[allocated_blocks[0]], sender_data_b[1]
    )

  def test_queue_flow_end_to_end(self):
    """Verifies complete flow: simulated Lookup, InsertAndPin, and Fetch."""
    self._require_tpu()
    sender_listener_port = portpicker.pick_unused_port()
    receiver_listener_port = portpicker.pick_unused_port()
    sender_controller_port = portpicker.pick_unused_port()
    receiver_controller_port = portpicker.pick_unused_port()

    num_blocks = 10
    num_layers = 1  # Match C++ test
    shape = (num_blocks, 128, 8, 8, 128)

    # Setup JAX arrays
    sender_kv_caches = []
    receiver_kv_caches = []

    for l in range(num_layers):
      # Fill sender with distinct data per block if needed, here just simple fill
      sender_data = np.zeros(shape, dtype=np.float32)
      sender_data[1] = 1.0  # Remote block 1
      sender_data[2] = 2.0  # Remote block 2
      sender_data[3] = 3.0  # Remote block 3
      sender_data[4] = 4.0  # Remote block 4
      sender_kv_caches.append(self._make_device_array(sender_data))

      receiver_kv_caches.append(
          self._make_device_array(np.zeros(shape, dtype=np.float32))
      )

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = self.num_devices
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    sender_mgr = KVCacheManager(
        kv_caches=sender_kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=sender_listener_port,
        listener_controller_port=sender_controller_port,
    )

    receiver_mgr = KVCacheManager(
        kv_caches=receiver_kv_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=receiver_listener_port,
        listener_controller_port=receiver_controller_port,
    )

    sender_id = kv_cache_store.RaidenId("sender_job", "0", "kv_cache", 0)
    receiver_id = kv_cache_store.RaidenId("receiver_job", "0", "data2", 0)

    sender_remote_config = kv_cache_store.RemoteFetchConfig()
    sender_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    sender_remote_config.controller_port = sender_controller_port
    sender_remote_config.local_worker_port = sender_listener_port
    sender_remote_config.bytes_per_block = local_bytes_per_block
    sender_remote_config.num_shards = self.num_devices

    receiver_remote_config = kv_cache_store.RemoteFetchConfig()
    receiver_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    receiver_remote_config.controller_port = receiver_controller_port
    receiver_remote_config.local_worker_port = receiver_listener_port
    receiver_remote_config.bytes_per_block = local_bytes_per_block
    receiver_remote_config.num_shards = self.num_devices

    sender_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=sender_id,
        remote_config=sender_remote_config,
    )

    receiver_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=receiver_id,
        remote_config=receiver_remote_config,
    )

    # Insert blocks into sender
    sender_store.insert(
        [b"remote_hash1", b"remote_hash2", b"remote_hash3", b"remote_hash4"],
        [
            kv_cache_store.RaidenBlockID(
                sender_id, 1, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                sender_id, 2, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                sender_id, 3, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                sender_id, 4, kv_cache_store.BlockStatus.HOST
            ),
        ],
        on_host=True,
    )

    receiver_store.insert(
        [b"local_hash1", b"local_hash2"],
        [
            kv_cache_store.RaidenBlockID(
                receiver_id, 7, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                receiver_id, 8, kv_cache_store.BlockStatus.HOST
            ),
        ],
        on_host=True,
    )

    # Manual D2H on sender to populate staging buffer
    sender_mgr.d2h([1, 2, 3, 4], [1, 2, 3, 4]).wait()

    # Simulate Global Registry Lookup Result and Prepare InsertAndPin
    # Allocated local blocks in receiver for remote data
    allocated_blocks = [5, 6, 7, 8]

    pin_hashes = [
        b"local_hash1",
        b"local_hash2",
        b"remote_hash1",
        b"remote_hash2",
        b"remote_hash3",
        b"remote_hash4",
    ]

    pin_slices = [
        kv_cache_store.RaidenBlockID(
            receiver_id, 7, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockID(
            receiver_id, 8, kv_cache_store.BlockStatus.HOST
        ),
        # Simulated global hits: REMOTE status with local allocated block ID
        kv_cache_store.RaidenBlockID(
            sender_id,
            allocated_blocks[0],
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            sender_id,
            allocated_blocks[1],
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            sender_id,
            allocated_blocks[2],
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            sender_id,
            allocated_blocks[3],
            kv_cache_store.BlockStatus.REMOTE,
        ),
    ]

    # Call InsertAndPin
    success, evicted = receiver_store.insert_and_pin(
        pin_hashes, pin_slices, True
    )
    self.assertTrue(success)
    self.assertEmpty(evicted)

    # Fetch Remote
    remote_hashes = [
        b"remote_hash1",
        b"remote_hash2",
        b"remote_hash3",
        b"remote_hash4",
    ]
    futures = receiver_store.fetch_remote(remote_hashes)
    self.assertEqual(len(futures), 4)

    # Wait for completion
    done = False
    for _ in range(50):
      done_recving, failed_recving, pending_recving = (
          receiver_store.poll_fetch_remote_status()
      )
      if failed_recving:
        self.fail(f"Transfer failed: {failed_recving}")
      if len(done_recving) == 4:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Receiver did not finish transfer in time")

    # Manual H2D on receiver
    receiver_mgr.h2d(allocated_blocks, allocated_blocks).wait()

    # Verify Data
    for l in range(num_layers):
      receiver_data = np.asarray(receiver_kv_caches[l])
      sender_data_np = np.asarray(sender_kv_caches[l])
      np.testing.assert_array_equal(
          receiver_data[allocated_blocks[0]], sender_data_np[1]
      )
      np.testing.assert_array_equal(
          receiver_data[allocated_blocks[1]], sender_data_np[2]
      )
      np.testing.assert_array_equal(
          receiver_data[allocated_blocks[2]], sender_data_np[3]
      )
      np.testing.assert_array_equal(
          receiver_data[allocated_blocks[3]], sender_data_np[4]
      )

    # Verify Status Upgrade
    for i, hash in enumerate(remote_hashes):
      results = receiver_store.lookup([hash])
      self.assertEqual(len(results), 1)
      block = results[0][1]
      self.assertEqual(block.status, kv_cache_store.BlockStatus.HOST)
      self.assertEqual(block.host_block_id, allocated_blocks[i])

  def test_queue_flow_multi_remote_end_to_end(self):
    """Verifies flow where blocks belong to multiple different remote RaidenIDs."""
    self._require_tpu()
    sender1_listener_port = portpicker.pick_unused_port()
    sender2_listener_port = portpicker.pick_unused_port()
    receiver_listener_port = portpicker.pick_unused_port()

    sender1_controller_port = portpicker.pick_unused_port()
    sender2_controller_port = portpicker.pick_unused_port()
    receiver_controller_port = portpicker.pick_unused_port()

    num_blocks = 10
    num_layers = 1
    shape = (num_blocks, 128, 8, 8, 128)

    # Sender 1 Data (ones)
    sender1_data = np.ones(shape, dtype=np.float32)
    sender1_buf = self._make_device_array(sender1_data)

    # Sender 2 Data (twos)
    sender2_data = np.full(shape, fill_value=2.0, dtype=np.float32)
    sender2_buf = self._make_device_array(sender2_data)

    # Receiver Buffers (zeros)
    receiver_buf = self._make_device_array(np.zeros(shape, dtype=np.float32))

    bytes_per_block = 128 * 8 * 8 * 128 * 4

    sender1_mgr = KVCacheManager(
        kv_caches=[sender1_buf],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=sender1_listener_port,
        listener_controller_port=sender1_controller_port,
    )

    sender2_mgr = KVCacheManager(
        kv_caches=[sender2_buf],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=sender2_listener_port,
        listener_controller_port=sender2_controller_port,
    )

    receiver_mgr = KVCacheManager(
        kv_caches=[receiver_buf],
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=receiver_listener_port,
        listener_controller_port=receiver_controller_port,
    )

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = self.num_devices
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    sender1_id = kv_cache_store.RaidenId("sender1_job", "0", "kv_cache", 0)
    sender2_id = kv_cache_store.RaidenId("sender2_job", "0", "kv_cache", 0)
    receiver_id = kv_cache_store.RaidenId("receiver_job", "0", "data2", 0)

    sender1_remote_config = kv_cache_store.RemoteFetchConfig()
    sender1_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    sender1_remote_config.controller_port = sender1_controller_port
    sender1_remote_config.local_worker_port = sender1_listener_port
    sender1_remote_config.bytes_per_block = local_bytes_per_block
    sender1_remote_config.num_shards = self.num_devices

    sender2_remote_config = kv_cache_store.RemoteFetchConfig()
    sender2_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    sender2_remote_config.controller_port = sender2_controller_port
    sender2_remote_config.local_worker_port = sender2_listener_port
    sender2_remote_config.bytes_per_block = local_bytes_per_block
    sender2_remote_config.num_shards = self.num_devices

    receiver_remote_config = kv_cache_store.RemoteFetchConfig()
    receiver_remote_config.orchestrator_address = (
        f"localhost:{_orchestrator_port}"
    )
    receiver_remote_config.controller_port = receiver_controller_port
    receiver_remote_config.local_worker_port = receiver_listener_port
    receiver_remote_config.bytes_per_block = local_bytes_per_block
    receiver_remote_config.num_shards = self.num_devices

    sender1_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=sender1_id,
        remote_config=sender1_remote_config,
    )

    sender2_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=sender2_id,
        remote_config=sender2_remote_config,
    )

    receiver_store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address="",
        raiden_id=receiver_id,
        remote_config=receiver_remote_config,
    )

    # Insert blocks into senders
    sender1_store.insert(
        [b"remote_hash1", b"remote_hash2"],
        [
            kv_cache_store.RaidenBlockID(
                sender1_id, 1, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                sender1_id, 2, kv_cache_store.BlockStatus.HOST
            ),
        ],
        on_host=True,
    )

    sender2_store.insert(
        [b"remote_hash3", b"remote_hash4"],
        [
            kv_cache_store.RaidenBlockID(
                sender2_id, 3, kv_cache_store.BlockStatus.HOST
            ),
            kv_cache_store.RaidenBlockID(
                sender2_id, 4, kv_cache_store.BlockStatus.HOST
            ),
        ],
        on_host=True,
    )

    # Manual D2H on senders
    sender1_mgr.d2h([1, 2], [1, 2]).wait()
    sender2_mgr.d2h([3, 4], [3, 4]).wait()

    # Simulate Global Registry Lookup Result and Prepare InsertAndPin
    allocated_blocks = [5, 6, 7, 8]

    pin_hashes = [
        b"remote_hash1",
        b"remote_hash2",
        b"remote_hash3",
        b"remote_hash4",
    ]

    pin_slices = [
        kv_cache_store.RaidenBlockID(
            sender1_id,
            allocated_blocks[0],
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            sender1_id,
            allocated_blocks[1],
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            sender2_id,
            allocated_blocks[2],
            kv_cache_store.BlockStatus.REMOTE,
        ),
        kv_cache_store.RaidenBlockID(
            sender2_id,
            allocated_blocks[3],
            kv_cache_store.BlockStatus.REMOTE,
        ),
    ]

    # Call InsertAndPin
    success, evicted = receiver_store.insert_and_pin(
        pin_hashes, pin_slices, True
    )
    self.assertTrue(success)
    self.assertEmpty(evicted)

    # Fetch Remote
    futures = receiver_store.fetch_remote(pin_hashes)
    self.assertEqual(len(futures), 4)

    # Wait for completion
    done = False
    for _ in range(50):
      done_recving, failed_recving, pending_recving = (
          receiver_store.poll_fetch_remote_status()
      )
      if failed_recving:
        self.fail(f"Transfer failed: {failed_recving}")
      if len(done_recving) == 4:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Receiver did not finish transfer in time")

    # Manual H2D on receiver
    receiver_mgr.h2d(allocated_blocks, allocated_blocks).wait()

    # Verify Data
    rec_data = np.asarray(receiver_buf)

    # Blocks 5, 6 should come from Sender 1 (ones)
    np.testing.assert_array_equal(
        rec_data[allocated_blocks[0]], sender1_data[1]
    )
    np.testing.assert_array_equal(
        rec_data[allocated_blocks[1]], sender1_data[2]
    )

    # Blocks 7, 8 should come from Sender 2 (twos)
    np.testing.assert_array_equal(
        rec_data[allocated_blocks[2]], sender2_data[3]
    )
    np.testing.assert_array_equal(
        rec_data[allocated_blocks[3]], sender2_data[4]
    )

  def test_e2e_load(self):
    """Tests end-to-end load (H2D) on CPU."""
    self._require_tpu()
    # 1. Setup Ports
    listener_port = portpicker.pick_unused_port()
    controller_port = portpicker.pick_unused_port()

    # 2. Setup JAX arrays (TPU)
    num_blocks = 10
    num_layers = 10
    shape = (num_blocks, 128, 8, 8, 128)
    devices = self.devices
    sharding = self._make_sharding(devices)

    device_caches = []
    for l in range(num_layers):
      data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
          l * 1000000.0
      )
      device_caches.append(self._make_device_array(data, sharding))

    # 3. Setup Manager
    manager = KVCacheManager(
        kv_caches=device_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        listener_port=listener_port,
        listener_controller_port=controller_port,
    )
    self.assertTrue(manager.is_listener_active)

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = len(devices)
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # 4. Setup Store
    config = kv_cache_store.RemoteFetchConfig()
    config.orchestrator_address = f"localhost:{_orchestrator_port}"
    config.controller_port = controller_port
    config.local_worker_port = listener_port
    config.bytes_per_block = local_bytes_per_block
    config.num_shards = len(devices)
    config.num_listeners = 1

    store_id = kv_cache_store.RaidenId("store_job", "0", "kv_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=store_id,
        remote_config=config,
    )

    time.sleep(1)

    # 5. Populate DRAM (HOST blocks 3 and 4) using D2H
    manager.d2h([0, 1], [3, 4]).wait()

    # Insert blocks to store directory as HOST status
    slices_1 = [
        kv_cache_store.RaidenBlockID(
            store_id, 3, kv_cache_store.BlockStatus.HOST
        )
    ]
    self.assertTrue(store.insert([b"hash1"], slices_1, True)[0])

    slices_2 = [
        kv_cache_store.RaidenBlockID(
            store_id, 4, kv_cache_store.BlockStatus.HOST
        )
    ]
    self.assertTrue(store.insert([b"hash2"], slices_2, True)[0])

    # 6. Trigger Load to HBM block 5 and 6
    futures = store.load([b"hash1", b"hash2"], [5, 6])
    self.assertEqual(len(futures), 2)

    # Wait for completion
    done = False
    for _ in range(50):
      done_loading, failed_loading, pending_loading = store.poll_load_status()
      if failed_loading:
        self.fail(f"Load failed: {failed_loading}")
      if len(done_loading) == 2:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Load did not finish in time")
    self.assertTrue(futures[b"hash1"].IsDone())
    self.assertTrue(futures[b"hash2"].IsDone())

    # 7. Verify Data on Device
    for l in range(num_layers):
      device_data = np.asarray(device_caches[l])
      np.testing.assert_array_equal(device_data[5], device_data[0])
      np.testing.assert_array_equal(device_data[6], device_data[1])

    # Verify LRU Status Upgrade in Store
    lookup_res1 = store.lookup([b"hash1"])
    self.assertEqual(len(lookup_res1), 1)
    self.assertEqual(
        lookup_res1[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res1[0][1].host_block_id, 3)
    self.assertEqual(lookup_res1[0][1].hbm_block_id, 5)

    lookup_res2 = store.lookup([b"hash2"])
    self.assertEqual(len(lookup_res2), 1)
    self.assertEqual(
        lookup_res2[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res2[0][1].host_block_id, 4)
    self.assertEqual(lookup_res2[0][1].hbm_block_id, 6)

  def test_e2e_save(self):
    """Tests end-to-end save (D2H) and load (H2D) back on TPU."""
    self._require_tpu()
    # 1. Setup Ports
    listener_port = portpicker.pick_unused_port()
    controller_port = portpicker.pick_unused_port()

    # 2. Setup JAX arrays (TPU)
    num_blocks = 10
    num_layers = 10
    shape = (num_blocks, 128, 8, 8, 128)
    devices = self.devices
    sharding = self._make_sharding(devices)

    device_caches = []
    for l in range(num_layers):
      data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
          l * 1000000.0
      )
      device_caches.append(self._make_device_array(data, sharding))

    # 3. Setup Manager
    num_slots = 2
    manager = KVCacheManager(
        kv_caches=device_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=num_slots,
        listener_port=listener_port,
        listener_controller_port=controller_port,
        host_blocks_to_allocate=num_slots * num_blocks
        + 2,  # Allocate 2 extra blocks for Save
    )
    self.assertTrue(manager.is_listener_active)

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = len(devices)
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # 4. Setup Store
    config = kv_cache_store.RemoteFetchConfig()
    config.orchestrator_address = f"localhost:{_orchestrator_port}"
    config.controller_port = controller_port
    config.local_worker_port = listener_port
    config.bytes_per_block = local_bytes_per_block
    config.num_shards = len(devices)
    config.num_listeners = 1

    store_id = kv_cache_store.RaidenId("store_job", "0", "kv_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=store_id,
        remote_config=config,
    )

    time.sleep(1)

    # 5. Insert blocks to store directory as HBM status and PIN them
    slices_1 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=-1,
            hbm_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(store.insert([b"hash1"], slices_1, False)[0])
    self.assertTrue(store.pin([b"hash1"]))

    slices_2 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=-1,
            hbm_block_id=1,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(store.insert([b"hash2"], slices_2, False)[0])
    self.assertTrue(store.pin([b"hash2"]))

    # 6. Trigger Save (D2H)
    save_futures = store.save([b"hash1", b"hash2"], [0, 1])
    self.assertEqual(len(save_futures), 2)

    # Wait for save completion
    done = False
    for _ in range(50):
      done_saving, failed_saving, pending_saving = store.poll_save_status()
      if failed_saving:
        self.fail(f"Save failed: {failed_saving}")
      if len(done_saving) == 2:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Save did not finish in time")
    self.assertTrue(save_futures[b"hash1"].IsDone())
    self.assertTrue(save_futures[b"hash2"].IsDone())

    # Verify status in LRU is HOST_AND_HBM, and host_block_id is allocated
    lookup_res1 = store.lookup([b"hash1"])
    self.assertEqual(len(lookup_res1), 1)
    self.assertEqual(
        lookup_res1[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    host_block_id1 = lookup_res1[0][1].host_block_id
    self.assertGreaterEqual(host_block_id1, 0)

    lookup_res2 = store.lookup([b"hash2"])
    self.assertEqual(len(lookup_res2), 1)
    self.assertEqual(
        lookup_res2[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    host_block_id2 = lookup_res2[0][1].host_block_id
    self.assertGreaterEqual(host_block_id2, 0)
    self.assertNotEqual(host_block_id1, host_block_id2)

    # Verify registration in global registry using a second store instance (forcing global lookup)
    store2 = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=kv_cache_store.RaidenId("receiver_job", "0", "kv_cache", 0),
    )
    global_verified = False
    for _ in range(50):
      lookup_res = store2.lookup([b"hash1", b"hash2"], enable_global=True)
      if len(lookup_res) == 2:
        self.assertEqual(lookup_res[0][0], b"hash1")
        self.assertEqual(
            lookup_res[0][1].status, kv_cache_store.BlockStatus.REMOTE
        )
        self.assertEqual(lookup_res[0][1].raiden_id, store_id)

        self.assertEqual(lookup_res[1][0], b"hash2")
        self.assertEqual(
            lookup_res[1][1].status, kv_cache_store.BlockStatus.REMOTE
        )
        self.assertEqual(lookup_res[1][1].raiden_id, store_id)
        global_verified = True
        break
      time.sleep(0.1)
    self.assertTrue(global_verified, "Global registration lookup failed")

    # 7. Trigger Load (H2D) to DIFFERENT HBM blocks (5 and 6) to verify DRAM content
    load_futures = store.load([b"hash1", b"hash2"], [5, 6])
    self.assertEqual(len(load_futures), 2)

    # Wait for load completion
    done = False
    for _ in range(50):
      done_loading, failed_loading, pending_loading = store.poll_load_status()
      if failed_loading:
        self.fail(f"Load failed: {failed_loading}")
      if len(done_loading) == 2:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Load did not finish in time")

    # 8. Verify Data on Device
    for l in range(num_layers):
      device_data = np.asarray(device_caches[l])
      np.testing.assert_array_equal(device_data[5], device_data[0])
      np.testing.assert_array_equal(device_data[6], device_data[1])

  def test_e2e_evict(self):
    """Tests end-to-end evict (unregistered from global and unlocked from host)."""
    self._require_tpu()
    # 1. Setup Ports
    listener_port = portpicker.pick_unused_port()
    controller_port = portpicker.pick_unused_port()

    # 2. Setup JAX arrays (TPU)
    num_blocks = 10
    num_layers = 2
    shape = (num_blocks, 128, 8, 8, 128)
    devices = self.devices
    sharding = self._make_sharding(devices)

    device_caches = []
    for l in range(num_layers):
      data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
          l * 1000000.0
      )
      device_caches.append(self._make_device_array(data, sharding))

    # 3. Setup Manager
    num_slots = 2
    manager = KVCacheManager(
        kv_caches=device_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=num_slots,
        listener_port=listener_port,
        listener_controller_port=controller_port,
        host_blocks_to_allocate=num_slots * num_blocks + 2,
    )
    self.assertTrue(manager.is_listener_active)

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = len(devices)
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # 4. Setup Store
    config = kv_cache_store.RemoteFetchConfig()
    config.orchestrator_address = f"localhost:{_orchestrator_port}"
    config.controller_port = controller_port
    config.local_worker_port = listener_port
    config.bytes_per_block = local_bytes_per_block
    config.num_shards = len(devices)
    config.num_listeners = 1

    store_id = kv_cache_store.RaidenId("store_job", "0", "kv_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=store_id,
        remote_config=config,
    )

    time.sleep(1)

    # 5. Insert hash1 as HOST (local only)
    slices_1 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=0,
            hbm_block_id=-1,
            status=kv_cache_store.BlockStatus.HOST,
        )
    ]
    self.assertTrue(store.insert([b"hash1"], slices_1, True)[0])

    # 6. Insert hash2 as HBM and SAVE it to make it HOST_AND_HBM (registered)
    slices_2 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=-1,
            hbm_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(store.insert([b"hash2"], slices_2, False)[0])
    self.assertTrue(store.pin([b"hash2"]))

    save_futures = store.save([b"hash2"], [0])
    self.assertEqual(len(save_futures), 1)

    done = False
    for _ in range(50):
      done_saving, failed_saving, pending_saving = store.poll_save_status()
      if failed_saving:
        self.fail(f"Save failed: {failed_saving}")
      if len(done_saving) == 1:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Save did not finish in time")
    store.release([b"hash2"])  # Release pin after save

    # Verify hash2 is now HOST_AND_HBM and has host_block_id allocated
    lookup_before = store.lookup([b"hash2"])
    self.assertEqual(len(lookup_before), 1)
    self.assertEqual(
        lookup_before[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    host_block_id_hash2 = lookup_before[0][1].host_block_id
    self.assertGreaterEqual(host_block_id_hash2, 0)

    # Verify hash2 is registered in global registry
    store2 = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=kv_cache_store.RaidenId("receiver_job", "0", "kv_cache", 0),
    )
    global_verified = False
    for _ in range(50):
      lookup_res = store2.lookup([b"hash2"], enable_global=True)
      if len(lookup_res) == 1:
        self.assertEqual(lookup_res[0][0], b"hash2")
        self.assertEqual(
            lookup_res[0][1].status, kv_cache_store.BlockStatus.REMOTE
        )
        global_verified = True
        break
      time.sleep(0.1)
    self.assertTrue(
        global_verified, "Global registration lookup failed before evict"
    )

    # 7. Trigger Evict for both hash1 and hash2
    evict_futures = store.evict([b"hash1", b"hash2"])
    self.assertEqual(len(evict_futures), 2)

    # Verify status changed IMMEDIATELY (HOST -> INIT, HOST_AND_HBM -> HBM)
    lookup_imm = store.lookup([b"hash1", b"hash2"])
    for h, block in lookup_imm:
      if h == b"hash1":
        self.assertEqual(block.status, kv_cache_store.BlockStatus.INIT)
      elif h == b"hash2":
        self.assertEqual(block.status, kv_cache_store.BlockStatus.HBM)

    # 8. Wait for evict completion
    done = False
    for _ in range(50):
      done_evict, failed_evict, pending_evict = store.poll_evict_status()
      if failed_evict:
        self.fail(f"Evict failed: {failed_evict}")
      if len(done_evict) == 2:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Evict did not finish in time")
    self.assertTrue(evict_futures[b"hash1"].IsDone())
    self.assertTrue(evict_futures[b"hash2"].IsDone())

    # 9. Verify final status in store
    # hash1 (was HOST -> INIT) should be ERASED from LRU
    self.assertEmpty(store.lookup([b"hash1"]))

    # hash2 (was HOST_AND_HBM -> HBM) should remain in LRU as HBM with host_block_id = -1
    lookup_after = store.lookup([b"hash2"])
    self.assertEqual(len(lookup_after), 1)
    self.assertEqual(lookup_after[0][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_after[0][1].host_block_id, -1)

    # 10. Verify unregistration in global registry (hash2 should not be found anymore)
    global_unregistered = False
    for _ in range(50):
      lookup_res = store2.lookup([b"hash2"], enable_global=True)
      if len(lookup_res) == 0:
        global_unregistered = True
        break
      time.sleep(0.1)
    self.assertTrue(global_unregistered, "Global unregistration failed")


if __name__ == "__main__":
  absltest.main()
