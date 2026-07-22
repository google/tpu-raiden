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

if not os.path.exists("/dev/accel0"):
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# pylint: disable=g-import-not-at-top
import socket
import subprocess
import time
import unittest

from absl.testing import absltest
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import numpy as np

resources = None
from tpu_raiden.api.jax import kv_cache_store
from tpu_raiden.api.jax.kv_cache_manager import KVCacheManager

# pylint: enable=g-import-not-at-top


def _pick_unused_port():
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(("localhost", 0))
  port = s.getsockname()[1]
  s.close()
  return port


# Global variables for subprocesses
_orchestrator_process = None
_registry_process = None
_orchestrator_port = None
_registry_port = None


def setUpModule():
  global _orchestrator_process, _registry_process
  global _orchestrator_port, _registry_port

  _orchestrator_port = _pick_unused_port()
  _registry_port = _pick_unused_port()

  this_dir = os.path.dirname(os.path.abspath(__file__))
  orchestrator_binary = os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "core",
          "controller",
          "raiden_orchestrator_main",
      )
  )
  registry_binary = os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "kv_cache",
          "global_registry",
          "global_registry_server",
      )
  )
  extra_flags = []

  print(f"Starting Orchestrator on port {_orchestrator_port}")
  orch_log = open("/tmp/raiden_orchestrator.log", "w")
  _orchestrator_process = subprocess.Popen(
      [
          orchestrator_binary,
          f"--port={_orchestrator_port}",
      ]
      + extra_flags,
      stdout=orch_log,
      stderr=subprocess.STDOUT,
  )

  print(f"Starting Registry on port {_registry_port}")
  reg_log = open("/tmp/raiden_registry.log", "w")
  _registry_process = subprocess.Popen(
      [
          registry_binary,
          f"--port={_registry_port}",
      ]
      + extra_flags,
      stdout=reg_log,
      stderr=subprocess.STDOUT,
  )

  # Give them some time to start
  time.sleep(2)


def tearDownModule():
  if _orchestrator_process:
    code = _orchestrator_process.poll()
    if code is not None and code != 0:
      print(f"--- Orchestrator exited with {code} ---")
      try:
        with open("/tmp/raiden_orchestrator.log", "r") as f:
          print(f.read())
      except OSError as e:
        print(f"Failed to read orchestrator log: {e}")
    _orchestrator_process.terminate()
    _orchestrator_process.wait()
  if _registry_process:
    code = _registry_process.poll()
    if code is not None and code != 0:
      print(f"--- Registry exited with {code} ---")
      try:
        with open("/tmp/raiden_registry.log", "r") as f:
          print(f.read())
      except OSError as e:
        print(f"Failed to read registry log: {e}")
    _registry_process.terminate()
    _registry_process.wait()


class KVCacheStoreTest(absltest.TestCase):

  def _make_sharding(self, devices):

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
    except RuntimeError:
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
        kv_cache_store._impl.RaidenId("job1", "0", "kv_cache", 0),
        42,
        kv_cache_store._impl.BlockStatus.REMOTE,
    )
    remote_id2 = kv_cache_store._impl.RaidenBlockID(
        kv_cache_store._impl.RaidenId("job2", "0", "kv_cache", 0),
        43,
        kv_cache_store._impl.BlockStatus.REMOTE,
    )
    mock_impl.lookup.return_value = [
        (b"global_1", remote_id1),
        (b"global_2", remote_id2),
    ]

    res = controller.lookup([b"global_1", b"global_2"], enable_global=True)
    self.assertLen(res, 2)
    self.assertEqual(res[0][0], b"global_1")
    self.assertEqual(res[0][1].raiden_id.job_name, "job1")
    self.assertEqual(res[0][1].host_block_id, 42)
    self.assertEqual(res[0][1].status, kv_cache_store.BlockStatus.REMOTE)
    self.assertEqual(res[1][0], b"global_2")
    self.assertEqual(res[1][1].raiden_id.job_name, "job2")
    self.assertEqual(res[1][1].host_block_id, 43)
    self.assertEqual(res[1][1].status, kv_cache_store.BlockStatus.REMOTE)
    mock_impl.lookup.assert_called_with([b"global_1", b"global_2"], True)

  def test_global_lookup_error_ignored(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=20, global_registry_address="invalid.address:12345"
    )
    hashes = [b"9001"]
    # Should not fail, just return empty because the registry is down
    res = controller.lookup(hashes, enable_global=True)
    self.assertEmpty(res)

  def test_save_and_load_mocked(self):
    controller = kv_cache_store.KVCacheStore(capacity=20)
    mock_impl = unittest.mock.MagicMock()
    controller._impl = mock_impl

    hashes = [b"hash_1", b"hash_2"]
    mock_impl.save.return_value = True
    self.assertTrue(controller.save(hashes))
    mock_impl.save.assert_called_with(hashes)
    mock_impl.save.return_value = False
    self.assertFalse(controller.save(hashes))

    device_block_ids = [2, 3]
    mock_impl.load.return_value = True
    self.assertTrue(controller.load(hashes, device_block_ids))
    mock_impl.load.assert_called_with(hashes, device_block_ids)
    mock_impl.load.return_value = False
    self.assertFalse(controller.load(hashes, device_block_ids))

    mock_impl.poll_save_status.return_value = ([b"hash_1"], [], [b"hash_2"])
    self.assertEqual(
        controller.poll_save_status(), ([b"hash_1"], [], [b"hash_2"])
    )
    mock_impl.poll_save_status.assert_called_once()

    mock_impl.poll_load_status.return_value = ([], [b"hash_1"], [b"hash_2"])
    self.assertEqual(
        controller.poll_load_status(), ([], [b"hash_1"], [b"hash_2"])
    )
    mock_impl.poll_load_status.assert_called_once()

  def test_insert_and_lock_release_and_delete(self):
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
    success = controller.insert_and_lock(remote_hashes, remote_slices, True)
    self.assertTrue(success)
    self.assertEmpty(controller.lookup([b"local_1"]))

    del_count = controller.release_and_delete(remote_hashes)
    self.assertEqual(del_count, 2)
    self.assertLen(controller.lookup([b"local_1", b"local_2"]), 2)

  def test_e2e_load(self):
    """Tests end-to-end load (H2D) on CPU."""
    self._require_tpu()
    # 1. Setup Ports
    listener_port = _pick_unused_port()

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

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = len(devices)
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # 3. Setup Store
    store_id = kv_cache_store.RaidenId("store_job", "0", "kv_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=store_id,
        num_shards=len(devices),
        shard_size_bytes=local_bytes_per_block,
        raiden_orchestrator_address=f"localhost:{_orchestrator_port}",
    )

    # 4. Setup Manager
    manager = KVCacheManager(
        kv_caches=device_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        raiden_worker_port=listener_port,
        raiden_controller_address=store.raiden_controller_address,
    )
    self.assertTrue(manager.is_listener_active)

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

    self.assertTrue(store.pin([b"hash1", b"hash2"]))

    # 6. Trigger Load to HBM block 5 and 6
    self.assertTrue(store.load([b"hash1", b"hash2"], [5, 6]))

    # Wait for completion
    done = False
    for _ in range(50):
      done_loading, failed_loading, _ = store.poll_load_status()
      if failed_loading:
        self.fail(f"Load failed: {failed_loading}")
      if len(done_loading) == 2:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Load did not finish in time")

    # 7. Verify Data on Device
    for l in range(num_layers):
      device_data = np.asarray(device_caches[l])
      np.testing.assert_array_equal(device_data[5], device_data[0])
      np.testing.assert_array_equal(device_data[6], device_data[1])

    # Verify LRU Status Upgrade in Store
    lookup_res1 = store.lookup([b"hash1"])
    self.assertLen(lookup_res1, 1)
    self.assertEqual(
        lookup_res1[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res1[0][1].host_block_id, 3)
    self.assertEqual(lookup_res1[0][1].device_block_id, 5)

    lookup_res2 = store.lookup([b"hash2"])
    self.assertLen(lookup_res2, 1)
    self.assertEqual(
        lookup_res2[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res2[0][1].host_block_id, 4)
    self.assertEqual(lookup_res2[0][1].device_block_id, 6)

  def test_e2e_save(self):
    """Tests end-to-end save (D2H) and load (H2D) back on TPU."""
    self._require_tpu()
    # 1. Setup Ports
    listener_port = _pick_unused_port()

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

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_shards_per_listener = len(devices)
    local_bytes_per_block = bytes_per_block // local_shards_per_listener

    # 3. Setup Store
    store_id = kv_cache_store.RaidenId("store_job", "0", "kv_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=store_id,
        num_shards=len(devices),
        shard_size_bytes=local_bytes_per_block,
        raiden_orchestrator_address=f"localhost:{_orchestrator_port}",
    )

    # 4. Setup Manager
    num_slots = 2
    manager = KVCacheManager(
        kv_caches=device_caches,
        node_id=0,
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=num_slots,
        raiden_worker_port=listener_port,
        raiden_controller_address=store.raiden_controller_address,
        host_blocks_to_allocate=num_slots * num_blocks
        + 2,  # Allocate 2 extra blocks for Save
    )
    self.assertTrue(manager.is_listener_active)

    time.sleep(1)

    # 5. Insert blocks to store directory as HBM status and PIN them
    slices_1 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=-1,
            device_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(store.insert([b"hash1"], slices_1, False)[0])
    self.assertTrue(store.pin([b"hash1"]))

    slices_2 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=-1,
            device_block_id=1,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(store.insert([b"hash2"], slices_2, False)[0])
    self.assertTrue(store.pin([b"hash2"]))

    # 6. Trigger Save (D2H)
    self.assertTrue(store.save([b"hash1", b"hash2"]))

    # Wait for save completion
    done = False
    for _ in range(50):
      done_saving, failed_saving, _ = store.poll_save_status()
      if failed_saving:
        self.fail(f"Save failed: {failed_saving}")
      if len(done_saving) == 2:
        done = True
        break
      time.sleep(0.1)
    self.assertTrue(done, "Save did not finish in time")

    # Verify status in LRU is HOST_AND_HBM, and host_block_id is allocated
    lookup_res1 = store.lookup([b"hash1"])
    self.assertLen(lookup_res1, 1)
    self.assertEqual(
        lookup_res1[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    host_block_id1 = lookup_res1[0][1].host_block_id
    self.assertGreaterEqual(host_block_id1, 0)

    lookup_res2 = store.lookup([b"hash2"])
    self.assertLen(lookup_res2, 1)
    self.assertEqual(
        lookup_res2[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    host_block_id2 = lookup_res2[0][1].host_block_id
    self.assertGreaterEqual(host_block_id2, 0)
    self.assertNotEqual(host_block_id1, host_block_id2)

    # Verify registration in global registry using a second store instance
    # (forcing global lookup)
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

    # 7. Trigger Load (H2D) to DIFFERENT HBM blocks (5 and 6) to verify DRAM
    # content
    self.assertTrue(store.load([b"hash1", b"hash2"], [5, 6]))

    # Wait for load completion
    done = False
    for _ in range(50):
      done_loading, failed_loading, _ = store.poll_load_status()
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


if __name__ == "__main__":
  absltest.main()
