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
from tpu_sync.api.jax import kv_cache_store
from tpu_sync.api.jax.kv_cache_manager import KVCacheManager

# pylint: enable=g-import-not-at-top


def _pick_unused_port():
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(("localhost", 0))
  port = s.getsockname()[1]
  s.close()
  return port


# Global variables for subprocesses
_registry_process = None
_registry_port = None


def _registry_binary_path():
  this_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "kv_cache",
          "global_registry",
          "global_registry_server",
      )
  )
  this_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(
          this_dir,
          "..",
          "..",
          "kv_cache",
          "global_registry",
          "global_registry_server",
      )
  )


def setUpModule():
  global _registry_process
  global _registry_port
  _registry_port = _pick_unused_port()

  this_dir = os.path.dirname(os.path.abspath(__file__))
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
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
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
    controller.release(hashes_with_miss)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], b"6001")
    self.assertEqual(lookup_res[0][1].raiden_id.job_name, "inference_server")
    self.assertEqual(lookup_res[0][1].raiden_id.job_replica_id, "0")

    # Lookup with an early miss
    hashes_early_miss = [b"6001", b"6003", b"6002"]
    lookup_res_early = controller.lookup(hashes_early_miss)
    controller.release(hashes_early_miss)
    self.assertLen(lookup_res_early, 1)
    self.assertEqual(lookup_res_early[0][0], b"6001")

    # 3. Delete
    controller.release(hashes)
    controller.delete(hashes, slices)
    self.assertTrue(
        controller.insert(hashes, slices, True)[0]
    )  # Successful again

  def test_pin_and_release(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=2, num_shards=1, store_server_ip="127.0.0.1"
    )

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

    res_old = controller.lookup([b"7001", b"7002"])
    self.assertLen(res_old, 2)
    controller.release([b"7001", b"7002"])
    res = controller.lookup([b"7002"])
    self.assertLen(res, 1)
    controller.release([b"7002"])

  def test_partial_pin_rollback(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=2, num_shards=1, store_server_ip="127.0.0.1"
    )

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

    res_old = controller.lookup([b"8001", b"8002"])
    self.assertLen(res_old, 2)
    controller.release([b"8001", b"8002"])
    res = controller.lookup([b"8004", b"8005"])
    self.assertLen(res, 2)
    controller.release([b"8004", b"8005"])

  def test_large_and_arbitrary_length_hashes(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=5, num_shards=1, store_server_ip="127.0.0.1"
    )

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
    controller.release(hashes)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][0], large_hash)
    self.assertEqual(lookup_res[1][0], long_hash)

  def test_global_lookup_case1_local_hit(self):
    # Case 1: Full local hit, no global hit.
    # We don't need a registry server for this because it shouldn't be queried.
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    hashes = [b"local_only"]
    slices = [
        kv_cache_store.RaidenId("local_job", "0", "kv_cache", 0),
    ]
    self.assertTrue(controller.insert(hashes, slices, True)[0])

    res = controller.lookup(hashes, enable_global=True)
    controller.release(hashes)
    self.assertLen(res, 1)
    self.assertEqual(res[0][0], b"local_only")
    self.assertEqual(res[0][1].raiden_id.job_name, "local_job")
    self.assertEqual(res[0][1].raiden_id.data_replica_idx, 0)

  def test_global_lookup_case2_and_3_mocked(self):
    # We mock _impl to simulate Case 2 and Case 3 because we don't
    # have a running registry server in Python tests.
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )

    # Create a mock for the C++ impl
    mock_impl = unittest.mock.MagicMock()
    controller._impl = mock_impl

    # Case 2: Both local and global have the same hit, but we return local.
    local_id = kv_cache_store._impl.RaidenBlockID(
        kv_cache_store._impl.RaidenId("local_job", "0", "kv_cache", 1)
    )
    mock_impl.lookup.return_value = [(b"shared_hash", local_id)]

    res = controller.lookup([b"shared_hash"], enable_global=True)
    controller.release([b"shared_hash"])
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
    controller.release([b"global_1", b"global_2"])
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
    # Construction requires a registry that is actually reachable
    # (RegisterStore failure now fails construction), so an unreachable address
    # can no longer model "registry down" at
    # construction time. Instead: start a real, throwaway registry, construct
    # against it, then kill it before the Lookup this test cares about --
    # genuinely down for that RPC, not mocked.
    throwaway_port = _pick_unused_port()
    throwaway_registry = subprocess.Popen(
        [_registry_binary_path(), f"--port={throwaway_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    try:
      controller = kv_cache_store.KVCacheStore(
          capacity=20,
          global_registry_address=f"localhost:{throwaway_port}",
          raiden_id=kv_cache_store.RaidenId(
              "lookup_error_job", "0", "kv_cache", 0
          ),
          num_shards=1,
          store_server_ip="127.0.0.1",
      )
    finally:
      throwaway_registry.terminate()
      throwaway_registry.wait()

    hashes = [b"9001"]
    # Should not fail, just return empty because the registry is now down.
    res = controller.lookup(hashes, enable_global=True)
    controller.release(hashes)
    self.assertEmpty(res)

  def test_save_and_load_mocked(self):
    controller = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
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
    controller = kv_cache_store.KVCacheStore(
        capacity=2, num_shards=1, store_server_ip="127.0.0.1"
    )

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
    res_local = controller.lookup([b"local_1"])
    self.assertLen(res_local, 1)
    controller.release([b"local_1"])

    del_count = controller.release_and_delete(remote_hashes)
    self.assertEqual(del_count, 2)
    res = controller.lookup([b"local_1", b"local_2"])
    self.assertLen(res, 2)
    controller.release([b"local_1", b"local_2"])

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
        store_server_ip="localhost",
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
    store.release([b"hash1"])
    self.assertLen(lookup_res1, 1)
    self.assertEqual(
        lookup_res1[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res1[0][1].host_block_id, 3)
    self.assertEqual(lookup_res1[0][1].device_block_id, 5)

    lookup_res2 = store.lookup([b"hash2"])
    store.release([b"hash2"])
    self.assertLen(lookup_res2, 1)
    self.assertEqual(
        lookup_res2[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res2[0][1].host_block_id, 4)
    self.assertEqual(lookup_res2[0][1].device_block_id, 6)

  def test_e2e_load_with_slices(self):
    """Tests end-to-end load (H2D) driven by pre-resolved slices, on TPU.

    Same transfer as test_e2e_load, but the caller hands the store the index
    entries it already looked up instead of letting the store resolve the
    hashes again. The bytes that land must be identical either way -- `slices`
    is a shortcut past the lookup, not a different transfer.
    """
    self._require_tpu()
    listener_port = _pick_unused_port()

    num_blocks = 10
    num_layers = 10
    shape = (num_blocks, 128, 8, 8, 128)
    devices = self.devices
    sharding = self._make_sharding(devices)

    device_caches = []
    for l in range(num_layers):
      data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + (
          l * 7919.0
      )
      device_caches.append(self._make_device_array(data, sharding))

    bytes_per_block = 128 * 8 * 8 * 128 * 4
    local_bytes_per_block = bytes_per_block // len(devices)

    store_id = kv_cache_store.RaidenId("slices_job", "0", "kv_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=10,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=store_id,
        num_shards=len(devices),
        shard_size_bytes=local_bytes_per_block,
        store_server_ip="localhost",
    )

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

    # Stage device blocks 0 and 1 into host blocks 3 and 4.
    manager.d2h([0, 1], [3, 4]).wait()

    hashes = [b"slice_hash1", b"slice_hash2"]
    self.assertTrue(
        store.insert(
            [hashes[0]],
            [
                kv_cache_store.RaidenBlockID(
                    store_id, 3, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )[0]
    )
    self.assertTrue(
        store.insert(
            [hashes[1]],
            [
                kv_cache_store.RaidenBlockID(
                    store_id, 4, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )[0]
    )

    # Pin BEFORE resolving, so the entries cannot be evicted out from under
    # the slices between the lookup and the load. This is the ordering the
    # slices form depends on.
    self.assertTrue(store.pin(hashes))

    looked_up = store.lookup(hashes)
    self.assertLen(looked_up, 2)
    slices = [entry for _, entry in looked_up]
    self.assertEqual(slices[0].host_block_id, 3)
    self.assertEqual(slices[1].host_block_id, 4)

    self.assertTrue(store.load(hashes, [5, 6], slices=slices))

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

    # Byte-exact, every layer: block 5 must equal block 0 and 6 equal 1.
    for l in range(num_layers):
      device_data = np.asarray(device_caches[l])
      np.testing.assert_array_equal(device_data[5], device_data[0])
      np.testing.assert_array_equal(device_data[6], device_data[1])

    for hash_val, expected_host, expected_device in (
        (hashes[0], 3, 5),
        (hashes[1], 4, 6),
    ):
      res = store.lookup([hash_val])
      store.release([hash_val])
      self.assertLen(res, 1)
      self.assertEqual(
          res[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
      )
      self.assertEqual(res[0][1].host_block_id, expected_host)
      self.assertEqual(res[0][1].device_block_id, expected_device)


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
        store_server_ip="localhost",
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

    slices_2 = [
        kv_cache_store.RaidenBlockID(
            store_id,
            host_block_id=-1,
            device_block_id=1,
            status=kv_cache_store.BlockStatus.HBM,
        )
    ]
    self.assertTrue(store.insert([b"hash2"], slices_2, False)[0])

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
    store.release([b"hash1"])
    self.assertLen(lookup_res1, 1)
    self.assertEqual(
        lookup_res1[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    host_block_id1 = lookup_res1[0][1].host_block_id
    self.assertGreaterEqual(host_block_id1, 0)

    lookup_res2 = store.lookup([b"hash2"])
    store.release([b"hash2"])
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
        num_shards=1,
        store_server_ip="localhost",
    )
    global_verified = False
    for _ in range(50):
      lookup_res = store2.lookup([b"hash1", b"hash2"], enable_global=True)
      store2.release([b"hash1", b"hash2"])
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

  def _make_registry_store(self, job_name, capacity=20):
    return kv_cache_store.KVCacheStore(
        capacity=capacity,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=kv_cache_store.RaidenId(job_name, "0", "kv_cache", 0),
        num_shards=1,
        store_server_ip="127.0.0.1",
    )

  def test_write_remote_requires_a_registry(self):
    # No global registry address: nothing to resolve the destination through,
    # and nothing to make the blocks reachable once they landed.
    store = kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )
    src_id = kv_cache_store.RaidenId("wr_noreg", "0", "kv_cache", 0)
    self.assertTrue(
        store.insert_and_lock(
            [b"a"],
            [
                kv_cache_store.RaidenBlockID(
                    src_id, 0, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )
    self.assertFalse(
        store.write_remote(
            [b"a"], kv_cache_store.RaidenId("wr_dst", "0", "kv_cache", 0)
        )
    )

  def test_write_remote_all_exist_settles_done(self):
    src = self._make_registry_store("wr_src_allexist")
    dst = self._make_registry_store("wr_dst_allexist")
    dst_id = dst.raiden_id

    hashes = [b"wr_a", b"wr_b"]
    src_slices = [
        kv_cache_store.RaidenBlockID(
            src.raiden_id, i, kv_cache_store.BlockStatus.HOST
        )
        for i in range(len(hashes))
    ]
    dst_slices = [
        kv_cache_store.RaidenBlockID(
            dst_id, 5 + i, kv_cache_store.BlockStatus.HOST
        )
        for i in range(len(hashes))
    ]
    self.assertTrue(src.insert_and_lock(hashes, src_slices, True))
    self.assertTrue(dst.insert_and_lock(hashes, dst_slices, True))

    # The destination already holds every offered hash. That is a SUCCESS:
    # hashes are content-addressed, so the peer having them is exactly the
    # post-condition the caller wanted, and no bytes move.
    self.assertTrue(src.write_remote(hashes, dst_id))
    done, failed, pending, existing, unregistered = (
        src.poll_remote_write_status()
    )
    self.assertCountEqual(done, hashes)
    self.assertEmpty(failed)
    self.assertEmpty(pending)
    self.assertEmpty(existing)

  def test_write_remote_partial_exist_reports_the_overlap(self):
    src = self._make_registry_store("wr_src_partial")
    dst = self._make_registry_store("wr_dst_partial")
    dst_id = dst.raiden_id

    hashes = [b"wr_p_a", b"wr_p_b"]
    self.assertTrue(
        src.insert_and_lock(
            hashes,
            [
                kv_cache_store.RaidenBlockID(
                    src.raiden_id, i, kv_cache_store.BlockStatus.HOST
                )
                for i in range(len(hashes))
            ],
            True,
        )
    )
    # Only one of the two.
    self.assertTrue(
        dst.insert_and_lock(
            [hashes[0]],
            [
                kv_cache_store.RaidenBlockID(
                    dst_id, 5, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )

    self.assertTrue(src.write_remote(hashes, dst_id))
    done, failed, pending, existing, unregistered = (
        src.poll_remote_write_status()
    )
    self.assertEmpty(done)
    self.assertCountEqual(failed, hashes)
    self.assertEmpty(pending)
    # The fourth element is the whole reason this poller returns four: the
    # caller decides whether to reissue with the remainder, and needs to know
    # what the remainder is. The store does not retry on its own.
    self.assertCountEqual(existing, [hashes[0]])

  def test_write_remote_carries_a_non_utf8_hash(self):
    # Real block hashes are raw digests, essentially never valid UTF-8. These
    # travel through the bindings as bytes and over the wire as proto `bytes`;
    # while those fields were declared `string` the sender serialized happily
    # and the receiver rejected the whole message.
    binary_hash = os.urandom(32)
    src = self._make_registry_store("wr_src_binary")
    dst = self._make_registry_store("wr_dst_binary")
    dst_id = dst.raiden_id

    self.assertTrue(
        src.insert_and_lock(
            [binary_hash],
            [
                kv_cache_store.RaidenBlockID(
                    src.raiden_id, 0, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )
    self.assertTrue(
        dst.insert_and_lock(
            [binary_hash],
            [
                kv_cache_store.RaidenBlockID(
                    dst_id, 5, kv_cache_store.BlockStatus.HOST
                )
            ],
            True,
        )
    )

    self.assertTrue(src.write_remote([binary_hash], dst_id))
    done, failed, _, _, _ = src.poll_remote_write_status()
    self.assertCountEqual(done, [binary_hash])
    self.assertEmpty(failed)

  def test_expected_worker_count_zero_does_not_block(self):
    """The default must stay non-blocking."""
    start = time.time()
    store = kv_cache_store.KVCacheStore(
        capacity=4,
        num_shards=1,
        store_server_ip="127.0.0.1",
        expected_worker_count=0,
        kv_pool_group="test_group",
    )
    self.assertLess(time.time() - start, 30.0)
    self.assertTrue(store.raiden_controller_address)
    self.assertEqual(store.capacity(), 4)

  def test_expected_worker_count_times_out_when_no_worker_registers(self):
    """Workers that never arrive must fail AT CONSTRUCTION with timeout."""
    orig = os.environ.get("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S")
    os.environ["RAIDEN_EXPECTED_WORKERS_TIMEOUT_S"] = "2"
    try:
      start = time.time()
      with self.assertRaises(RuntimeError) as cm:
        kv_cache_store.KVCacheStore(
            capacity=4,
            num_shards=1,
            store_server_ip="127.0.0.1",
            expected_worker_count=1,
        )
      elapsed = time.time() - start
    finally:
      if orig is not None:
        os.environ["RAIDEN_EXPECTED_WORKERS_TIMEOUT_S"] = orig
      else:
        os.environ.pop("RAIDEN_EXPECTED_WORKERS_TIMEOUT_S", None)

    self.assertIn("timed out waiting for", str(cm.exception).lower())
    self.assertIn("worker", str(cm.exception).lower())
    self.assertGreaterEqual(elapsed, 2.0)
    self.assertLess(elapsed, 60.0)


class KVCacheStoreLoadWithSlicesTest(absltest.TestCase):
  """Covers load(..., slices=...), the pre-resolved-entry form.

  The validation cases below need no TPU: every one of them is rejected before
  any transfer is issued, which is the whole point of asserting on them.
  """

  def _make_store(self):
    return kv_cache_store.KVCacheStore(
        capacity=20, num_shards=1, store_server_ip="127.0.0.1"
    )

  def test_rejects_slices_count_mismatch(self):
    store = self._make_store()
    store_id = store.raiden_id
    slices = [
        kv_cache_store.RaidenBlockID(
            store_id, 0, kv_cache_store.BlockStatus.HOST
        )
    ]
    # Two hashes, one slice, two destinations: the store must not guess which
    # hash the single slice belongs to.
    self.assertFalse(
        store.load([b"h1", b"h2"], [0, 1], slices=slices)
    )

  def test_rejects_device_block_ids_count_mismatch(self):
    store = self._make_store()
    store_id = store.raiden_id
    slices = [
        kv_cache_store.RaidenBlockID(
            store_id, 0, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockID(
            store_id, 1, kv_cache_store.BlockStatus.HOST
        ),
    ]
    # Every block needs a destination; there is no load-to-host mode.
    self.assertFalse(store.load([b"h1", b"h2"], [0], slices=slices))

  def test_rejects_mixed_local_and_remote_slices(self):
    store = self._make_store()
    local_id = store.raiden_id
    peer_id = kv_cache_store.RaidenId("peer_job", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockID(
            local_id, 0, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockID(
            peer_id, 7, kv_cache_store.BlockStatus.REMOTE
        ),
    ]
    # One call is one source. A partially cached prefix looks exactly like
    # this, and the caller has to split it rather than hand it over whole.
    self.assertFalse(store.load([b"h1", b"h2"], [0, 1], slices=slices))

  def test_rejects_two_different_peers_in_one_call(self):
    store = self._make_store()
    peer_a = kv_cache_store.RaidenId("peer_a", "0", "kv_cache", 0)
    peer_b = kv_cache_store.RaidenId("peer_b", "0", "kv_cache", 0)
    slices = [
        kv_cache_store.RaidenBlockID(
            peer_a, 7, kv_cache_store.BlockStatus.REMOTE
        ),
        kv_cache_store.RaidenBlockID(
            peer_b, 9, kv_cache_store.BlockStatus.REMOTE
        ),
    ]
    self.assertFalse(store.load([b"h1", b"h2"], [0, 1], slices=slices))

  def test_rejects_slice_without_a_host_block(self):
    store = self._make_store()
    store_id = store.raiden_id
    # host_block_id -1 means the entry names no host block, so there is
    # nothing to copy up even though the status claims HOST.
    slices = [
        kv_cache_store.RaidenBlockID(
            store_id, -1, kv_cache_store.BlockStatus.HOST
        )
    ]
    self.assertFalse(store.load([b"h1"], [0], slices=slices))

  def test_empty_request_is_a_no_op(self):
    store = self._make_store()
    self.assertTrue(store.load([], [], slices=[]))
    done, failed, pending = store.poll_load_status()
    self.assertEmpty(done)
    self.assertEmpty(failed)
    self.assertEmpty(pending)

  def test_forwards_slices_in_the_order_the_impl_expects(self):
    """Guards the one transposition this wrapper can make.

    Python takes (hashes, device_block_ids, slices) because slices is the
    optional one; the bound C++ overload takes (hashes, slices,
    device_block_ids). Swapping the last two is a type error the binding would
    reject, but only once something actually calls it -- and on the Torch side
    there is no end-to-end test to do that. So assert the order here, where it
    costs nothing and needs no hardware.
    """
    store = self._make_store()
    store_id = store.raiden_id
    slices = [
        kv_cache_store.RaidenBlockID(
            store_id, 0, kv_cache_store.BlockStatus.HOST
        ),
        kv_cache_store.RaidenBlockID(
            store_id, 1, kv_cache_store.BlockStatus.HOST
        ),
    ]
    hashes = [b"h1", b"h2"]
    device_block_ids = [4, 5]

    mock_impl = unittest.mock.MagicMock()
    store._impl = mock_impl
    mock_impl.load.return_value = True

    self.assertTrue(store.load(hashes, device_block_ids, slices=slices))
    mock_impl.load.assert_called_with(
        hashes,
        [s._impl for s in slices],  # pylint: disable=protected-access
        device_block_ids,
    )

    # Omitting slices must still reach the two-argument overload, not the
    # three-argument one with an empty list.
    mock_impl.reset_mock()
    self.assertTrue(store.load(hashes, device_block_ids))
    mock_impl.load.assert_called_with(hashes, device_block_ids)


if __name__ == "__main__":
  absltest.main()
