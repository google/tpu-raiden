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

"""E2E test for JAX KVCacheStore with TPUs."""

import os
import socket
import subprocess
import threading
import time
import unittest
import uuid

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

resources = None
from tpu_raiden.api.jax import kv_cache_manager
from tpu_raiden.api.jax import kv_cache_store

# Set XLA flags to force CPU/Host platform devices if running locally on
# simulator
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def _pick_unused_port():
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def find_free_port() -> int:
  return _pick_unused_port()


def get_local_ip() -> str:
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try:
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
  except OSError:
    ip = "127.0.0.1"
  finally:
    s.close()
  return ip


# Global variables for subprocesses
_registry_process = None
_registry_port = None


def start_servers():
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


def stop_servers():
  global _registry_process
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
    _registry_process = None


def setUpModule():
  os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"


def tearDownModule():
  pass


class KVCacheStoreE2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.controller_port = find_free_port()

  def setUp(self):
    super().setUp()
    start_servers()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      self.devices = jax.devices()

    if not self.devices:
      raise AssertionError("No JAX devices found")

    self.num_devices = len(self.devices)
    self.num_layers = 1
    self.skip_lock = True

  def tearDown(self):
    stop_servers()
    super().tearDown()

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
    axis_shapes = (1, self.num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model", None, None)
    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    return tpu_sharding

  def setup_sharding_for_devices(self, devices):
    axis_shapes = (1, len(devices))
    axis_names = ("data", "model")
    mesh = self.create_mesh(axis_shapes, axis_names, devices)
    spec = jax.sharding.PartitionSpec(None, None, "model", None, None)
    return jax.sharding.NamedSharding(mesh, spec)

  def _run_e2e_test(self, enable_multi_numa: bool):
    if enable_multi_numa and len(self.devices) > 4:
      # TODO(jcgu): Create a new multi-host setup
      # to test True Multi-NUMA ENABLE_MULTI_NUMA=1 correctly, since running
      # two distinct jobs inside this single-process sandbox blocks cross-NIC UDP routing.
      self.skipTest(
          "Multi-NUMA E2E test is not supported on single-host shared device "
          f"configurations (devices={len(self.devices)}). Skipping."
      )

    os.environ["ENABLE_MULTI_NUMA"] = "1" if enable_multi_numa else "0"

    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data
    # np.arange creates unique values for each element, ensuring different
    # values for different shards
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)
    expected_ref = host_data

    # 2. Get free port for controller
    controller_port = find_free_port()

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    # 3. Create KVCacheStore (Controller)
    rid = kv_cache_store.RaidenId("e2e_job", "0", "e2e_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=num_blocks,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )

    # 4. Create KVCacheManager (Worker)
    manager = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=0,
        # Must match the address the store's controller binds
        # ("localhost:{controller_port}", see the KVCacheStore above); using
        # get_local_ip() here dials a LAN IP the controller is not listening on,
        # so RegisterWorker never lands and Save fails with "No registered
        # workers available for TransferBuffers".
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_0",
    )

    # 5. Insert HBM blocks to KVCacheStore
    hashes = [b"hash_0", b"hash_1"]
    slices = [
        kv_cache_store.RaidenBlockID(
            rid,
            host_block_id=-1,
            device_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        ),
        kv_cache_store.RaidenBlockID(
            rid,
            host_block_id=-1,
            device_block_id=1,
            status=kv_cache_store.BlockStatus.HBM,
        ),
    ]
    inserted, evicted = store.insert(hashes, slices, on_host=False)
    self.assertTrue(inserted)
    self.assertEmpty(evicted)

    # Verify status in store is HBM
    lookup_res = store.lookup(hashes)
    self.assertLen(lookup_res, 2)
    self.assertEqual(lookup_res[0][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[0][1].device_block_id, 0)
    self.assertEqual(lookup_res[1][1].status, kv_cache_store.BlockStatus.HBM)
    self.assertEqual(lookup_res[1][1].device_block_id, 1)

    # 6. Save HBM blocks to host memory
    self.assertTrue(store.pin(hashes))

    @jax.jit
    def get_slice_e2e(x):
      return x[0, 0, 0, 0, 0:16]

    print(f"DEBUG: test_e2e tpu_cache before Save: {get_slice_e2e(tpu_cache)}")

    store.save(hashes)

    # Wait for save completion
    done = False
    while not done:
      save_done, save_failed, _ = store.poll_save_status()
      if save_failed:
        raise RuntimeError(f"Async Save failed: {save_failed}")
      if save_done:
        done = True
      if not done:
        time.sleep(0.01)

    # Release them so we can test pinning before load
    store.release(hashes)

    # Verify status in store is updated to HOST_AND_HBM
    lookup_res = store.lookup(hashes)
    self.assertLen(lookup_res, 2)
    self.assertEqual(
        lookup_res[0][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res[0][1].host_block_id, 0)
    self.assertEqual(
        lookup_res[1][1].status, kv_cache_store.BlockStatus.HOST_AND_HBM
    )
    self.assertEqual(lookup_res[1][1].host_block_id, 1)

    # 7. Overwrite device memory with zeros
    # host blocks 2 and 3 are empty/uninitialized, containing zeros
    manager.h2d([2, 2], [0, 1]).wait()

    # Verify they are indeed zeros using JIT sum to avoid host caching
    # sum_val = jax.jit(jnp.sum)(tpu_cache)
    # self.assertEqual(float(sum_val), 0.0)
    # print(f"DEBUG: np.asarray(tpu_cache) after overwrite with zeros: {np.asarray(tpu_cache)[0, 0, 0, 0, 0:5]}")

    # 8. Load from host DRAM back to device HBM
    self.assertTrue(store.pin(hashes))
    store.load(hashes, [0, 1])

    # Wait for load completion
    done = False
    while not done:
      load_done, load_failed, _ = store.poll_load_status()
      if load_failed:
        raise RuntimeError(f"Async Load failed: {load_failed}")
      if load_done:
        done = True
      if not done:
        time.sleep(0.01)

    # Release at the very end
    store.release(hashes)

    # 9. Verify device memory contains the original random data
    np.testing.assert_array_equal(np.asarray(tpu_cache), expected_ref)

  def test_e2e_without_multi_numa(self):
    self._run_e2e_test(enable_multi_numa=False)

  def test_e2e_with_multi_numa(self):
    self._run_e2e_test(enable_multi_numa=True)

  def _run_remote_read_e2e_test(
      self,
      enable_multi_numa: bool,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_read_success: bool = True,
  ):
    if enable_multi_numa and len(self.devices) > 4:
      # TODO(jcgu): Create a new multi-host setup
      # to test True Multi-NUMA ENABLE_MULTI_NUMA=1 correctly, since running
      # two distinct jobs inside this single-process sandbox blocks cross-NIC UDP routing.
      self.skipTest(
          "Multi-NUMA E2E test is not supported on single-host shared device "
          f"configurations (devices={len(self.devices)}). Skipping."
      )

    os.environ["ENABLE_MULTI_NUMA"] = "1" if enable_multi_numa else "0"

    if len(self.devices) < 1:
      self.skipTest(
          f"Requires at least 1 device, but only got {len(self.devices)}"
      )

    devices_a = self.devices
    devices_b = self.devices

    sharding_a = self.setup_sharding_for_devices(devices_a)
    sharding_b = self.setup_sharding_for_devices(devices_b)

    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data for Job A
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = jax.device_put(jnp.array(host_data_a), sharding_a)
    jax.block_until_ready(tpu_cache_a)

    # Overwrite Job B device memory with zeros
    zeros_b = np.zeros(shape, dtype=np.float32)
    tpu_cache_b = jax.device_put(jnp.array(zeros_b), sharding_b)
    jax.block_until_ready(tpu_cache_b)

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    num_shards = len(self.devices)
    shard_size_bytes = (block_elements * 4) // num_shards

    controller_port = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    # 2. Create Job A's KVCacheStore & KVCacheManager
    rid_a = kv_cache_store.RaidenId("job_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_a",
        node_id=producer_node_id,
    )

    controller_port_b = find_free_port()
    # 3. Create Job B's KVCacheStore & KVCacheManager
    rid_b = kv_cache_store.RaidenId("job_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_b,
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_b",
        host_blocks_to_allocate=4,  # Allocating enough space for receiver host blocks
        node_id=consumer_node_id,
    )

    # Wait for listeners to start
    time.sleep(1)

    hashes = [b"hash_0", b"hash_1"]

    # 4. Job A inserts HBM status and calls Save
    slices_a = [
        kv_cache_store.RaidenBlockID(
            rid_a,
            host_block_id=-1,
            device_block_id=0,
            status=kv_cache_store.BlockStatus.HBM,
        ),
        kv_cache_store.RaidenBlockID(
            rid_a,
            host_block_id=-1,
            device_block_id=1,
            status=kv_cache_store.BlockStatus.HBM,
        ),
    ]
    inserted_a, evicted_a = store_a.insert(hashes, slices_a, on_host=False)
    self.assertTrue(inserted_a)
    self.assertEmpty(evicted_a)

    self.assertTrue(store_a.pin(hashes))

    @jax.jit
    def get_slice(x):
      return x[0, 0, 0, 0, 0:16]

    print(f"DEBUG: Job A tpu_cache_a before Save: {get_slice(tpu_cache_a)}")

    store_a.save(hashes)

    # Wait for save completion
    done = False
    while not done:
      save_done, save_failed, _ = store_a.poll_save_status()
      if save_failed:
        raise RuntimeError(f"Job A Async Save failed: {save_failed}")
      if save_done:
        done = True
      if not done:
        time.sleep(0.01)

    data_a = manager_a._impl.read_host_memory(0, 0, 16)
    print(f"DEBUG: Job A host memory (layer 0, shard 0) after Save: {data_a}")

    store_a.release(hashes)

    # 5. Job B calls Lookup (enable_global=True)
    # Give some time for registry propagation
    time.sleep(0.5)
    lookup_res_b = store_b.lookup(hashes, enable_global=True)
    self.assertLen(lookup_res_b, 2)

    # Verify REMOTE status and owner job_a
    self.assertEqual(lookup_res_b[0][0], hashes[0])
    self.assertEqual(
        lookup_res_b[0][1].status, kv_cache_store.BlockStatus.REMOTE
    )
    self.assertEqual(lookup_res_b[0][1].raiden_id, rid_a)

    self.assertEqual(lookup_res_b[1][0], hashes[1])
    self.assertEqual(
        lookup_res_b[1][1].status, kv_cache_store.BlockStatus.REMOTE
    )
    self.assertEqual(lookup_res_b[1][1].raiden_id, rid_a)

    # Verify correct source host block IDs
    lookup_res_a = store_a.lookup(hashes)
    self.assertEqual(
        lookup_res_b[0][1].host_block_id, lookup_res_a[0][1].host_block_id
    )
    self.assertEqual(
        lookup_res_b[1][1].host_block_id, lookup_res_a[1][1].host_block_id
    )

    # 6. Job B controller calls insert_and_lock for the remote slices
    slices_b = [lookup_res_b[0][1], lookup_res_b[1][1]]
    self.assertTrue(store_b.insert_and_lock(hashes, slices_b, on_host=True))

    # 7. Job B calls ReadRemote
    self.assertTrue(store_b.read_remote(hashes))

    if not expect_read_success:
      # Strict node_id matching: the producer worker's node_id must equal the
      # consumer (destination) worker's node_id. A mismatch makes the source
      # controller find no destination group and the remote read fail.
      failed = False
      for _ in range(500):
        _, read_failed, _ = store_b.poll_remote_read_status()
        if read_failed:
          failed = True
          break
        time.sleep(0.01)
      self.assertTrue(
          failed,
          "expected ReadRemote to fail on producer/consumer node_id mismatch",
      )
      return

    # Wait for ReadRemote completion
    done = False
    while not done:
      read_done, read_failed, _ = store_b.poll_remote_read_status()
      if read_failed:
        raise RuntimeError(f"Job B ReadRemote failed: {read_failed}")
      if len(read_done) == 2:
        done = True
      if not done:
        time.sleep(0.01)

    data_b = manager_b._impl.read_host_memory(0, 0, 16)
    print(
        "DEBUG: Job B host memory (layer 0, shard 0) after ReadRemote:"
        f" {data_b}"
    )

    # 8. Verify Job B's LRU block status becomes HOST
    lookup_res_b_after = store_b.lookup(hashes)
    self.assertLen(lookup_res_b_after, 2)
    self.assertEqual(
        lookup_res_b_after[0][1].status, kv_cache_store.BlockStatus.HOST
    )
    self.assertEqual(
        lookup_res_b_after[1][1].status, kv_cache_store.BlockStatus.HOST
    )

    # 9. Job B controller calls Load to transfer data to TPU blocks
    self.assertTrue(store_b.load(hashes, [0, 1]))

    # Wait for Load completion
    done = False
    while not done:
      load_done, load_failed, _ = store_b.poll_load_status()
      if load_failed:
        raise RuntimeError(f"Job B Load failed: {load_failed}")
      if len(load_done) == 2:
        done = True
      if not done:
        time.sleep(0.01)

    store_b.release(hashes)

    # 10. Verify byte-exact match on Job B TPU devices
    np.testing.assert_array_equal(np.asarray(tpu_cache_b), host_data_a)

  # =========================================================================
  # ReadRemote to HBM (receiver-initiated pull straight into device memory)
  # =========================================================================

  def _reread_device(self, arr):
    """Re-reads device memory after a DMA.

    jax.device_get memoizes, so a plain np.asarray() of an array whose buffer
    was written behind JAX's back returns the stale pre-DMA host copy and every
    assertion passes vacuously. A jit round-trip forces a genuine read.
    """
    return np.asarray(jax.jit(lambda x: x * 1.0)(arr))

  def _run_remote_read_to_hbm_test(self, enable_multi_numa: bool):
    """Job A saves; Job B pulls straight into scattered device blocks.

    Covers, on real TPU and byte-exactly: save (D2H), read_remote to HBM
    (remote host -> local host staging -> local device), the resulting
    HOST_AND_HBM state, and load (H2D) from the host copy that the pull left
    behind as its staging hop.
    """
    if enable_multi_numa and len(self.devices) <= 4:
      self.skipTest(
          "Multi-NUMA E2E test is not supported on single-host shared device "
          f"configurations (devices={len(self.devices)}). Skipping."
      )
    os.environ["ENABLE_MULTI_NUMA"] = "1" if enable_multi_numa else "0"

    sharding_a = self.setup_sharding_for_devices(self.devices)
    sharding_b = self.setup_sharding_for_devices(self.devices)

    # 4 blocks: pull source blocks 0 and 1 into device blocks 3 and 0
    # (scattered AND reversed, so a cross-wired id cannot pass), leaving
    # 1 and 2 as untouched sentinels.
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    hashes = [b"\x93\xff\x00hbm_hash_0", b"\x93\xff\x00hbm_hash_1"]
    src_device_blocks = [0, 1]
    dst_device_blocks = [3, 0]
    sentinel_blocks = [1, 2]

    # Per-element random payloads with per-block-distinct content: a constant
    # fill compares "byte-exact" even when bytes were shuffled within a block
    # or blocks were cross-wired, which is exactly the failure class the
    # sharded pull path can introduce.
    rng = np.random.default_rng(20260728)
    host_data_a = rng.standard_normal(shape, dtype=np.float32)
    tpu_cache_a = jax.device_put(jnp.array(host_data_a), sharding_a)
    # Job B starts RANDOM, not zero, so a block that is never written cannot
    # accidentally match the source.
    host_data_b = np.random.default_rng(31415926).standard_normal(
        shape, dtype=np.float32
    )
    tpu_cache_b = jax.device_put(jnp.array(host_data_b), sharding_b)
    jax.block_until_ready(tpu_cache_a)
    jax.block_until_ready(tpu_cache_b)

    block_elements = 128 * 8 * 8 * 128
    num_shards = len(self.devices)
    shard_size_bytes = (block_elements * 4) // num_shards

    controller_port = find_free_port()
    controller_port_b = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    rid_a = kv_cache_store.RaidenId("job_hbm_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=8,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id="worker_hbm_a",
        node_id=0,
    )

    rid_b = kv_cache_store.RaidenId("job_hbm_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=8,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_b,
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_hbm_b",
        host_blocks_to_allocate=8,
        node_id=0,
    )
    # manager_a/manager_b must stay referenced for the whole test: they own the
    # worker gRPC servers and the transport listeners the stores talk to.
    self.assertIsNotNone(manager_a)
    self.assertIsNotNone(manager_b)
    time.sleep(1)

    # --- Job A: save (D2H) so the blocks are HOST-resident and leasable. ----
    slices_a = [
        kv_cache_store.RaidenBlockID(
            rid_a,
            host_block_id=-1,
            device_block_id=blk,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for blk in src_device_blocks
    ]
    inserted_a, _ = store_a.insert(hashes, slices_a, on_host=False)
    self.assertTrue(inserted_a)
    self.assertTrue(store_a.pin(hashes))
    store_a.save(hashes)
    self._await_terminal(store_a.poll_save_status, len(hashes), "Job A save")
    store_a.release(hashes)

    # --- Job B: discover the blocks as REMOTE. -----------------------------
    time.sleep(0.5)
    lookup_b = store_b.lookup(hashes, enable_global=True)
    self.assertLen(lookup_b, 2)
    for _, blk in lookup_b:
      self.assertEqual(blk.status, kv_cache_store.BlockStatus.REMOTE)
      self.assertEqual(blk.raiden_id, rid_a)

    # insert_and_lock pins the entries; the contract requires holding those
    # pins until poll_remote_read_status reports the hashes terminal.
    self.assertTrue(
        store_b.insert_and_lock(hashes, [b for _, b in lookup_b], on_host=True)
    )

    # --- The thing under test: pull straight into HBM. ---------------------
    self.assertTrue(store_b.read_remote(hashes, dst_device_blocks))
    self._await_terminal(
        store_b.poll_remote_read_status, len(hashes), "Job B read_remote"
    )

    # The batch commits as a unit into HOST_AND_HBM: the bytes are in the
    # caller's device blocks AND in the host landing blocks that were the
    # staging hop.
    after = store_b.lookup(hashes)
    self.assertLen(after, 2)
    for i, (_, blk) in enumerate(after):
      self.assertEqual(blk.status, kv_cache_store.BlockStatus.HOST_AND_HBM)
      self.assertEqual(blk.device_block_id, dst_device_blocks[i])

    # --- Byte-exact verification of device memory. -------------------------
    actual_b = self._reread_device(tpu_cache_b)
    for src_blk, dst_blk in zip(src_device_blocks, dst_device_blocks):
      np.testing.assert_array_equal(
          actual_b[dst_blk],
          host_data_a[src_blk],
          err_msg=(
              f"device block {dst_blk} does not byte-match source block"
              f" {src_blk}"
          ),
      )
    for blk in sentinel_blocks:
      np.testing.assert_array_equal(
          actual_b[blk],
          host_data_b[blk],
          err_msg=f"sentinel device block {blk} was clobbered by the pull",
      )

    # --- The host copy left behind by the staging hop must be usable. ------
    # load() the same hashes into the sentinel blocks; if the staging blocks
    # did not really hold the data, this produces garbage.
    self.assertTrue(store_b.load(hashes, sentinel_blocks))
    self._await_terminal(store_b.poll_load_status, len(hashes), "Job B load")
    store_b.release(hashes)

    reloaded = self._reread_device(tpu_cache_b)
    for src_blk, dst_blk in zip(src_device_blocks, sentinel_blocks):
      np.testing.assert_array_equal(
          reloaded[dst_blk],
          host_data_a[src_blk],
          err_msg=(
              f"load() into device block {dst_blk} does not byte-match source"
              f" block {src_blk}: the pull's host staging copy is not valid"
          ),
      )

  def _await_terminal(self, poll_fn, expected_done, what, timeout_s=120.0):
    """Polls until `expected_done` hashes report done, or anything fails."""
    deadline = time.time() + timeout_s
    seen_done = 0
    while time.time() < deadline:
      done, failed, _ = poll_fn()
      if failed:
        raise RuntimeError(f"{what} failed: {failed}")
      seen_done += len(done)
      if seen_done >= expected_done:
        return
      time.sleep(0.01)
    raise RuntimeError(f"{what} did not finish within {timeout_s}s")

  def _await_write_terminal(self, store, expected, timeout_s=120.0):
    """Polls a remote write to a verdict.

    Separate from _await_terminal because poll_remote_write_status returns
    four vectors, not three: `existing` carries what a destination already held
    when it refused a partial batch.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
      done, failed, pending, existing, unregistered = (
          store.poll_remote_write_status()
      )
      if done or failed:
        return done, failed, existing
      if not pending:
        raise RuntimeError("remote write vanished without a verdict")
      time.sleep(0.01)
    raise RuntimeError(f"remote write did not settle within {timeout_s}s")

  def _run_remote_write_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_write_success: bool = True,
      preload_destination: bool = False,
  ):
    """Job A offers blocks it owns; Job B pulls them and keeps them.

    The mirror image of _run_remote_read_e2e_test: there the destination asks,
    here the source offers. The data plane is the same pull either way -- only
    the destination ever writes the destination's memory -- so what this adds
    over the read path is the WriteRemote control plane: the ack, the
    destination's all-or-nothing insert, global registration, and the source
    polling to COMMITTED.

    Byte-exactness is the point. The control-plane unit tests can all pass
    while nothing is actually transferred.
    """
    os.environ["ENABLE_MULTI_NUMA"] = "0"
    if len(self.devices) < 1:
      self.skipTest("Requires at least 1 device")

    sharding_a = self.setup_sharding_for_devices(self.devices)
    sharding_b = self.setup_sharding_for_devices(self.devices)

    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = jax.device_put(jnp.array(host_data_a), sharding_a)
    jax.block_until_ready(tpu_cache_a)

    # Zeroed, so a byte comparison cannot pass on data that was already there.
    tpu_cache_b = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding_b
    )
    jax.block_until_ready(tpu_cache_b)

    block_elements = 128 * 8 * 8 * 128
    num_shards = len(self.devices)
    shard_size_bytes = (block_elements * 4) // num_shards

    controller_port = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    tag = f"write_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port}",
        worker_id=f"{tag}_worker_a",
        node_id=producer_node_id,
    )

    controller_port_b = find_free_port()
    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_b,
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id=f"{tag}_worker_b",
        host_blocks_to_allocate=4,
        node_id=consumer_node_id,
    )
    # Both managers must outlive the transfer: each owns its node's
    # WorkerService, which is what performs the D2H and serves/issues the pull.
    self.assertIsNotNone(manager_a)
    self.assertIsNotNone(manager_b)
    time.sleep(1)

    hashes = [b"\x93\xff\x00wr_hash_0", b"\x93\xff\x00wr_hash_1"]

    # 1. Job A puts the blocks in HBM and saves them to host DRAM. Only
    #    host-resident blocks can be offered: the pull reads host memory.
    slices_a = [
        kv_cache_store.RaidenBlockID(
            rid_a,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(num_blocks)
    ]
    inserted_a, evicted_a = store_a.insert(hashes, slices_a, on_host=False)
    self.assertTrue(inserted_a)
    self.assertEmpty(evicted_a)
    self.assertTrue(store_a.pin(hashes))

    self.assertTrue(store_a.save(hashes))
    self._await_terminal(store_a.poll_save_status, len(hashes), "Job A save")

    if preload_destination:
      # The destination already holds everything offered. A SUCCESS that moves
      # no bytes, and the one case where the source's blocks never travel.
      self.assertTrue(
          store_b.insert_and_lock(
              hashes,
              [
                  kv_cache_store.RaidenBlockID(
                      rid_b,
                      host_block_id=i,
                      status=kv_cache_store.BlockStatus.HOST,
                  )
                  for i in range(num_blocks)
              ],
              on_host=True,
          )
      )

    # 2. Job A offers them. Returns once Job B has decided, not once the bytes
    #    have moved.
    self.assertTrue(store_a.write_remote(hashes, rid_b))
    done, failed, existing = self._await_write_terminal(store_a, hashes)

    if not expect_write_success:
      self.assertNotEmpty(
          failed, "expected the remote write to fail, but it reported done"
      )
      self.assertEmpty(done)
      return

    self.assertCountEqual(done, hashes)
    self.assertEmpty(failed)
    self.assertEmpty(existing)
    store_a.release(hashes)

    # 3. Job B holds them locally, host-resident, as its own.
    lookup_b = store_b.lookup(hashes, enable_global=False)
    self.assertLen(lookup_b, len(hashes))
    for _, slice_b in lookup_b:
      self.assertEqual(slice_b.status, kv_cache_store.BlockStatus.HOST)

    if preload_destination:
      # Nothing was transferred, so there is nothing of Job A's to compare.
      return

    # 4. Prove the bytes are real. Landed blocks arrive UNPINNED -- a remote
    #    write leaves ordinary evictable entries -- and load() refuses a block
    #    nothing is holding.
    self.assertTrue(store_b.pin(hashes))
    self.assertTrue(store_b.load(hashes, list(range(num_blocks))))
    self._await_terminal(store_b.poll_load_status, len(hashes), "Job B load")
    store_b.release(hashes)

    np.testing.assert_array_equal(
        self._reread_device(tpu_cache_b),
        host_data_a,
        err_msg=(
            "Job B's device memory does not byte-match what Job A offered:"
            " the remote write reported COMMITTED without moving the data"
        ),
    )

  def test_remote_write_e2e(self):
    self._run_remote_write_e2e_test()

  def test_remote_write_e2e_matching_node_id(self):
    # Single-host matching node_ids: exercises the node_id plumbing end to end,
    # including the destination's host_blocks_to_allocate branch.
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=0,
        expect_write_success=True,
    )

  def test_remote_write_e2e_mismatched_node_id_fails(self):
    # The destination pairs each of its workers with the source worker holding
    # its shards by node_id. A mismatch leaves the pull with no group to read
    # from, so the transfer fails and the source is told so rather than being
    # left pending.
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=1,
        expect_write_success=False,
    )

  def test_remote_write_e2e_all_exist_moves_nothing(self):
    # The destination already holds every offered hash. SUCCESS with no
    # transfer: hashes are content-addressed, so the peer having them is
    # exactly the post-condition the caller wanted.
    self._run_remote_write_e2e_test(preload_destination=True)

  def test_remote_read_to_hbm_without_multi_numa(self):
    self._run_remote_read_to_hbm_test(enable_multi_numa=False)

  @unittest.skip("Multi-NUMA test skipped in sandboxed test env due to port contention")
  def test_remote_read_to_hbm_with_multi_numa(self):
    self._run_remote_read_to_hbm_test(enable_multi_numa=True)

  def test_remote_read_e2e_without_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=False)

  @unittest.skip("Multi-NUMA test skipped in sandboxed test env due to port contention")
  def test_remote_read_e2e_with_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=True)

  def test_remote_read_e2e_matching_node_id(self):
    # Non-zero, matching node_ids on producer and consumer: exercises node_id
    # plumbing end-to-end (including the consumer's host_blocks_to_allocate
    # branch) and strict node_id matching succeeding.
    self._run_remote_read_e2e_test(
        enable_multi_numa=False,
        producer_node_id=7,
        consumer_node_id=7,
        expect_read_success=True,
    )

  def test_remote_read_e2e_mismatched_node_id_fails(self):
    # Producer and consumer node_ids differ: strict matching finds no
    # destination group for the producer worker and the remote read fails.
    self._run_remote_read_e2e_test(
        enable_multi_numa=False,
        producer_node_id=1,
        consumer_node_id=2,
        expect_read_success=False,
    )

  def test_remote_read_e2e_source_missing_block_fails(self):
    # ReadRemote All-or-Nothing validate & pin block hashes at the src controller: the destination requests a block hash the SOURCE does
    # not hold in host DRAM. The source controller's verify hook returns
    # BLOCK_HASH_NOT_FOUND, the read fails, and the destination frees the local
    # host block it allocated.
    if len(self.devices) < 1:
      self.skipTest("Requires at least 1 device")
    os.environ["ENABLE_MULTI_NUMA"] = "0"

    sharding = self.setup_sharding_for_devices(self.devices)
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    tpu_cache_b = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    jax.block_until_ready(tpu_cache_a)
    jax.block_until_ready(tpu_cache_b)

    num_shards = len(self.devices)
    shard_size_bytes = (128 * 8 * 8 * 128 * 4) // num_shards
    controller_port_a = find_free_port()
    controller_port_b = find_free_port()

    # Source (Job A): running + registered, but never saves any block, so its
    # LRU has no HOST-resident blocks.
    rid_a = kv_cache_store.RaidenId("miss_job_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id="worker_a",
    )

    # Destination (Job B).
    rid_b = kv_cache_store.RaidenId("miss_job_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_b",
        host_blocks_to_allocate=4,
    )
    time.sleep(1)

    # Job B manually records a REMOTE block pointing at Job A for a hash Job A
    # never saved, then tries to read it.
    ghost = [b"ghost_hash"]
    slices = [
        kv_cache_store.RaidenBlockID(
            rid_a,
            host_block_id=0,
            device_block_id=-1,
            status=kv_cache_store.BlockStatus.REMOTE,
        )
    ]
    self.assertTrue(store_b.insert_and_lock(ghost, slices, on_host=True))

    self.assertTrue(store_b.read_remote(ghost))

    failed = False
    for _ in range(500):
      _, read_failed, _ = store_b.poll_remote_read_status()
      if read_failed:
        failed = True
        break
      time.sleep(0.01)
    self.assertTrue(
        failed, "expected ReadRemote to fail (source is missing the block)"
    )
    del manager_a, manager_b, store_a, store_b

  def test_remote_read_e2e_source_wrong_status_fails(self):
    # ReadRemote step 6a: the source HOLDS the block but only in HBM (never
    # saved to host DRAM). The source verify hook returns FAILED_PRECONDITION and
    # the destination read fails.
    if len(self.devices) < 1:
      self.skipTest("Requires at least 1 device")
    os.environ["ENABLE_MULTI_NUMA"] = "0"

    sharding = self.setup_sharding_for_devices(self.devices)
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    tpu_cache_b = jax.device_put(
        jnp.array(np.zeros(shape, dtype=np.float32)), sharding
    )
    jax.block_until_ready(tpu_cache_a)
    jax.block_until_ready(tpu_cache_b)

    num_shards = len(self.devices)
    shard_size_bytes = (128 * 8 * 8 * 128 * 4) // num_shards
    controller_port_a = find_free_port()
    controller_port_b = find_free_port()

    rid_a = kv_cache_store.RaidenId("ws_job_a", "0", "cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_a],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id="worker_a",
    )
    # Source records the block in HBM only (no Save -> not host-resident).
    hashes = [b"\x93\xff\x00hbm_only"]
    self.assertTrue(
        store_a.insert(
            hashes,
            [
                kv_cache_store.RaidenBlockID(
                    rid_a,
                    host_block_id=-1,
                    device_block_id=0,
                    status=kv_cache_store.BlockStatus.HBM,
                )
            ],
            on_host=False,
        )[0]
    )

    rid_b = kv_cache_store.RaidenId("ws_job_b", "0", "cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[tpu_cache_b],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id="worker_b",
        host_blocks_to_allocate=4,
    )
    time.sleep(1)

    # Destination records a REMOTE reference to the source's HBM-only block.
    self.assertTrue(
        store_b.insert_and_lock(
            hashes,
            [
                kv_cache_store.RaidenBlockID(
                    rid_a,
                    host_block_id=0,
                    device_block_id=-1,
                    status=kv_cache_store.BlockStatus.REMOTE,
                )
            ],
            on_host=True,
        )
    )
    self.assertTrue(store_b.read_remote(hashes))

    failed = False
    for _ in range(500):
      _, read_failed, _ = store_b.poll_remote_read_status()
      if read_failed:
        failed = True
        break
      time.sleep(0.01)
    self.assertTrue(
        failed, "expected ReadRemote to fail (source block not host-resident)"
    )
    del manager_a, manager_b, store_a, store_b

  def test_expected_worker_count_waits_for_a_concurrent_registration(self):
    """The barrier replaces the sleep-and-hope idiom.

    Every other test here builds the store, builds the manager, then sleeps and
    hopes registration landed. With expected_worker_count the store constructor
    IS the wait: it returns only once the worker is registered, so the transfer
    below needs no sleep at all.

    This also covers the binding releasing the GIL. The worker registers from
    another Python thread, so a constructor that held the GIL while blocking
    would starve that thread and time out instead.
    """
    os.environ["ENABLE_MULTI_NUMA"] = "0"
    tpu_sharding = self.setup_shardings()
    num_blocks = 2
    shape = (num_blocks, 128, 8, 8, 128)
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = jax.device_put(jnp.array(host_data), tpu_sharding)
    jax.block_until_ready(tpu_cache)

    controller_port = find_free_port()
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    # The manager is what registers the worker, and it can only be built once
    # the controller address is known -- which is why the port is fixed here
    # rather than left to gRPC.
    registration_delay_s = 2.0
    built = {}

    def build_manager_after_a_delay():
      time.sleep(registration_delay_s)
      built["manager"] = kv_cache_manager.KVCacheManager(
          kv_caches=[tpu_cache],
          local_control_port=0,
          max_blocks=num_blocks,
          num_slots=2,
          unsafe_skip_buffer_lock=self.skip_lock,
          raiden_worker_port=0,
          raiden_controller_address=f"localhost:{controller_port}",
          worker_id="worker_0",
      )

    worker_thread = threading.Thread(target=build_manager_after_a_delay)
    worker_thread.start()
    try:
      rid = kv_cache_store.RaidenId("barrier_job", "0", "barrier_cache", 0)
      start = time.time()
      store = kv_cache_store.KVCacheStore(
          capacity=num_blocks,
          raiden_id=rid,
          num_shards=self.num_devices,
          shard_size_bytes=shard_size_bytes,
          store_server_ip="localhost",
          raiden_controller_port=controller_port,
          expected_worker_count=1,
      )
      elapsed = time.time() - start
    finally:
      worker_thread.join(timeout=180)

    self.assertFalse(worker_thread.is_alive())
    # It really waited for the registration rather than returning early and
    # getting lucky.
    self.assertGreaterEqual(elapsed, registration_delay_s)

    # And the store is usable IMMEDIATELY -- no sleep between here and a
    # transfer. That is the property the barrier exists to provide, and it is
    # what fails if the constructor returns before the worker is registered.
    hashes = [b"barrier_hash_0", b"barrier_hash_1"]
    slices = [
        kv_cache_store.RaidenBlockID(
            rid,
            host_block_id=-1,
            device_block_id=i,
            status=kv_cache_store.BlockStatus.HBM,
        )
        for i in range(num_blocks)
    ]
    inserted, _ = store.insert(hashes, slices, on_host=False)
    self.assertTrue(inserted)
    self.assertTrue(store.pin(hashes))
    self.assertTrue(store.save(hashes))

    deadline = time.time() + 60
    done = []
    while time.time() < deadline:
      done, failed, _ = store.poll_save_status()
      self.assertEmpty(failed)
      if done:
        break
      time.sleep(0.01)
    self.assertCountEqual(done, hashes)
    store.release(hashes)
    del built


if __name__ == "__main__":
  absltest.main()
