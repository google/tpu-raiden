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
import time

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
_orchestrator_process = None
_registry_process = None
_orchestrator_port = None
_registry_port = None


def start_servers():
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


def stop_servers():
  global _orchestrator_process, _registry_process
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
    _orchestrator_process = None
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
        raiden_controller_address=f"localhost:{controller_port}",
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
        raiden_orchestrator_address=f"localhost:{_orchestrator_port}",
        raiden_controller_address=f"localhost:{controller_port}",
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
        raiden_orchestrator_address=f"localhost:{_orchestrator_port}",
        raiden_controller_address=f"localhost:{controller_port_b}",
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
    self.assertEqual(lookup_res_b[0][0], b"hash_0")
    self.assertEqual(
        lookup_res_b[0][1].status, kv_cache_store.BlockStatus.REMOTE
    )
    self.assertEqual(lookup_res_b[0][1].raiden_id, rid_a)

    self.assertEqual(lookup_res_b[1][0], b"hash_1")
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

  def test_remote_read_e2e_without_multi_numa(self):
    self._run_remote_read_e2e_test(enable_multi_numa=False)

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
    # ReadRemote step 6a: the destination requests a block hash the SOURCE does
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
        raiden_orchestrator_address=f"localhost:{_orchestrator_port}",
        raiden_controller_address=f"localhost:{controller_port_a}",
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
        raiden_orchestrator_address=f"localhost:{_orchestrator_port}",
        raiden_controller_address=f"localhost:{controller_port_b}",
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


if __name__ == "__main__":
  absltest.main()
