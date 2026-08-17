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
import uuid

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu

resources = None
from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import kv_cache_store


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
_reg_log_file = None


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

  global _reg_log_file
  print(f"Starting Registry on port {_registry_port}")
  _reg_log_file = open("/tmp/raiden_registry.log", "w")
  _registry_process = subprocess.Popen(
      [
          registry_binary,
          f"--port={_registry_port}",
      ]
      + extra_flags,
      stdout=_reg_log_file,
      stderr=subprocess.STDOUT,
  )

  # Give them some time to start
  time.sleep(2)


def stop_servers():
  global _registry_process, _reg_log_file
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
  if _reg_log_file:
    try:
      _reg_log_file.close()
    except Exception:
      pass
    _reg_log_file = None


def setUpModule():
  os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"


def tearDownModule():
  pass


class KVCacheStoreE2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    start_servers()

  @classmethod
  def tearDownClass(cls):
    stop_servers()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")
    assert self.device.type == "tpu", f"Expected real PyTorch TPU device, got {self.device}"
    print(f"=== [DEVICE VERIFIED] Using real PyTorch TPU device: {self.device} ===")

    self.num_devices = 1  # E2E tests for PyTorch currently use single device logic for kv caches
    self.num_layers = 1
    self.skip_lock = True

  def tearDown(self):
    super().tearDown()

  def _run_e2e_save_and_load(self, use_slices: bool = False):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data
    # np.arange creates unique values for each element, ensuring different
    # values for different shards
    host_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache = torch.tensor(host_data, device=self.device)

    # Expected reference after loading saved blocks 0 and 1 into blocks 2 and 3: [a, b, a, b]
    expected_ref = host_data.copy()
    expected_ref[2] = host_data[0]
    expected_ref[3] = host_data[1]

    # 2. Get free port for controller
    controller_port = find_free_port()

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    # 3. Create KVCacheStore (Controller)
    print("=== [Step 3/9] Creating KVCacheStore (Controller) ===")
    tag = f"save_{uuid.uuid4().hex[:8]}"
    rid = kv_cache_store.RaidenId(f"{tag}_job", "0", f"{tag}_cache", 0)
    store = kv_cache_store.KVCacheStore(
        capacity=num_blocks,
        raiden_id=rid,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port,
    )

    # 4. Create KVCacheManager (Worker)
    print("=== [Step 4/9] Creating KVCacheManager (Worker) ===")
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
        worker_id=f"{tag}_worker_0",
    )

    # 5. Insert HBM blocks to KVCacheStore
    print("=== [Step 5/9] Inserting HBM blocks into KVCacheStore ===")
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
    print("=== [Step 6/9] Saving HBM blocks to Host DRAM (store.save) ===")
    self.assertTrue(store.pin(hashes))

    def get_slice_e2e(x):
      return x[0, 0, 0, 0, 0:16].cpu().numpy()

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

    # 7. Load from host DRAM into device HBM blocks [2, 3]
    print("=== [Step 7/8] Loading checkpoint from Host DRAM into TPU HBM blocks [2, 3] (store.load) ===")
    self.assertTrue(store.pin(hashes))
    if use_slices:
      # Hand the store the entries the caller already has instead of letting
      # it resolve the hashes again. Resolved AFTER the pin above, so nothing
      # can evict them out from under the slices -- that ordering is the whole
      # safety contract of this form, since it performs no lookup of its own.
      load_slices = [entry for _, entry in store.lookup(hashes)]
      self.assertLen(load_slices, 2)
      for entry in load_slices:
        self.assertEqual(entry.status, kv_cache_store.BlockStatus.HOST_AND_HBM)
      self.assertTrue(store.load(hashes, [2, 3], slices=load_slices))
    else:
      self.assertTrue(store.load(hashes, [2, 3]))

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

    try:
      torch.tpu.synchronize()
    except (AttributeError, RuntimeError):
      pass
    # 8. Verify device memory blocks [2, 3] match saved blocks [0, 1]
    print("=== [Step 8/8] Verifying restored TPU memory matches expected array [a, b, a, b] ===")
    np.testing.assert_array_equal(tpu_cache.cpu().numpy(), expected_ref)
    print("=== [SUCCESS] E2E Save/Load [0, 1] -> [2, 3] roundtrip verified on physical TPU! ===")
    del manager, store

  def test_e2e_save_and_load(self):
    self._run_e2e_save_and_load()

  # The same save/load/compare pipeline, driven through the slices form of
  # load. Running it as a variant rather than a separate test is the point:
  # `slices` is a shortcut past the store's own lookup, not a different
  # transfer, so the bytes that land in blocks [2, 3] must be the ones the
  # no-slices path produces.
  def test_e2e_save_and_load_with_slices(self):
    self._run_e2e_save_and_load(use_slices=True)

  def _run_remote_read_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_read_success: bool = True,
      use_slices: bool = False,
  ):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    # 1. Generate sequential distinct cache data for Job A
    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = torch.tensor(host_data_a, device=self.device)

    # Create empty Job B device memory with zeros
    zeros_b = np.zeros(shape, dtype=np.float32)
    tpu_cache_b = torch.tensor(zeros_b, device=self.device)

    # Calculate shard size in bytes
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    controller_port_a = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    # 2. Create Job A's KVCacheStore & KVCacheManager
    tag = f"read_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
        node_id=producer_node_id,
    )

    controller_port_b = find_free_port()
    # 3. Create Job B's KVCacheStore & KVCacheManager
    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
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

    try:
      # Wait for listeners to start
      time.sleep(1)

      # Raw non-UTF-8 bytes on purpose: production hashes are binary
      # digests, and the registry round-trip must survive them (the proto
      # hash fields are `bytes`; as `string` they were UTF-8-verified on
      # the wire and cross-store sharing silently found nothing).
      hashes = [
          b"\x93\xff\x00" + f"{tag}_h0".encode(),
          b"\x93\xff\x00" + f"{tag}_h1".encode(),
      ]

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


      # 5. Job B calls Lookup (enable_global=True)
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

      # 6. Job B reads straight from Job A into its own device blocks. The
      # source coordinates come from the lookup answer, so nothing needs to be
      # inserted into Job B's cache first.
      slices_b = [lookup_res_b[0][1], lookup_res_b[1][1]]
      self.assertTrue(store_b.read_remote(hashes, slices_b, [0, 1]))

      if not expect_read_success:
        failed = False
        for _ in range(500):
          _, read_failed, _ = store_b.poll_remote_read_status()
          if read_failed:
            self.assertEqual(set(read_failed), set(hashes))
            failed = True
            break
          time.sleep(0.01)
        self.assertTrue(
            failed,
            "expected read_remote to fail on producer/consumer node_id mismatch",
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

      # 8. The read is already in HBM -- there is no second Load step, and no
      # local record of it either. Job B's cache is still a miss for these
      # hashes: the bytes live only in the device blocks it named.
      self.assertEmpty(store_b.lookup(hashes))


      # 9. Verify byte-exact match on Job B TPU device
      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      np.testing.assert_array_equal(tpu_cache_b[0:2].cpu().numpy(), host_data_a[0:2])
    finally:
      del manager_a, manager_b, store_a, store_b

  def _run_remote_write_e2e_test(
      self,
      producer_node_id: int = 0,
      consumer_node_id: int = 0,
      expect_write_success: bool = True,
  ):
    """Job A offers blocks it owns; Job B pulls them and keeps them.

    The mirror image of _run_remote_read_e2e_test: there the destination asks,
    here the source offers. What this adds over the read path is the
    WriteRemote control plane -- the ack, the destination's all-or-nothing
    insert, global registration, and the source polling to COMMITTED -- with a
    byte comparison at the end, because every control-plane assertion can pass
    while nothing is actually transferred.

    NOTE: like every torch test in this tree, this is written for parity with
    the jax suite and is NOT part of any gate. It has never been executed.
    """
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)

    host_data_a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    tpu_cache_a = torch.tensor(host_data_a, device=self.device)
    # Zeroed, so a byte comparison cannot pass on data already present.
    tpu_cache_b = torch.tensor(
        np.zeros(shape, dtype=np.float32), device=self.device
    )

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

    controller_port_a = find_free_port()
    worker_port_a = find_free_port()
    worker_port_b = find_free_port()

    tag = f"write_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=worker_port_a,
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
        node_id=producer_node_id,
    )

    controller_port_b = find_free_port()
    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
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

    try:
      time.sleep(1)
      hashes = [f"{tag}_h0".encode(), f"{tag}_h1".encode()]

      # 1. Job A puts the blocks in HBM and saves them to host DRAM. Only
      #    host-resident blocks can be offered: the pull reads host memory.
      slices_a = [
          kv_cache_store.RaidenBlockID(
              rid_a,
              host_block_id=-1,
              device_block_id=i,
              status=kv_cache_store.BlockStatus.HBM,
          )
          for i in range(2)
      ]
      inserted_a, evicted_a = store_a.insert(hashes, slices_a, on_host=False)
      self.assertTrue(inserted_a)
      self.assertEmpty(evicted_a)
      self.assertTrue(store_a.pin(hashes))
      store_a.save(hashes)

      deadline = time.time() + 120
      while True:
        save_done, save_failed, _ = store_a.poll_save_status()
        if save_failed:
          raise RuntimeError(f"Job A Async Save failed: {save_failed}")
        if len(save_done) == len(hashes):
          break
        if time.time() > deadline:
          raise RuntimeError("Job A save did not complete in time")
        time.sleep(0.01)

      # 2. Job A offers them. Returns once Job B has decided, not once the
      #    bytes have moved.
      self.assertTrue(store_a.write_remote(hashes, rid_b))

      deadline = time.time() + 120
      done, failed, existing = [], [], []
      while time.time() < deadline:
        done, failed, pending, existing, unregistered = (
            store_a.poll_remote_write_status()
        )
        if done or failed or existing:
          break
        time.sleep(0.01)

      if not expect_write_success:
        self.assertNotEmpty(failed)
        self.assertEmpty(done)
        return

      self.assertCountEqual(done, hashes)
      self.assertEmpty(failed)
      self.assertEmpty(existing)

      # 3. Job B holds them locally, host-resident, as its own.
      lookup_b = store_b.lookup(hashes, enable_global=False)
      self.assertLen(lookup_b, len(hashes))
      for _, slice_b in lookup_b:
        self.assertEqual(slice_b.status, kv_cache_store.BlockStatus.HOST)

      # 4. Prove the bytes are real. Landed blocks arrive UNPINNED -- a remote
      #    write leaves ordinary evictable entries -- and load() refuses a
      #    block nothing is holding.
      self.assertTrue(store_b.pin(hashes))
      self.assertTrue(store_b.load(hashes, [0, 1]))
      deadline = time.time() + 120
      while True:
        load_done, load_failed, _ = store_b.poll_load_status()
        if load_failed:
          raise RuntimeError(f"Job B Load failed: {load_failed}")
        if len(load_done) == len(hashes):
          break
        if time.time() > deadline:
          raise RuntimeError("Job B load did not complete in time")
        time.sleep(0.01)

      try:
        torch.tpu.synchronize()
      except (AttributeError, RuntimeError):
        pass
      np.testing.assert_array_equal(
          tpu_cache_b[0:2].cpu().numpy(),
          host_data_a[0:2],
          err_msg=(
              "Job B's device memory does not byte-match what Job A offered:"
              " the remote write reported COMMITTED without moving the data"
          ),
      )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_remote_write_e2e(self):
    self._run_remote_write_e2e_test()

  def test_remote_write_e2e_matching_node_id(self):
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=0,
        expect_write_success=True,
    )

  def test_remote_write_e2e_mismatched_node_id_fails(self):
    # The destination pairs each of its workers with the source worker holding
    # its shards by node_id; a mismatch leaves the pull with no group to read
    # from, so the transfer fails and the source is told rather than left
    # pending.
    self._run_remote_write_e2e_test(
        producer_node_id=0,
        consumer_node_id=1,
        expect_write_success=False,
    )

  def test_remote_read_e2e_matching_node_id(self):
    self._run_remote_read_e2e_test(
        producer_node_id=7,
        consumer_node_id=7,
        expect_read_success=True,
    )

  def test_remote_read_e2e_mismatched_node_id_fails(self):
    self._run_remote_read_e2e_test(
        producer_node_id=1,
        consumer_node_id=2,
        expect_read_success=False,
    )

  def test_remote_read_e2e_source_missing_block_fails(self):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = torch.zeros(shape, dtype=torch.float32, device=self.device)
    tpu_cache_b = torch.zeros(shape, dtype=torch.float32, device=self.device)

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices
    controller_port_a = find_free_port()
    controller_port_b = find_free_port()

    tag = f"miss_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
    )

    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id=f"{tag}_worker_b",
        host_blocks_to_allocate=4,
    )

    try:
      time.sleep(1)
      ghost = [f"{tag}_ghost".encode()]
      slices = [
          kv_cache_store.RaidenBlockID(
              rid_a,
              host_block_id=0,
              device_block_id=-1,
              status=kv_cache_store.BlockStatus.REMOTE,
          )
      ]
      self.assertTrue(store_b.read_remote(ghost, slices, [0]))

      failed = False
      for _ in range(500):
        _, read_failed, _ = store_b.poll_remote_read_status()
        if read_failed:
          self.assertEqual(set(read_failed), set(ghost))
          failed = True
          break
        time.sleep(0.01)
      self.assertTrue(
          failed, "expected read_remote to fail (source is missing the block)"
      )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_remote_read_e2e_source_wrong_status_fails(self):
    num_blocks = 4
    shape = (num_blocks, 128, 8, 8, 128)
    tpu_cache_a = torch.zeros(shape, dtype=torch.float32, device=self.device)
    tpu_cache_b = torch.zeros(shape, dtype=torch.float32, device=self.device)

    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices
    controller_port_a = find_free_port()
    controller_port_b = find_free_port()

    tag = f"ws_{uuid.uuid4().hex[:8]}"
    rid_a = kv_cache_store.RaidenId(f"{tag}_job_a", "0", f"{tag}_cache_a", 0)
    store_a = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_a,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_a,
    )
    manager_a = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_a]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_a}",
        worker_id=f"{tag}_worker_a",
        host_blocks_to_allocate=4,
    )

    rid_b = kv_cache_store.RaidenId(f"{tag}_job_b", "0", f"{tag}_cache_b", 0)
    store_b = kv_cache_store.KVCacheStore(
        capacity=4,
        global_registry_address=f"localhost:{_registry_port}",
        raiden_id=rid_b,
        num_shards=self.num_devices,
        shard_size_bytes=shard_size_bytes,
        store_server_ip="localhost",
        raiden_controller_port=controller_port_b,
    )
    manager_b = kv_cache_manager.KVCacheManager(
        kv_caches=[[tpu_cache_b]],
        local_control_port=0,
        max_blocks=num_blocks,
        num_slots=2,
        unsafe_skip_buffer_lock=self.skip_lock,
        raiden_worker_port=find_free_port(),
        raiden_controller_address=f"localhost:{controller_port_b}",
        worker_id=f"{tag}_worker_b",
        host_blocks_to_allocate=4,
    )

    try:
      time.sleep(1)
      # Raw non-UTF-8 bytes on purpose: production hashes are binary
      # digests, and the registry round-trip must survive them (the proto
      # hash fields are `bytes`; as `string` they were UTF-8-verified on
      # the wire and cross-store sharing silently found nothing).
      hashes = [
          b"\x93\xff\x00" + f"{tag}_h0".encode(),
          b"\x93\xff\x00" + f"{tag}_h1".encode(),
      ]
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
      self.assertTrue(store_a.insert(hashes, slices_a, on_host=False)[0])

      time.sleep(0.5)
      slices_b = [
          kv_cache_store.RaidenBlockID(
              rid_a,
              host_block_id=0,
              device_block_id=-1,
              status=kv_cache_store.BlockStatus.REMOTE,
          ),
          kv_cache_store.RaidenBlockID(
              rid_a,
              host_block_id=0,
              device_block_id=-1,
              status=kv_cache_store.BlockStatus.REMOTE,
          ),
      ]
      self.assertTrue(store_b.read_remote(hashes, slices_b, [0, 1]))

      failed = False
      for _ in range(500):
        _, read_failed, _ = store_b.poll_remote_read_status()
        if read_failed:
          self.assertEqual(set(read_failed), set(hashes))
          failed = True
          break
        time.sleep(0.01)
      self.assertTrue(
          failed,
          "expected read_remote to fail (source only has block in HBM)",
      )
    finally:
      del manager_a, manager_b, store_a, store_b

  def test_expected_worker_count_waits_for_a_concurrent_registration(self):
    """The barrier replaces the sleep-and-hope idiom."""
    tpu_cache = torch.zeros(
        (2, 128, 8, 8, 128), dtype=torch.float32, device="tpu"
    )
    controller_port = find_free_port()
    num_blocks = 2
    block_elements = 128 * 8 * 8 * 128
    shard_size_bytes = (block_elements * 4) // self.num_devices

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
    self.assertGreaterEqual(elapsed, registration_delay_s)

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
    del built


if __name__ == "__main__":
  absltest.main()
