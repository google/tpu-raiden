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
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from tpu_raiden.api.jax import kv_cache_manager
from tpu_raiden.api.jax import kv_cache_store

# Set XLA flags to force CPU/Host platform devices if running locally on
# simulator
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def find_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    return s.getsockname()[1]


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


class KVCacheStoreE2ETest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.controller_port = find_free_port()

  def setUp(self):
    super().setUp()
    device_type = "tpu"
    try:
      self.devices = jax.devices(device_type)
    except RuntimeError as exc:
      raise AssertionError(f"No {device_type} devices found") from exc

    if not self.devices:
      raise AssertionError(f"No {device_type} devices found")

    self.num_devices = len(self.devices)
    self.num_layers = 1
    self.skip_lock = True

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

  def _run_e2e_test(self, enable_multi_numa: bool):
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
    controller_port = self.controller_port

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
        raiden_controller_address=f"{get_local_ip()}:{controller_port}",
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
    sum_val = jax.jit(jnp.sum)(tpu_cache)
    self.assertEqual(float(sum_val), 0.0)

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


if __name__ == "__main__":
  absltest.main()
