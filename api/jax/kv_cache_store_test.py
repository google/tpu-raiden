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
import socket
import subprocess
import time

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np


from api.jax import kv_cache_store

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


def pick_unused_port():
  s = socket.socket()
  s.bind(("localhost", 0))
  port = s.getsockname()[1]
  s.close()
  return port


class KVCacheStoreTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError:
      try:
        self.devices = jax.devices("cpu")
      except RuntimeError as exc:
        raise AssertionError("No devices found") from exc

    if not self.devices:
      raise AssertionError("No devices found")

    self.num_devices = len(self.devices)

  def tearDown(self):
    if hasattr(self, "registry_process"):
      self.registry_process.terminate()
      self.registry_process.wait()
    super().tearDown()

  def create_mesh(self, devices, axis_shapes, axis_names):
    devices_arr = np.array(devices)
    num_required_devices = np.prod(axis_shapes)
    if len(devices_arr) < num_required_devices:
      raise AssertionError(
          f"Need {num_required_devices} devices, got {len(devices_arr)}"
      )
    device_array = devices_arr[:num_required_devices].reshape(axis_shapes)
    return jax.sharding.Mesh(device_array, axis_names)

  def setup_shardings(self, devices):
    axis_shapes = (1, self.num_devices)
    axis_names = ("data", "model")
    mesh = self.create_mesh(devices, axis_shapes, axis_names)
    spec = jax.sharding.PartitionSpec(None, None, "model")
    sharding = jax.sharding.NamedSharding(mesh, spec)
    return sharding

  @parameterized.named_parameters(
      ("fp32", jnp.float32),
  )
  def test_e2e_local_store(self, dtype):
    tpu_sharding = self.setup_shardings(self.devices)
    key = jax.random.key(101)

    n_layers = 4
    test_shape = (4, 128, 8, 8, 128)
    tpu_arrs = []
    for i in range(n_layers):
      sub_key = jax.random.fold_in(key, i)
      base = jax.random.uniform(sub_key, test_shape, dtype=dtype)
      tpu_arrs.append(jax.device_put(base, tpu_sharding))

    jax.block_until_ready(tpu_arrs)


    store = kv_cache_store.KVCacheStore(1, 4)

    # Store chunk 0 into hash 1 (local LRU insert)
    store.insert([1], tpu_arrs, [0], [1])

    # Store chunk 2 into hash 2 (local LRU insert)
    store.insert([2], tpu_arrs, [2], [1])

    # Lookup hash 2 into TPU slice 3 (local LRU hit)
    hits, future = store.lookup_and_fetch([2], tpu_arrs, [3], [1])
    self.assertTrue(hits[0])
    if future is not None:
      future.Await()

    for i in range(n_layers):
      tpu_np = np.asarray(tpu_arrs[i])
      np.testing.assert_array_equal(tpu_np[3:4], tpu_np[2:3])

  @parameterized.named_parameters(
      ("fp32", jnp.float32),
  )
  def test_global_lookup_fallback(self, dtype):
    # 1. Spin up global registry server
    server_path = os.path.join(
        os.environ["TEST_SRCDIR"],
        "google3/third_party/tpu_raiden/kv_cache/global_registry/global_registry_server",
    )
    registry_port = pick_unused_port()
    self.registry_process = subprocess.Popen(
        [server_path, f"--port={registry_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Simple sleep to wait for server initialization
    time.sleep(1.0)

    tpu_sharding = self.setup_shardings(self.devices)
    key = jax.random.key(101)

    n_layers = 4
    test_shape = (4, 128, 8, 8, 128)

    # Allocate separate remote and local JAX arrays
    remote_tpu_arrs = []
    local_tpu_arrs = []
    for i in range(n_layers):
      sub_key1 = jax.random.fold_in(key, i)
      sub_key2 = jax.random.fold_in(key, i + 10)
      remote_tpu_arrs.append(
          jax.device_put(
              jax.random.uniform(sub_key1, test_shape, dtype=dtype),
              tpu_sharding,
          )
      )
      local_tpu_arrs.append(
          jax.device_put(
              jax.random.uniform(sub_key2, test_shape, dtype=dtype),
              tpu_sharding,
          )
      )

    jax.block_until_ready(remote_tpu_arrs)
    jax.block_until_ready(local_tpu_arrs)

    # 2. Setup the "Remote Node" (listening on dynamic port)

    remote_store = kv_cache_store.KVCacheStore(
        block_size=1,
        capacity=4,
        global_registry_address=f"localhost:{registry_port}",
        local_address="[::1]:0",
    )

    # 3. Setup the "Local Node" (listening on separate dynamic port)

    local_store = kv_cache_store.KVCacheStore(
        block_size=1,
        capacity=4,
        global_registry_address=f"localhost:{registry_port}",
        local_address="[::1]:0",
    )

    # Insert chunk 2 into remote_store under hash 100
    # This automatically registers hash 100 globally
    remote_store.insert([100], remote_tpu_arrs, [2], [1])

    # Wait for remote insert async copies to be ready
    time.sleep(0.5)

    # Lookup hash 100 on local_store (will local miss, global hit on remote
    # node, and pull via H2H socket)
    hits, future = local_store.lookup_and_fetch([100], local_tpu_arrs, [0], [1])
    self.assertTrue(hits[0])

    if future is not None:
      future.Await()

    # Assert that local array at slice 0 is identical to the remote array
    # source slice 2
    for i in range(n_layers):
      remote_np = np.asarray(remote_tpu_arrs[i])
      local_np = np.asarray(local_tpu_arrs[i])
      np.testing.assert_array_equal(local_np[0:1], remote_np[2:3])


if __name__ == "__main__":
  absltest.main()
