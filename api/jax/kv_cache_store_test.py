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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from api.jax.kv_cache_store import KVCacheStore

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


class KVCacheStoreTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.devices = jax.devices("tpu")
    except RuntimeError as exc:
      raise AssertionError("No TPU devices found") from exc

    if not self.devices:
      raise AssertionError("No TPU devices found")

    self.num_devices = len(self.devices)

    try:
      self.cpu_devices = jax.devices("cpu")
    except RuntimeError:
      self.cpu_devices = []

  def create_mesh(self, devices, axis_shapes, axis_names):
    try:
      devices_arr = np.array(devices)
      num_required_devices = np.prod(axis_shapes)
      if len(devices_arr) < num_required_devices:
        raise AssertionError(
            f"Need {num_required_devices} devices, got {len(devices_arr)}"
        )
      device_array = devices_arr[:num_required_devices].reshape(axis_shapes)
      return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
      self.skipTest("Cannot create mesh.")
      return None

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
  def test_e2e_store(self, dtype):
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

    store = KVCacheStore(1, 4)

    # Store chunk 0 into hash 1
    # TPU slices 0:1 -> Host
    store.insert([1], tpu_arrs, [0], [1])

    # Store chunk 2 into hash 2
    # TPU slices 2:3 -> Host
    store.insert([2], tpu_arrs, [2], [1])

    # Lookup hash 2 into TPU slice 3
    hits, future = store.lookup_and_fetch([2], tpu_arrs, [3], [1])
    self.assertTrue(hits[0])
    if future is not None:
      future.Await()

    for i in range(n_layers):
      tpu_np = np.asarray(tpu_arrs[i])
      np.testing.assert_array_equal(tpu_np[3:4], tpu_np[2:3])


if __name__ == "__main__":
  absltest.main()
