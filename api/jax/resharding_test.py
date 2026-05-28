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

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
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
"""Integration tests for TPU Raiden's FFI resharding collective on CPU."""

import os

# Force host platform to emulate multiple CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from api.jax import resharding_engine
from api.jax import resharding_test_lib
import numpy as np


class ReshardingIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.devices = jax.devices("cpu")
    self.assertGreaterEqual(
        len(self.devices),
        12,
        "Emulated CPU device count must be at least 12 to run resharding tests",
    )
    self.K = 128
    self.N = 1024
    self.global_shape = (self.K, self.N)
    # TODO(b/12345678): Use jnp.bfloat16 on physical TPU accelerators. Standard
    # float32 is used here to avoid custom JAX bfloat16 NumPy array assignment
    # limitations during local CPU mesh emulation.
    self.dtype = jnp.float32

  def test_reshard_axis_1_4dev_to_axis_0_8dev(self):
    # Mesh 1: Source Mesh (4 devices, 1D)
    src_mesh = jax.sharding.Mesh(np.array(self.devices[0:4]), ("x",))
    src_sharding = jax.sharding.NamedSharding(
        src_mesh, jax.sharding.PartitionSpec(None, "x")  # sharded on Axis 1
    )

    # Mesh 2: Destination Mesh (8 devices, 1D)
    dst_mesh = jax.sharding.Mesh(np.array(self.devices[4:12]), ("y",))
    dst_sharding = jax.sharding.NamedSharding(
        dst_mesh, jax.sharding.PartitionSpec("y", None)  # sharded on Axis 0
    )

    # Create global data
    global_data = jnp.array(
        np.arange(self.K * self.N, dtype=np.float32).reshape(self.K, self.N),
        dtype=self.dtype,
    )

    # Place on source devices
    src_sharded_array = jax.device_put(global_data, src_sharding)
    jax.block_until_ready(src_sharded_array)

    # Trigger generalized worker-to-worker resharding collective (Using WeightSynchronizer!)
    dst_sharded_array = resharding_engine.reshard_matrix(
        src_sharded_array=src_sharded_array,
        dst_sharding=dst_sharding,
    )
    jax.block_until_ready(dst_sharded_array)

    # Assert parity
    local_dst_view = multihost_utils.global_array_to_host_local_array(
        dst_sharded_array, dst_mesh, jax.sharding.PartitionSpec("y", None)
    )
    gathered_dst = multihost_utils.process_allgather(local_dst_view)
    np.testing.assert_array_equal(
        np.squeeze(np.asarray(gathered_dst), axis=0), np.asarray(global_data)
    )

  def test_reshard_2D_2x2_to_1D_1x4(self):
    # Mesh 1: Source Mesh (4 devices, arranged in 2D grid 2x2)
    src_mesh = jax.sharding.Mesh(
        np.array(self.devices[0:4]).reshape(2, 2), ("x", "y")
    )
    src_sharding = jax.sharding.NamedSharding(
        src_mesh,
        jax.sharding.PartitionSpec("x", "y"),  # sharded on both dimensions!
    )

    # Mesh 2: Destination Mesh (4 devices, arranged in 1D grid 1x4)
    dst_mesh = jax.sharding.Mesh(np.array(self.devices[4:8]), ("z",))
    dst_sharding = jax.sharding.NamedSharding(
        dst_mesh,
        jax.sharding.PartitionSpec(None, "z"),  # sharded on Axis 1 only!
    )

    # Create global data
    global_data = jnp.array(
        np.arange(self.K * self.N, dtype=np.float32).reshape(self.K, self.N),
        dtype=self.dtype,
    )

    # Place on source 2D devices
    src_sharded_array = jax.device_put(global_data, src_sharding)
    jax.block_until_ready(src_sharded_array)

    # Assert source shard shapes: each device holds shape [128/2, 1024/2] = [64, 512]!
    self.assertLen(src_sharded_array.addressable_shards, 4)
    for shard in src_sharded_array.addressable_shards:
      self.assertEqual(shard.data.shape, (64, 512))

    # Trigger generalized worker-to-worker resharding collective E2E!
    dst_sharded_array = resharding_engine.reshard_matrix(
        src_sharded_array=src_sharded_array,
        dst_sharding=dst_sharding,
    )
    jax.block_until_ready(dst_sharded_array)

    # Assert destination shard shapes: each device holds shape [128, 1024/4] = [128, 256]!
    self.assertLen(dst_sharded_array.addressable_shards, 4)
    for shard in dst_sharded_array.addressable_shards:
      self.assertEqual(shard.data.shape, (128, 256))

    # Assert perfect mathematical parity!
    local_dst_view = multihost_utils.global_array_to_host_local_array(
        dst_sharded_array, dst_mesh, jax.sharding.PartitionSpec(None, "z")
    )
    gathered_dst = multihost_utils.process_allgather(local_dst_view)
    np.testing.assert_array_equal(
        np.squeeze(np.asarray(gathered_dst), axis=0), np.asarray(global_data)
    )

  def test_controller_piped_resharding_axis_1_4dev_to_axis_0_8dev(self):
    # Mesh 1: Source Mesh (4 devices, 1D)
    src_mesh = jax.sharding.Mesh(np.array(self.devices[0:4]), ("x",))
    src_sharding = jax.sharding.NamedSharding(
        src_mesh, jax.sharding.PartitionSpec(None, "x")
    )

    # Mesh 2: Destination Mesh (8 devices, 1D)
    dst_mesh = jax.sharding.Mesh(np.array(self.devices[4:12]), ("y",))
    dst_sharding = jax.sharding.NamedSharding(
        dst_mesh, jax.sharding.PartitionSpec("y", None)
    )

    global_data = jnp.array(
        np.arange(self.K * self.N, dtype=np.float32).reshape(self.K, self.N),
        dtype=self.dtype,
    )
    src_sharded_array = jax.device_put(global_data, src_sharding)
    jax.block_until_ready(src_sharded_array)

    # Trigger controller-piped prototype
    dst_sharded_array = resharding_test_lib.reshard_matrix_controller_piped(
        src_sharded_array=src_sharded_array,
        dst_sharding=dst_sharding,
    )
    jax.block_until_ready(dst_sharded_array)
    np.testing.assert_array_equal(
        np.asarray(dst_sharded_array), np.asarray(global_data)
    )


if __name__ == "__main__":
  absltest.main()
