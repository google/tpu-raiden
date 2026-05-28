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
"""Tests for resharding_planner."""

import os

# Force host platform to emulate multiple CPU devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

from absl.testing import absltest
import jax
from api.jax import resharding_planner
import numpy as np


class ReshardingPlannerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.devices = jax.devices("cpu")

  def test_reshard_axis_1_to_axis_0(self):
    # Global Shape: [128, 1024]
    # Source: 4 devices, sharded on Axis 1.
    src_mesh = jax.sharding.Mesh(np.array(self.devices[0:4]), ("x",))
    src_sharding = jax.sharding.NamedSharding(
        src_mesh, jax.sharding.PartitionSpec(None, "x")
    )

    # Destination: 8 devices, sharded on Axis 0.
    dst_mesh = jax.sharding.Mesh(np.array(self.devices[4:12]), ("y",))
    dst_sharding = jax.sharding.NamedSharding(
        dst_mesh, jax.sharding.PartitionSpec("y", None)
    )

    plan = resharding_planner.make_resharding_plan(
        global_shape=(128, 1024),
        src_sharding=src_sharding,
        dst_sharding=dst_sharding,
    )

    # Total combination checks: 4 source devices * 8 destination devices = 32 chunks
    self.assertLen(plan, 32)

    # Verify a few specific chunks:
    # Chunk from src 0 to dst 0:
    chunk_0_0 = [
        c for c in plan if c.src_device_id == 0 and c.dst_device_id == 0
    ][0]
    self.assertEqual(chunk_0_0.src_slice, (0, 16, 0, 256))
    self.assertEqual(chunk_0_0.dst_slice, (0, 16, 0, 256))
    self.assertEqual(chunk_0_0.shape, (16, 256))

    # Chunk from src 1 to dst 3:
    chunk_1_3 = [
        c for c in plan if c.src_device_id == 1 and c.dst_device_id == 3
    ][0]
    self.assertEqual(chunk_1_3.src_slice, (48, 64, 0, 256))
    self.assertEqual(chunk_1_3.dst_slice, (0, 16, 256, 512))
    self.assertEqual(chunk_1_3.shape, (16, 256))

    # Chunk from src 3 to dst 7:
    chunk_3_7 = [
        c for c in plan if c.src_device_id == 3 and c.dst_device_id == 7
    ][0]
    self.assertEqual(chunk_3_7.src_slice, (112, 128, 0, 256))
    self.assertEqual(chunk_3_7.dst_slice, (0, 16, 768, 1024))
    self.assertEqual(chunk_3_7.shape, (16, 256))


if __name__ == "__main__":
  absltest.main()
