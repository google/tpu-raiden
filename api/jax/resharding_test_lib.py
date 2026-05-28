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
"""Prototyped controller-piped resharding test library for TPU Raiden."""

import jax
import numpy as np
from api.jax import resharding_planner


def reshard_matrix_controller_piped(
    src_sharded_array: jax.Array,
    dst_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Performs a sharded axis transition resharding collective piping all data through client."""
  global_shape = src_sharded_array.shape
  K, N = global_shape

  src_devices = src_sharded_array.sharding.device_set
  dst_devices = dst_sharding.device_set
  src_count = len(src_devices)
  dst_count = len(dst_devices)

  src_map = src_sharded_array.sharding.devices_indices_map(global_shape)
  dst_map = dst_sharding.devices_indices_map(global_shape)

  sorted_src_devices = sorted(
      src_sharded_array.sharding.addressable_devices,
      key=lambda d: (src_map[d][0].start or 0, src_map[d][1].start or 0),
  )
  sorted_dst_devices = sorted(
      dst_sharding.addressable_devices,
      key=lambda d: (dst_map[d][0].start or 0, dst_map[d][1].start or 0),
  )

  plan = resharding_planner.make_resharding_plan(
      global_shape=global_shape,
      src_sharding=src_sharded_array.sharding,
      dst_sharding=dst_sharding,
  )

  src_host_shards = {}
  for shard in src_sharded_array.addressable_shards:
    dev_idx = sorted_src_devices.index(shard.device)
    src_host_shards[dev_idx] = np.asarray(shard.data)

  dst_host_shards = {}
  for dev in dst_sharding.addressable_devices:
    j = sorted_dst_devices.index(dev)
    row_slice, col_slice = dst_map[dev]
    row_start = row_slice.start if row_slice.start is not None else 0
    row_end = row_slice.stop if row_slice.stop is not None else K
    col_start = col_slice.start if col_slice.start is not None else 0
    col_end = col_slice.stop if col_slice.stop is not None else N
    dst_host_shards[j] = np.zeros(
        (row_end - row_start, col_end - col_start),
        dtype=src_sharded_array.dtype,
    )

  for chunk in plan:
    i = chunk.src_device_id
    j = chunk.dst_device_id

    r_start, r_end, c_start, c_end = chunk.src_slice
    sub_block = src_host_shards[i][r_start:r_end, c_start:c_end]

    dr_start, dr_end, dc_start, dc_end = chunk.dst_slice
    dst_host_shards[j][dr_start:dr_end, dc_start:dc_end] = sub_block

  dst_device_arrays = []
  for dev in dst_sharding.addressable_devices:
    j = sorted_dst_devices.index(dev)
    dev_arr = jax.device_put(dst_host_shards[j], dev)
    dst_device_arrays.append(dev_arr)

  dst_sharded_array = jax.make_array_from_single_device_arrays(
      global_shape, dst_sharding, dst_device_arrays
  )

  return dst_sharded_array
