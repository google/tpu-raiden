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
"""Lightweight generalized 2D block resharding planner for TPU Raiden."""

from dataclasses import dataclass
from typing import List, Tuple
import jax


@dataclass
class ReshardChunk:
  src_device_id: int
  dst_device_id: int
  # Slices represented as (row_start, row_end, col_start, col_end)
  src_slice: Tuple[int, int, int, int]
  dst_slice: Tuple[int, int, int, int]
  shape: Tuple[int, int]


def make_resharding_plan(
    global_shape: Tuple[int, int],
    src_sharding: jax.sharding.NamedSharding,
    dst_sharding: jax.sharding.NamedSharding,
) -> List[ReshardChunk]:
  """Generates the complete 2D sub-block transfer plan between two arbitrary JAX NamedShardings.

  Args:
    global_shape: The global 2D shape [K, N] of the weight matrix.
    src_sharding: The source NamedSharding specification.
    dst_sharding: The destination NamedSharding specification.

  Returns:
    A list of ReshardChunk descriptions mapping source device slices to
    destination device offsets.
  """
  K, N = global_shape

  # Retrieve the complete logical global slice maps for both layouts
  src_map = src_sharding.devices_indices_map(global_shape)
  dst_map = dst_sharding.devices_indices_map(global_shape)

  # Canonical Sorting: Sort addressable devices strictly by global row/col slice starts!
  # This maps physical device placements to deterministic logical index coordinates.
  src_devices = sorted(
      src_sharding.addressable_devices,
      key=lambda d: (src_map[d][0].start or 0, src_map[d][1].start or 0),
  )
  dst_devices = sorted(
      dst_sharding.addressable_devices,
      key=lambda d: (dst_map[d][0].start or 0, dst_map[d][1].start or 0),
  )

  # TODO(b/12345678): Support arbitrary TPU hardware memory tiling alignments
  # (e.g. padding sub-blocks to multiples of 8/32/128 bytes) to avoid padding
  # degradation during transfer. Currently we assume K and N are sufficiently
  # large and divisible.

  plan = []

  # Loop through all combinations of source and destination devices
  for i, src_dev in enumerate(src_devices):
    # Get global coordinates of source shard i
    src_row_slice, src_col_slice = src_map[src_dev]
    src_row_start = (
        src_row_slice.start if src_row_slice.start is not None else 0
    )
    src_row_end = src_row_slice.stop if src_row_slice.stop is not None else K
    src_col_start = (
        src_col_slice.start if src_col_slice.start is not None else 0
    )
    src_col_end = src_col_slice.stop if src_col_slice.stop is not None else N

    for j, dst_dev in enumerate(dst_devices):
      # Get global coordinates of destination shard j
      dst_row_slice, dst_col_slice = dst_map[dst_dev]
      dst_row_start = (
          dst_row_slice.start if dst_row_slice.start is not None else 0
      )
      dst_row_end = dst_row_slice.stop if dst_row_slice.stop is not None else K
      dst_col_start = (
          dst_col_slice.start if dst_col_slice.start is not None else 0
      )
      dst_col_end = dst_col_slice.stop if dst_col_slice.stop is not None else N

      # Find the intersection bounds in global coordinate space
      intersect_row_start = max(src_row_start, dst_row_start)
      intersect_row_end = min(src_row_end, dst_row_end)
      intersect_col_start = max(src_col_start, dst_col_start)
      intersect_col_end = min(src_col_end, dst_col_end)

      # Record transfer chunk if there is a non-empty overlap
      if (
          intersect_row_start < intersect_row_end
          and intersect_col_start < intersect_col_end
      ):
        chunk_shape = (
            intersect_row_end - intersect_row_start,
            intersect_col_end - intersect_col_start,
        )

        # Map intersection bounds relative to source i's local shard buffer
        local_src_row_start = intersect_row_start - src_row_start
        local_src_row_end = intersect_row_end - src_row_start
        local_src_col_start = intersect_col_start - src_col_start
        local_src_col_end = intersect_col_end - src_col_start

        # Map intersection bounds relative to destination j's local shard buffer
        local_dst_row_start = intersect_row_start - dst_row_start
        local_dst_row_end = intersect_row_end - dst_row_start
        local_dst_col_start = intersect_col_start - dst_col_start
        local_dst_col_end = intersect_col_end - dst_col_start

        plan.append(
            ReshardChunk(
                src_device_id=i,
                dst_device_id=j,
                src_slice=(
                    local_src_row_start,
                    local_src_row_end,
                    local_src_col_start,
                    local_src_col_end,
                ),
                dst_slice=(
                    local_dst_row_start,
                    local_dst_row_end,
                    local_dst_col_start,
                    local_dst_col_end,
                ),
                shape=chunk_shape,
            )
        )

  return plan
