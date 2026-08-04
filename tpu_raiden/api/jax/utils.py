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
"""Utility functions for JAX Weight Synchronizer."""

from typing import List
import jax
import numpy as np


def get_shard_sorting_permutation(arr: jax.Array) -> List[int]:
  """Computes the permutation to sort JAX shards for Raiden Controller."""
  sharding = arr.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return list(range(len(arr.addressable_shards)))

  mesh = sharding.mesh
  spec = sharding.spec

  # 1. Reconstruct logical_mesh_shape and layout
  logical_mesh_shape = [mesh.shape[ax] for ax in mesh.axis_names]

  sharded_axes = []
  for axis in spec:
    if axis is None:
      continue
    if isinstance(axis, str):
      sharded_axes.append(axis)
    else:
      sharded_axes.extend(axis)

  major_to_minor = sorted([mesh.axis_names.index(ax) for ax in sharded_axes])

  # 2. Compute controller's expected global indices
  num_shards = len(arr.addressable_shards)
  num_physical_hosts = jax.process_count()
  replica_id = jax.process_index()

  phys_mesh = [logical_mesh_shape[d] for d in major_to_minor]

  host_axis_logical = None
  for d, size in enumerate(logical_mesh_shape):
    if size == num_physical_hosts:
      host_axis_logical = d
      break

  non_host_axes = [
      d for d in range(len(logical_mesh_shape)) if d != host_axis_logical
  ]

  controller_global_indices = []
  for j in range(num_shards):
    local_coords = {}
    temp = j
    for d in reversed(non_host_axes):
      size = logical_mesh_shape[d]
      local_coords[d] = temp % size
      temp = temp // size

    full_coords = [0] * len(logical_mesh_shape)
    for d in range(len(logical_mesh_shape)):
      if d == host_axis_logical:
        full_coords[d] = replica_id
      else:
        full_coords[d] = local_coords.get(d, 0)

    tensor_coords = [full_coords[m_axis] for m_axis in major_to_minor]

    global_idx = 0
    stride = 1
    for val, size in zip(reversed(tensor_coords), reversed(phys_mesh)):
      global_idx += val * stride
      stride *= size
    controller_global_indices.append(global_idx)

  # 3. Compute ACTUAL JAX global shard indices
  jax_shard_global_indices = []
  for shard in arr.addressable_shards:
    device = shard.device
    coords = np.argwhere(mesh.devices == device)
    if coords.size == 0:
      raise ValueError(f"Device {device} not found in mesh")
    m_coords = coords[0]
    full_coords = list(m_coords)

    # Map array dimensions to mesh axes using spec
    tensor_coords = []
    tensor_shape = []
    for axis in spec:
      if axis is None:
        tensor_coords.append(0)
        tensor_shape.append(1)
      elif isinstance(axis, str):
        tensor_coords.append(full_coords[mesh.axis_names.index(axis)])
        tensor_shape.append(mesh.shape[axis])
      else:
        for ax in axis:
          tensor_coords.append(full_coords[mesh.axis_names.index(ax)])
          tensor_shape.append(mesh.shape[ax])

    # Compute flat index (row-major)
    global_idx = 0
    stride = 1
    for val, size in zip(reversed(tensor_coords), reversed(tensor_shape)):
      global_idx += val * stride
      stride *= size
    jax_shard_global_indices.append(global_idx)

  # 4. Sort indices
  indices = list(range(len(arr.addressable_shards)))

  def sort_key(idx):
    g = jax_shard_global_indices[idx]
    occurrence = jax_shard_global_indices[:idx].count(g)
    matching_indices = [
        i for i, val in enumerate(controller_global_indices) if val == g
    ]
    if occurrence < len(matching_indices):
      target_j = matching_indices[occurrence]
    else:
      target_j = matching_indices[-1]
    return target_j

  sorted_indices = sorted(indices, key=sort_key)
  return sorted_indices
