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
"""High-performance worker-to-worker resharding engine for TPU Raiden."""

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from api.jax import resharding_planner
from api.jax import weight_synchronizer_ffi as raiden_ffi


def reshard_matrix(
    src_sharded_array: jax.Array,
    dst_sharding: jax.sharding.NamedSharding,
) -> jax.Array:
  """Performs optimal worker-to-worker resharding collective using global WeightSynchronizers.

  Directly copies memory blocks between worker CPU Host buffers over Loopback
  sockets, completely avoiding intermediate JAX array allocations and copies.

  Args:
    src_sharded_array: The JAX sharded array on the source mesh.
    dst_sharding: The target NamedSharding on the destination mesh.

  Returns:
    A new JAX sharded array on the target destination mesh.
  """
  global_shape = src_sharded_array.shape
  rows, cols = global_shape

  # Retrieve the complete logical global slice maps for both layouts
  src_map = src_sharded_array.sharding.devices_indices_map(global_shape)
  dst_map = dst_sharding.devices_indices_map(global_shape)

  # Canonical Sorting: Sort physical devices strictly by global coordinate
  # starts
  sorted_src_devices = sorted(
      src_sharded_array.sharding.addressable_devices,
      key=lambda d: (src_map[d][0].start or 0, src_map[d][1].start or 0),
  )
  sorted_dst_devices = sorted(
      dst_sharding.addressable_devices,
      key=lambda d: (dst_map[d][0].start or 0, dst_map[d][1].start or 0),
  )

  # Step 1: Initialize Global WeightSynchronizers E2E!
  # A. Allocate JAX single-device zeros for Destination
  dst_device_arrays = []
  for dev in sorted_dst_devices:
    row_slice, col_slice = dst_map[dev]
    row_start = row_slice.start if row_slice.start is not None else 0
    row_end = row_slice.stop if row_slice.stop is not None else rows
    col_start = col_slice.start if col_slice.start is not None else 0
    col_end = col_slice.stop if col_slice.stop is not None else cols

    # Create clean device-placed zeros
    zero_arr = jax.device_put(
        np.zeros((row_end - row_start, col_end - col_start), dtype=np.float32),
        dev,
    )
    dst_device_arrays.append(zero_arr)

  # B. Build JAX global sharded arrays E2E
  dst_sharded_array_initial = jax.make_array_from_single_device_arrays(
      global_shape, dst_sharding, dst_device_arrays
  )

  src_mesh = src_sharded_array.sharding.mesh
  src_mesh_shape = tuple(src_mesh.shape[name] for name in src_mesh.axis_names)
  src_global_ids = jnp.array(
      [d.id for d in src_mesh.devices.flatten()], dtype=jnp.int32
  ).reshape(src_mesh_shape)
  src_shard_idx = jax.device_put(
      src_global_ids,
      jax.sharding.NamedSharding(
          src_mesh, jax.sharding.PartitionSpec(*src_mesh.axis_names)
      ),
  )
  print(f"src_shard_idx shape: {src_shard_idx.shape}")

  dst_mesh = dst_sharding.mesh
  dst_mesh_shape = tuple(dst_mesh.shape[name] for name in dst_mesh.axis_names)
  dst_global_ids = jnp.array(
      [d.id for d in dst_mesh.devices.flatten()], dtype=jnp.int32
  ).reshape(dst_mesh_shape)
  dst_shard_idx = jax.device_put(
      dst_global_ids,
      jax.sharding.NamedSharding(
          dst_mesh, jax.sharding.PartitionSpec(*dst_mesh.axis_names)
      ),
  )

  src_slice_byte_size = src_sharded_array.addressable_shards[0].data.nbytes
  dst_slice_byte_size = dst_device_arrays[0].nbytes

  # Initialize WS for source
  src_ws_info = raiden_ffi.init_weight_synchronizer(
      device_array=src_sharded_array,
      shard_idx=src_shard_idx,
      mesh=src_mesh,
      slice_byte_size=src_slice_byte_size,
      parallelism=1,
      num_layers=1,
  )

  # Populate host buffers on source devices!
  raiden_ffi.weight_synchronizer_d2h(
      device_array=src_sharded_array,
      shard_idx=src_shard_idx,
      mesh=src_mesh,
  )

  # Initialize WS for destination
  dst_ws_info = raiden_ffi.init_weight_synchronizer(
      device_array=dst_sharded_array_initial,
      shard_idx=dst_shard_idx,
      mesh=dst_mesh,
      slice_byte_size=dst_slice_byte_size,
      parallelism=1,
      num_layers=1,
  )

  # TODO(agy): Support the scenario of multiple Pathways backends.
  # In the case of two independent clients, here is a sketch:
  # 1. Client A (source) gathers its local worker IPs/ports using
  #    process_allgather locally.
  # 2. Client A starts CoordinationServer and exposes this list.
  # 3. Client B (destination) connects via CoordinationClient to fetch the list.
  # 4. Client B passes the fetched list to execute_resharding FFI via replicated
  #    JAX arrays.

  local_src_ws_info = multihost_utils.global_array_to_host_local_array(
      src_ws_info,
      src_mesh,
      jax.sharding.PartitionSpec(src_mesh.axis_names[0], None),
  )

  gathered_src_ws_info = multihost_utils.process_allgather(
      local_src_ws_info
  ).reshape(-1, 5)

  # Step 3: Compute plan and flatten
  plan = resharding_planner.make_resharding_plan(
      global_shape=global_shape,
      src_sharding=src_sharded_array.sharding,
      dst_sharding=dst_sharding,
  )

  src_ips = []
  src_ports = []
  src_offsets = []
  dst_offsets = []
  sizes = []
  dst_device_ids = []

  for chunk in plan:
    logical_src_idx = chunk.src_device_id
    logical_dst_idx = chunk.dst_device_id

    src_dev = sorted_src_devices[logical_src_idx]
    dst_dev = sorted_dst_devices[logical_dst_idx]

    src_global_id = src_dev.id
    dst_global_id = dst_dev.id

    src_ip = gathered_src_ws_info[src_global_id, 0:4]
    src_port = gathered_src_ws_info[src_global_id, 4]

    r_start, r_end, c_start, c_end = chunk.src_slice
    dr_start, _, dc_start, _ = chunk.dst_slice
    chunk_width = c_end - c_start

    _, src_col_slice = src_map[src_dev]
    n_src = (src_col_slice.stop or cols) - (src_col_slice.start or 0)

    _, dst_col_slice = dst_map[dst_dev]
    n_dst = (dst_col_slice.stop or cols) - (dst_col_slice.start or 0)

    # Planner already returns LOCAL coordinates relative to the shard!
    for row in range(r_end - r_start):
      src_offset_bytes = ((r_start + row) * n_src + c_start) * 4
      dst_offset_bytes = ((dr_start + row) * n_dst + dc_start) * 4

      src_ips.extend(src_ip)
      src_ports.append(src_port)
      src_offsets.append(src_offset_bytes)
      dst_offsets.append(dst_offset_bytes)
      sizes.append(chunk_width * 4)
      dst_device_ids.append(dst_global_id)

  # Convert lists to JAX arrays
  src_ips_arr = jnp.array(src_ips, dtype=jnp.uint32)
  src_ports_arr = jnp.array(src_ports, dtype=jnp.int32)
  src_offsets_arr = jnp.array(src_offsets, dtype=jnp.int32)
  dst_offsets_arr = jnp.array(dst_offsets, dtype=jnp.int32)
  sizes_arr = jnp.array(sizes, dtype=jnp.int32)
  dst_device_ids_arr = jnp.array(dst_device_ids, dtype=jnp.int32)

  # Dispatch to FFI
  meta_sharding = jax.sharding.NamedSharding(
      dst_mesh, jax.sharding.PartitionSpec()
  )

  src_ips_arr = jax.device_put(src_ips_arr, meta_sharding)
  src_ports_arr = jax.device_put(src_ports_arr, meta_sharding)
  src_offsets_arr = jax.device_put(src_offsets_arr, meta_sharding)
  dst_offsets_arr = jax.device_put(dst_offsets_arr, meta_sharding)
  sizes_arr = jax.device_put(sizes_arr, meta_sharding)
  dst_device_ids_arr = jax.device_put(dst_device_ids_arr, meta_sharding)

  dst_sharded_array = raiden_ffi.execute_resharding(
      device_array=dst_sharded_array_initial,
      shard_idx=dst_shard_idx,
      src_ips=src_ips_arr,
      src_ports=src_ports_arr,
      src_offsets=src_offsets_arr,
      dst_offsets=dst_offsets_arr,
      sizes=sizes_arr,
      dst_device_ids=dst_device_ids_arr,
      mesh=dst_mesh,
  )

  return dst_sharded_array
