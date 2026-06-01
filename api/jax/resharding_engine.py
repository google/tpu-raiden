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

from typing import Optional, Tuple
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from api.jax import resharding_planner
from api.jax import weight_synchronizer_ffi as raiden_ffi
from rpc import coordination_helper

_COORDINATION_SERVER: Optional[coordination_helper.CoordinationServer] = None

DEFAULT_IPV4_ADDRESS = "10.255.255.255"
DEFAULT_IPV6_ADDRESS = "2001:4860:4860::8888"
DEFAULT_HOST_IP = "127.0.0.1"


def prepare_reshard(src_sharded_array: jax.Array) -> str:
  """Prepares source array for resharding and returns coordination address.

  This is called by the source client.

  Args:
    src_sharded_array: The JAX sharded array on the source mesh.

  Returns:
    The coordination address for the destination client to connect to.
  """
  global _COORDINATION_SERVER

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

  src_slice_byte_size = src_sharded_array.addressable_shards[0].data.nbytes

  src_ws_info = raiden_ffi.init_weight_synchronizer_and_d2h(
      device_array=src_sharded_array,
      shard_idx=src_shard_idx,
      mesh=src_mesh,
      slice_byte_size=src_slice_byte_size,
      parallelism=1,
      num_layers=1,
  )

  local_src_ws_info = multihost_utils.global_array_to_host_local_array(
      src_ws_info,
      src_mesh,
      jax.sharding.PartitionSpec(src_mesh.axis_names[0], None),
  )

  gathered_src_ws_info = multihost_utils.process_allgather(
      local_src_ws_info
  ).reshape(-1, 5)

  if jax.process_index() == 0:
    _COORDINATION_SERVER = coordination_helper.CoordinationServer()
    port = _COORDINATION_SERVER.start()
    import socket

    try:
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      s.connect((DEFAULT_IPV4_ADDRESS, 1))
      host_ip = s.getsockname()[0]
      s.close()
    except Exception:
      try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect((DEFAULT_IPV6_ADDRESS, 1))
        host_ip = s.getsockname()[0]
        s.close()
      except Exception:
        host_ip = DEFAULT_HOST_IP

    gathered_src_ws_info_np = np.ascontiguousarray(
        gathered_src_ws_info, dtype=np.int32
    )

    src_map = src_sharded_array.sharding.devices_indices_map(
        src_sharded_array.shape
    )

    devices = src_mesh.devices.flatten()
    device_ids = [int(d.id) for d in devices]
    r_starts = [
        int(src_map[d][0].start if src_map[d][0].start is not None else 0)
        for d in devices
    ]
    r_ends = [
        int(
            src_map[d][0].stop
            if src_map[d][0].stop is not None
            else src_sharded_array.shape[0]
        )
        for d in devices
    ]
    c_starts = [
        int(src_map[d][1].start if src_map[d][1].start is not None else 0)
        for d in devices
    ]
    c_ends = [
        int(
            src_map[d][1].stop
            if src_map[d][1].stop is not None
            else src_sharded_array.shape[1]
        )
        for d in devices
    ]

    gathered_src_ws_info_list = gathered_src_ws_info_np.flatten().tolist()
    extended_info = raiden_ffi.prepare_extended_info(
        gathered_src_ws_info_list,
        device_ids,
        r_starts,
        r_ends,
        c_starts,
        c_ends,
    )

    _COORDINATION_SERVER.set_metadata(
        port=0, block_ids=extended_info, host_ip=host_ip
    )
    if ":" in host_ip:
      address = f"[{host_ip}]:{port}"
    else:
      address = f"{host_ip}:{port}"

    address_bytes = np.zeros(64, dtype=np.uint8)
    address_bytes[: len(address)] = np.frombuffer(
        address.encode(), dtype=np.uint8
    )
  else:
    address_bytes = np.zeros(64, dtype=np.uint8)

  broadcasted_address_bytes = multihost_utils.broadcast_one_to_all(
      address_bytes, is_source=(jax.process_index() == 0)
  )

  address = broadcasted_address_bytes.tobytes().decode().strip("\x00")
  return address


def reshard(
    global_shape: Tuple[int, int],
    dst_sharding: jax.sharding.NamedSharding,
    coordination_address: str,
) -> jax.Array:
  """Performs resharding on destination client using coordination address."""

  num_src_devices_arr = jnp.array([0], dtype=jnp.int32)
  if jax.process_index() == 0:
    client = coordination_helper.CoordinationClient(coordination_address)
    _, block_ids, _ = client.get_metadata()
    # Each device has 10 elements: 4 IPs, 1 port, 1 device_id, 4 slice indices
    num_src_devices = len(block_ids) // 10
    num_src_devices_arr = jnp.array([num_src_devices], dtype=jnp.int32)

  num_src_devices_arr = multihost_utils.broadcast_one_to_all(
      num_src_devices_arr, is_source=(jax.process_index() == 0)
  )
  num_src_devices = int(num_src_devices_arr[0])
  expected_size = num_src_devices * 10

  if jax.process_index() == 0:
    assert (
        len(block_ids) == expected_size
    ), f"Expected {expected_size} ints, got {len(block_ids)}"
    gathered_src_ws_info = np.array(block_ids, dtype=np.int32).reshape(-1, 10)
  else:
    gathered_src_ws_info = np.zeros((num_src_devices, 10), dtype=np.int32)

  gathered_src_ws_info = multihost_utils.broadcast_one_to_all(
      gathered_src_ws_info, is_source=(jax.process_index() == 0)
  )

  rows, cols = global_shape
  dst_map = dst_sharding.devices_indices_map(global_shape)

  sorted_dst_devices = sorted(
      dst_sharding.addressable_devices,
      key=lambda d: (dst_map[d][0].start or 0, dst_map[d][1].start or 0),
  )

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

  dst_slice_byte_size = dst_device_arrays[0].nbytes

  # Initialize WS for destination
  dst_ws_info = raiden_ffi.init_weight_synchronizer(
      device_array=dst_sharded_array_initial,
      shard_idx=dst_shard_idx,
      mesh=dst_mesh,
      slice_byte_size=dst_slice_byte_size,
      parallelism=1,
      num_layers=1,
  )

  # Step 3: Compute plan
  # Extract sharding info from gathered_src_ws_info
  # gathered_src_ws_info has rows: [ip0, ip1, ip2, ip3, port, device_id, r_start, r_end, c_start, c_end]
  src_metadata = []
  for row in gathered_src_ws_info:
    dev_id = int(row[5])
    r_start = int(row[6])
    r_end = int(row[7])
    c_start = int(row[8])
    c_end = int(row[9])
    src_metadata.append((dev_id, r_start, r_end, c_start, c_end))

  plan = resharding_planner.make_resharding_plan_from_metadata(
      global_shape=global_shape,
      src_metadata=src_metadata,
      dst_sharding=dst_sharding,
  )

  src_ips = []
  src_ports = []
  src_offsets = []
  dst_offsets = []
  sizes = []
  dst_device_ids = []

  for chunk in plan:
    src_global_id = chunk.src_device_id
    src_row = next(r for r in gathered_src_ws_info if r[5] == src_global_id)
    n_src = int(src_row[9]) - int(src_row[8])

    dst_global_id = chunk.dst_device_id
    dst_dev = next(
        d for d in dst_sharding.mesh.devices.flatten() if d.id == dst_global_id
    )
    _, dst_col_slice = dst_map[dst_dev]
    n_dst = (dst_col_slice.stop or cols) - (dst_col_slice.start or 0)

    # Find row in gathered_src_ws_info for IP/Port lookup
    src_row = next(r for r in gathered_src_ws_info if r[5] == src_global_id)
    src_ip = src_row[0:4]
    src_port = src_row[4]

    r_start, r_end, c_start, c_end = chunk.src_slice
    dr_start, _, dc_start, _ = chunk.dst_slice
    chunk_width = c_end - c_start

    for row in range(r_end - r_start):
      src_offset_bytes = ((r_start + row) * n_src + c_start) * 4
      dst_offset_bytes = ((dr_start + row) * n_dst + dc_start) * 4

      src_ips.extend(src_ip)
      src_ports.append(src_port)
      src_offsets.append(src_offset_bytes)
      dst_offsets.append(dst_offset_bytes)
      sizes.append(chunk_width * 4)
      dst_device_ids.append(dst_global_id)

  src_ips_arr = jnp.array(src_ips, dtype=jnp.uint32)
  src_ports_arr = jnp.array(src_ports, dtype=jnp.int32)
  src_offsets_arr = jnp.array(src_offsets, dtype=jnp.int32)
  dst_offsets_arr = jnp.array(dst_offsets, dtype=jnp.int32)
  sizes_arr = jnp.array(sizes, dtype=jnp.int32)
  dst_device_ids_arr = jnp.array(dst_device_ids, dtype=jnp.int32)

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

  if jax.process_index() == 0:
    client = coordination_helper.CoordinationClient(coordination_address)
    client.shutdown()

  return dst_sharded_array


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
  address = prepare_reshard(src_sharded_array)
  return reshard(
      src_sharded_array.shape,
      dst_sharding,
      address,
  )
