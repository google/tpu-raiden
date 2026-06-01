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

"""JAX bindings for WeightSynchronizer FFI, enabling host/device weight synchronization."""

import jax
from jax.experimental import compute_on
import jax.numpy as jnp
import numpy as np

from frameworks.jax import _weight_synchronizer_ffi


def init_weight_synchronizer(
    device_array,
    shard_idx,
    mesh,
    slice_byte_size: int = 0,
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int = 1,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer FFI custom call."""

  @compute_on.compute_on("device_host")
  def _local_init(anchor, s_idx):
    axis_names = mesh.axis_names
    out_shape = tuple([1] * len(axis_names)) + (5,)
    return jax.ffi.ffi_call(
        "init_weight_synchronizer",
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
        has_side_effect=True,
    )(
        anchor,
        s_idx,
        slice_byte_size=slice_byte_size,
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        num_layers=np.int32(num_layers),
    )

  axis_names = mesh.axis_names
  anchor_spec = jax.sharding.PartitionSpec(
      *axis_names, *([None] * (len(device_array.shape) - len(axis_names)))
  )
  index_spec = jax.sharding.PartitionSpec(*axis_names)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  return jax.shard_map(
      _local_init,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec),
      out_specs=out_spec,
  )(device_array, shard_idx)


def init_weight_synchronizer_and_d2h(
    device_array,
    shard_idx,
    mesh,
    slice_byte_size: int = 0,
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int = 1,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer_and_d2h FFI custom call."""

  @compute_on.compute_on("device_host")
  def _local_init_and_d2h(anchor, s_idx):
    axis_names = mesh.axis_names
    out_shape = tuple([1] * len(axis_names)) + (5,)
    return jax.ffi.ffi_call(
        "init_weight_synchronizer_and_d2h",
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
        has_side_effect=True,
    )(
        anchor,
        s_idx,
        slice_byte_size=slice_byte_size,
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        num_layers=np.int32(num_layers),
    )

  axis_names = mesh.axis_names
  anchor_spec = device_array.sharding.spec
  index_spec = jax.sharding.PartitionSpec(*axis_names)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  return jax.shard_map(
      _local_init_and_d2h,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec),
      out_specs=out_spec,
  )(device_array, shard_idx)


def execute_resharding(
    device_array: jax.Array,
    shard_idx: jax.Array,
    src_ips: jax.Array,
    src_ports: jax.Array,
    src_offsets: jax.Array,
    dst_offsets: jax.Array,
    sizes: jax.Array,
    dst_device_ids: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Executes resharding plan via FFI.

  Args:
    device_array: The destination sharded array to be updated.
    shard_idx: The shard index array (contains local device ID).
    src_ips: Replicated array of source IPs (uint32).
    src_ports: Replicated array of source ports (int32).
    src_offsets: Replicated array of source offsets (int32).
    dst_offsets: Replicated array of destination offsets (int32).
    sizes: Replicated array of sizes (int32).
    dst_device_ids: Replicated array of destination global device IDs (int32).
    mesh: The JAX Mesh.
  """
  print("Python execute_resharding called!")

  @compute_on.compute_on("device_host")
  def _local_execute(
      anchor, s_idx, s_ips, s_ports, s_offs, d_offs, szs, d_dev_ids
  ):
    return jax.ffi.ffi_call(
        "execute_resharding",
        jax.ShapeDtypeStruct(anchor.shape, anchor.dtype),
        has_side_effect=True,
        input_output_aliases={0: 0},
    )(anchor, s_idx, s_ips, s_ports, s_offs, d_offs, szs, d_dev_ids)

  meta_spec = jax.sharding.PartitionSpec()
  cache_spec = device_array.sharding.spec
  index_spec = jax.sharding.PartitionSpec(*mesh.axis_names)

  return jax.shard_map(
      _local_execute,
      mesh=mesh,
      in_specs=(
          cache_spec,
          index_spec,
          meta_spec,
          meta_spec,
          meta_spec,
          meta_spec,
          meta_spec,
          meta_spec,
      ),
      out_specs=cache_spec,
  )(
      device_array,
      shard_idx,
      src_ips,
      src_ports,
      src_offsets,
      dst_offsets,
      sizes,
      dst_device_ids,
  )


def prepare_extended_info(
    gathered_info, device_ids, r_starts, r_ends, c_starts, c_ends
):
  """Packs metadata for coordination."""
  return _weight_synchronizer_ffi.prepare_extended_info(
      gathered_info, device_ids, r_starts, r_ends, c_starts, c_ends
  )


def destroy_weight_synchronizer():
  """Cleans up WeightSynchronizer instances."""
  _weight_synchronizer_ffi.destroy_weight_synchronizer()
