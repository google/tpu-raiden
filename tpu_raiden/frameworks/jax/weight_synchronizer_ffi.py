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

from tpu_raiden.frameworks.jax import _weight_synchronizer_ffi


def make_logical_shard_idx(mesh):
  """Canonical logical shard index per device (mesh row-major == controller
  itertools.product order). Build once at setup, pass to the calls below to keep
  it off the hot path."""
  index_spec = jax.sharding.PartitionSpec(*mesh.axis_names)
  return jax.device_put(
      jnp.arange(mesh.size, dtype=jnp.int32).reshape(mesh.devices.shape),
      jax.sharding.NamedSharding(mesh, index_spec))


def init_weight_synchronizer(
    device_array,
    shard_idx,
    mesh,
    slice_byte_size: int = 0,
    local_port: int = 0,
    parallelism: int = 1,
    num_layers: int = 1,
    listener_port: int = -1,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer FFI custom call on each device rank.

  Args:
    device_array: Sharded input device array serving as the FFI target anchor.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    slice_byte_size: Byte size of the physical weight slice to synchronize per
      shard.
    local_port: Port number for the local sender transport server (`0` for
      auto-assign).
    parallelism: Number of parallel TCP/IPC streams to use for memory transfer.
    num_layers: Total number of network layers or synchronization iterations.
    listener_port: Optional pre-assigned listener port to share across ranks
      (`-1` to initialize new).

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=6` if
    `listener_port >= 0`).
  """

  @compute_on.compute_on("device_host")
  def _local_init(anchor, s_idx):
    axis_names = mesh.axis_names
    out_dim = 6 if listener_port >= 0 else 5
    out_shape = tuple([1] * len(axis_names)) + (out_dim,)
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
        listener_port=np.int32(listener_port),
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
    listener_port: int = -1,
    logical_shard_idx=None,
) -> jax.Array:
  """Registers and executes init_weight_synchronizer_and_d2h FFI custom call on each device rank.

  Args:
    device_array: Sharded input device array serving as the FFI target anchor.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.
    slice_byte_size: Byte size of the physical weight slice to synchronize per
      shard.
    local_port: Port number for the local sender transport server (`0` for
      auto-assign).
    parallelism: Number of parallel TCP/IPC streams to use for memory transfer.
    num_layers: Total number of network layers or synchronization iterations.
    listener_port: Optional pre-assigned listener port to share across ranks
      (`-1` to initialize new).

  Returns:
    A sharded 1D int32 array containing synchronization metadata (`out_dim=6` if
    `listener_port >= 0`).
  """

  @compute_on.compute_on("device_host")
  def _local_init_and_d2h(anchor, s_idx, l_idx):
    axis_names = mesh.axis_names
    out_dim = 6 if listener_port >= 0 else 5
    out_shape = tuple([1] * len(axis_names)) + (out_dim,)
    return jax.ffi.ffi_call(
        "init_weight_synchronizer_and_d2h",
        jax.ShapeDtypeStruct(out_shape, jnp.int32),
        has_side_effect=True,
    )(
        anchor,
        s_idx,
        l_idx,
        slice_byte_size=slice_byte_size,
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        num_layers=np.int32(num_layers),
        listener_port=np.int32(listener_port),
    )

  if logical_shard_idx is None:
    logical_shard_idx = make_logical_shard_idx(mesh)

  axis_names = mesh.axis_names
  anchor_spec = device_array.sharding.spec
  index_spec = jax.sharding.PartitionSpec(*axis_names)
  out_spec = jax.sharding.PartitionSpec(*axis_names, None)

  return jax.shard_map(
      _local_init_and_d2h,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec, index_spec),
      out_specs=out_spec,
  )(device_array, shard_idx, logical_shard_idx)





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


def is_listener_active(shard_idx: int = 0) -> bool:
  """Returns whether the native C++ listener for the shard is active."""
  return _weight_synchronizer_ffi.is_listener_active(shard_idx)


def h2d(device_array, shard_idx, mesh, logical_shard_idx=None) -> jax.Array:
  """Executes asynchronous Host-to-Device (H2D) copy from local staging buffer directly onto device memory via FFI.

  Args:
    device_array: Sharded destination device array onto which host memory is
      copied.
    shard_idx: Sharding index array representing shard IDs on each local rank.
    mesh: JAX device mesh across all participating physical devices/hosts.

  Returns:
    The updated sharded device array with data copied from the local host
    buffer.
  """

  @compute_on.compute_on("device_host")
  def _local_h2d(anchor, s_idx, l_idx):
    return jax.ffi.ffi_call(
        "ws_h2d",
        jax.ShapeDtypeStruct(anchor.shape, anchor.dtype),
        has_side_effect=True,
    )(anchor, s_idx, l_idx)

  if logical_shard_idx is None:
    logical_shard_idx = make_logical_shard_idx(mesh)

  axis_names = mesh.axis_names
  anchor_spec = device_array.sharding.spec
  index_spec = jax.sharding.PartitionSpec(*axis_names)

  return jax.shard_map(
      _local_h2d,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec, index_spec),
      out_specs=anchor_spec,
  )(device_array, shard_idx, logical_shard_idx)
