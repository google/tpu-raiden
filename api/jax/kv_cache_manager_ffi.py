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

"""JAX bindings for Raiden FFI, enabling host/device KV cache management.

This module provides Python functions that wrap C++ FFI calls to manage
KV caches on TPU devices, including initialization, host-to-device (h2d),
device-to-host (d2h) transfers, and synchronization.
"""

import functools
import sys

import jax
from jax.experimental import compute_on
import jax.numpy as jnp
import numpy as np

from api.jax import _kv_cache_manager_ffi


def init(
    device_array,  # Single sharded device array (Pathways / PJRT mesh!)
    shard_idx,  # Isolated JAX device shard index E2E!
    mesh,  # Dynamic sharded JAX Mesh coordinating the multi-device layout E2E!
    slice_byte_size: int = 0,
    block_size: int = 1,
    local_port: int = 0,
    parallelism: int = 1,
    host_blocks_to_allocate: int = 0,
    num_layers: int = 1,
) -> jax.Array:
  """Registers and executes init_kv_cache_manager FFI custom call to instantiate the manager directly on worker C++ heap E2E."""

  @compute_on.compute_on("device_host")
  def _local_init(anchor, s_idx):
    return jax.ffi.ffi_call(
        "init_kv_cache_manager",
        jax.ShapeDtypeStruct(anchor.shape, anchor.dtype),
        has_side_effect=False,
        input_output_aliases={
            0: 0
        },  # In-place alias preserves the cache values on-chip perfectly!
    )(
        anchor,  # Positional argument 0: anchors the FFI execution on TPU E2E!
        s_idx,  # Positional argument 1: isolated device index E2E!
        slice_byte_size=slice_byte_size,
        block_size=np.int32(block_size),
        local_port=np.int32(local_port),
        parallelism=np.int32(parallelism),
        host_blocks_to_allocate=np.int32(host_blocks_to_allocate),
        num_layers=np.int32(num_layers),
    )

  axis_name = mesh.axis_names[0]  # Resolve partition axis name dynamically E2E!
  anchor_spec = jax.sharding.PartitionSpec(axis_name, None, None, None)
  index_spec = jax.sharding.PartitionSpec(axis_name)

  return jax.shard_map(
      _local_init,
      mesh=mesh,
      in_specs=(anchor_spec, index_spec),
      out_specs=anchor_spec,
  )(device_array, shard_idx)


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "layer_idx"),
)
def h2d(
    cache_data: jax.Array,
    layer_idx: int,
    src_offsets: jax.Array,
    dst_offsets: jax.Array,
    copy_sizes: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Asynchronously copies multiple sharded chunks from host CPU memory to TPU device memory for a SINGLE layer."""
  num_devices = mesh.size
  local_device_count = num_devices // jax.process_count()
  global_indices = jnp.arange(num_devices, dtype=jnp.int32)
  local_indices = global_indices % local_device_count
  axis_name = mesh.axis_names[0]  # Resolve partition axis name dynamically E2E!
  shard_indices = jax.device_put(
      local_indices,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name)),
  )

  # Capture layer_idx statically inside the local eager closure scope
  @compute_on.compute_on("device_host")
  def _local_h2d(s_off, d_off, c_sz, s_idx, cache_slice):
    return jax.ffi.ffi_call(
        "h2d",
        jax.ShapeDtypeStruct(
            cache_slice.shape, cache_slice.dtype
        ),  # Matches cache slice!
        input_output_aliases={
            4: 0
        },  # Bind Output 0 as in-place alias of Input 4 (cache_slice)!
        has_side_effect=True,
    )(
        s_off,
        d_off,
        c_sz,
        s_idx,
        cache_slice,  # Pass the physical device cache buffer slice dynamically!
        layer_idx=np.int32(
            layer_idx
        ),  # Passed statically as dynamic named attribute!
    )

  meta_spec = jax.sharding.PartitionSpec()  # Fully replicated metadata!
  index_spec = jax.sharding.PartitionSpec(axis_name)
  cache_spec = jax.sharding.PartitionSpec(axis_name, None, None, None)

  # Distribute FFI custom call across devices using shard_map
  return jax.shard_map(
      _local_h2d,
      mesh=mesh,
      in_specs=(
          meta_spec,
          meta_spec,
          meta_spec,
          index_spec,
          cache_spec,  # Sharded layout of cache buffer slice
      ),
      out_specs=cache_spec,  # Output shares identical sharded layout!
  )(src_offsets, dst_offsets, copy_sizes, shard_indices, cache_data)


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "layer_idx"),
)
def d2h(
    # [global_blocks, block_size, head_count, head_dim] (Single Layer!)
    cache_data: jax.Array,
    layer_idx: int,  # Static Python int representing the layer index!
    src_offsets: jax.Array,
    dst_offsets: jax.Array,
    copy_sizes: jax.Array,
    mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Asynchronously copies multiple sharded chunks from TPU device memory to host CPU memory for a SINGLE layer."""
  num_devices = mesh.size
  local_device_count = num_devices // jax.process_count()
  global_indices = jnp.arange(num_devices, dtype=jnp.int32)
  local_indices = global_indices % local_device_count
  axis_name = mesh.axis_names[0]  # Resolve partition axis name dynamically E2E!
  shard_indices = jax.device_put(
      local_indices,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name)),
  )

  # Capture layer_idx statically inside the local eager closure scope
  @compute_on.compute_on("device_host")
  def _local_d2h(s_off, d_off, c_sz, s_idx, cache_slice):
    return jax.ffi.ffi_call(
        "d2h",
        jax.ShapeDtypeStruct(
            cache_slice.shape, cache_slice.dtype
        ),  # Matches cache slice!
        input_output_aliases={
            4: 0
        },  # Bind Output 0 as in-place alias of Input 4 (cache_slice)!
        has_side_effect=True,
    )(
        s_off,
        d_off,
        c_sz,
        s_idx,
        cache_slice,  # Pass the physical device cache buffer slice dynamically!
        layer_idx=np.int32(
            layer_idx
        ),  # Passed statically as dynamic named attribute!
    )

  meta_spec = jax.sharding.PartitionSpec()  # Fully replicated metadata!
  index_spec = jax.sharding.PartitionSpec(axis_name)
  cache_spec = jax.sharding.PartitionSpec(axis_name, None, None, None)

  return jax.shard_map(
      _local_d2h,
      mesh=mesh,
      in_specs=(
          meta_spec,
          meta_spec,
          meta_spec,
          index_spec,
          cache_spec,  # Sharded layout of cache buffer slice
      ),
      out_specs=cache_spec,  # Output shares identical sharded layout!
  )(src_offsets, dst_offsets, copy_sizes, shard_indices, cache_data)


def sync_copies() -> None:
  """Synchronizes and awaits all outstanding background physical TPU PCIe DMA copies eagerly from Python."""
  _kv_cache_manager_ffi.sync_copies()


def destroy_kv_cache() -> None:
  """Safely releases and destroys the C++ heap-allocated global KVCacheManager instance."""
  _kv_cache_manager_ffi.destroy_kv_cache()
