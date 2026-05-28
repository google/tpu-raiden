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

"""High-performance JAX KV Cache Manager for trainer-inference pipelines."""

from typing import Any, List, Optional

# Import JAX Nanobind binary extension E2E!
from api.jax import _kv_cache_manager


class KVCacheManager:
  """Zero-copy JAX KV Cache Manager."""

  def __init__(
      self,
      device_arrays: List[Any],
      block_size: int = 1,
      local_port: Optional[int] = None,
      host_blocks_to_allocate: Optional[int] = None,
      external_host_ptrs: Optional[List[int]] = None,
      unsafe_skip_buffer_lock: bool = False,
      parallelism: int = 1,
  ):
    """Instantiates the JAX KV Cache Manager shims.

    Args:
      device_arrays: List of physical sharded JAX device PJRT arrays
        representing the sharded KV caches.
      block_size: Physical tokens count per allocation page block.
      local_port: TCP socket server port for remote pulls/pushes coordinates.
      host_blocks_to_allocate: Max host blocks staging cache size.
      external_host_ptrs: Pinned external CPU host memory addresses list.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
      parallelism: Parallel TCP sockets workers count.
    """
    self._impl = _kv_cache_manager.KVCacheManager(
        device_arrays,
        block_size,
        local_port,
        host_blocks_to_allocate,
        external_host_ptrs,
        unsafe_skip_buffer_lock,
        parallelism,
    )

  def h2d(
      self,
      src_offsets_major_dim: List[int] = [],
      dst_offsets_major_dim: List[int] = [],
      copy_sizes_major_dim: List[int] = [],
  ) -> Any:
    """Triggers host-to-device copy of cache slices."""
    return self._impl.h2d(
        src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim
    )

  def d2h(
      self,
      src_offsets_major_dim: List[int] = [],
      dst_offsets_major_dim: List[int] = [],
      copy_sizes_major_dim: List[int] = [],
  ) -> Any:
    """Triggers device-to-host copy of cache slices."""
    return self._impl.d2h(
        src_offsets_major_dim, dst_offsets_major_dim, copy_sizes_major_dim
    )

  def d2h_auto_allocate(
      self,
      src_offsets_major_dim: List[int] = [],
      copy_sizes_major_dim: List[int] = [],
      entity_id: int = 0,
  ) -> tuple[List[int], Any]:
    """Triggers device-to-host copy with automatic host buffer allocation."""
    return self._impl.d2h_auto_allocate(
        src_offsets_major_dim, copy_sizes_major_dim, entity_id
    )

  def h2h_write(
      self,
      peer: str,
      src_block_ids: List[int],
      entity_id: int = 0,
  ) -> tuple[List[int], Any]:
    """Trainer pushes cache pages to a remote peer inference server."""
    return self._impl.h2h_write(peer, src_block_ids, entity_id)

  def h2h_read(
      self,
      peer: str,
      src_block_ids: List[int],
      entity_id: int = 0,
  ) -> tuple[List[int], Any]:
    """Inference pulls cache pages from a remote peer trainer source."""
    return self._impl.h2h_read(peer, src_block_ids, entity_id)

  def local_port(self) -> Optional[int]:
    """Returns assigned ephemeral listener port coordinates."""
    return self._impl.local_port()

  @property
  def num_layers(self) -> int:
    """Returns total layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns physical JAX devices shards count."""
    return self._impl.num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns individual major dimension slice byte size capacity."""
    return self._impl.slice_byte_size
