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

"""High-performance PyTorch KV Cache Manager for distributed vLLM pipelines."""

from typing import List, Optional

import torch

# Import Pybind11 dynamic binary extension E2E!
from api.torch import _kv_cache_manager


class KVCacheManager:
  """Zero-copy PyTorch KV Cache Manager."""

  def __init__(
      self,
      device_tensors: List[List[torch.Tensor]],
      block_size: int = 1,
      local_port: Optional[int] = None,
      host_blocks_to_allocate: Optional[int] = None,
      parallelism: int = 1,
  ):
    """Instantiates the PyTorch KV Cache Manager shims.

    Args:
      device_tensors: List of list of device-placed contiguous Tensors
        representing the sharded KV caches [layers, shards].
      block_size: Physical tokens count per allocation page block.
      local_port: TCP socket server port for remote pulls/pushes coordinates.
      host_blocks_to_allocate: Max host blocks staging cache size.
      parallelism: Parallel TCP sockets workers count.
    """
    self._impl = _kv_cache_manager.KVCacheManager(
        device_tensors,
        block_size,
        local_port,
        host_blocks_to_allocate,
        parallelism,
    )

  def h2d(
      self,
      stream_idx: int,
      peer: str,
      src_block_ids: List[int],
      dst_block_ids: List[int],
      entity_id: int = 0,
  ) -> None:
    """Asynchronously offloads host CPU staging pages to remote TPU devices."""
    self._impl.H2d(stream_idx, peer, src_block_ids, dst_block_ids, entity_id)

  def d2h(
      self,
      stream_idx: int,
      peer: str,
      src_block_ids: List[int],
      dst_block_ids: List[int],
      entity_id: int = 0,
  ) -> None:
    """Asynchronously fetches remote TPU device pages into local host CPU staging."""
    self._impl.D2h(stream_idx, peer, src_block_ids, dst_block_ids, entity_id)

  @property
  def local_port(self) -> Optional[int]:
    """Returns assigned ephemeral listener port coordinates."""
    return self._impl.local_port

  @property
  def num_layers(self) -> int:
    """Returns total layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns physical devices shards count."""
    return self._impl.num_shards

  @property
  def block_size(self) -> int:
    """Returns active page block size."""
    return self._impl.block_size

  @property
  def slice_byte_size(self) -> int:
    """Returns individual major dimension slice byte size capacity."""
    return self._impl.slice_byte_size
