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
"""Host-only Python wrapper for experimental Raiden KV cache tests."""

from __future__ import annotations

from typing import Optional

from tpu_raiden.api.torch import kv_cache_manager


class HostKVCacheManager(kv_cache_manager.KVCacheManager):
  """CPU-only manager backed by Raiden-owned host memory."""

  def __init__(
      self,
      *,
      num_layers: int,
      num_shards: int = 1,
      slice_byte_size: int,
      node_id: int,
      local_port: Optional[int] = None,
      host_blocks: int,
      parallelism: int = 4,
  ):
    manager = kv_cache_manager.KVCacheManager.create_host_only(
        num_layers=num_layers,
        num_shards=num_shards,
        slice_byte_size=slice_byte_size,
        node_id=node_id,
        local_port=local_port,
        host_blocks=host_blocks,
        parallelism=parallelism,
    )
    self.__dict__.update(manager.__dict__)
