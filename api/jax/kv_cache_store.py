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

"""Python wrapper for the compiled C++ KVCacheStore."""

from typing import Any
from frameworks.jax import kv_cache_store as _impl


class KVCacheStore:
  """Wrapper around compiled C++ KVCacheStore."""

  def __init__(
      self,
      block_size: int,
      capacity: int,
      global_registry_address: str = "",
      local_address: str = "",
  ):
    self._impl = _impl.KVCacheStore(
        block_size=block_size,
        capacity=capacity,
        global_registry_address=global_registry_address,
        local_address=local_address,
    )

  def lookup_and_fetch(
      self,
      block_hashes: list[int],
      device_arrays: Any,
      dst_offsets_major_dim: list[int],
      copy_sizes_major_dim: list[int],
  ) -> tuple[list[bool], Any]:
    return self._impl.lookup_and_fetch(
        block_hashes,
        device_arrays,
        dst_offsets_major_dim,
        copy_sizes_major_dim,
    )

  def insert(
      self,
      block_hashes: list[int],
      device_arrays: Any,
      src_offsets_major_dim: list[int],
      copy_sizes_major_dim: list[int],
  ) -> None:
    self._impl.insert(
        block_hashes,
        device_arrays,
        src_offsets_major_dim,
        copy_sizes_major_dim,
    )
