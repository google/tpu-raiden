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

from tpu_raiden.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_raiden.frameworks.torch import _tpu_raiden_torch as _impl
# pylint: enable=g-import-not-at-top


class RaidenId:
  """Wrapper around compiled C++ RaidenId."""

  def __init__(
      self,
      job_name: str = "",
      job_replica_id: str = "",
      data_name: str = "",
      data_replica_idx: int = 0,
      impl: Any = None,
  ):
    if impl is not None:
      self._impl = impl
    else:
      self._impl = _impl.RaidenId(
          job_name, job_replica_id, data_name, data_replica_idx
      )

  @property
  def job_name(self) -> str:
    return self._impl.job_name

  @property
  def job_replica_id(self) -> str:
    return self._impl.job_replica_id

  @property
  def data_name(self) -> str:
    return self._impl.data_name

  @property
  def data_replica_idx(self) -> int:
    return self._impl.data_replica_idx

  def __repr__(self) -> str:
    return (
        f"RaidenId(job='{self.job_name}', replica='{self.job_replica_id}',"
        f" data='{self.data_name}', data_idx={self.data_replica_idx})"
    )


class KVCacheStore:
  """Wrapper around compiled C++ KVCacheStore."""

  def __init__(self, capacity: int):
    self._impl = _impl.KVCacheStore(capacity=capacity)

  def lookup(
      self,
      block_hashes: list[bytes],
  ) -> list[tuple[bytes, list[RaidenId]]]:
    """Checks the LRU directory for cached block hashes.

    Args:
      block_hashes: Incoming block hashes to check.

    Returns:
      A list of tuples containing the block hash and a list of matching RaidenId
      replicas, halting immediately upon the first cache miss.
    """
    raw_res = self._impl.lookup(block_hashes)
    final_res = []
    for hash_val, raw_slices in raw_res:
      wrapped_slices = [RaidenId(impl=rs) for rs in raw_slices]
      final_res.append((hash_val, wrapped_slices))
    return final_res

  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenId]],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, list[RaidenId]]]]:
    raw_slices = []
    for slice_list in slices:
      raw_slices.append(
          [s._impl for s in slice_list]  # pylint: disable=protected-access
      )
    all_inserted, raw_evicted = self._impl.insert(
        block_hashes, raw_slices, on_host
    )
    wrapped_evicted = []
    for hash_val, raw_slices in raw_evicted:
      wrapped_slices = [RaidenId(impl=rs) for rs in raw_slices]
      wrapped_evicted.append((hash_val, wrapped_slices))
    return all_inserted, wrapped_evicted

  def delete(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenId]],
  ) -> None:
    raw_slices = []
    for slice_list in slices:
      raw_slices.append(
          [s._impl for s in slice_list]  # pylint: disable=protected-access
      )
    self._impl.delete(block_hashes, raw_slices)

  def capacity(self) -> int:
    return self._impl.capacity()

  def pin(self, block_hashes: list[bytes]) -> bool:
    """Pins cached block hashes in memory, protecting them against LRU eviction while in active use."""
    return self._impl.pin(block_hashes)

  def release(self, block_hashes: list[bytes]) -> None:
    """Releases previously pinned block hashes, making them eligible for LRU eviction when capacity is exceeded."""
    self._impl.release(block_hashes)
