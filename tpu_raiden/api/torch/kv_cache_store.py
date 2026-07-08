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

import ctypes
import enum
import os
import pathlib
from typing import Any

import torch_tpu
from torch_tpu import _loader as _torch_tpu_loader

Path = pathlib.Path


from tpu_raiden.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_raiden.frameworks.torch import _tpu_raiden_torch as _impl
# pylint: enable=g-import-not-at-top


class BlockStatus(enum.Enum):
  INIT = 0
  REMOTE = 1
  HBM = 2
  HOST = 3
  HOST_AND_HBM = 4


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


class RaidenBlockID:
  """Wrapper around compiled C++ RaidenBlockID."""

  def __init__(
      self,
      raiden_id: RaidenId | None = None,
      host_block_id: int = -1,
      status: BlockStatus = BlockStatus.INIT,
      impl: Any = None,
  ):
    if impl is not None:
      self._impl = impl
    else:
      if raiden_id is None:
        raiden_id = RaidenId()
      # Map Python enum to C++ enum
      status_val = getattr(_impl.BlockStatus, status.name)
      self._impl = _impl.RaidenBlockID(
          raiden_id._impl, host_block_id, status_val  # pylint: disable=protected-access
      )

  @property
  def raiden_id(self) -> RaidenId:
    return RaidenId(impl=self._impl.raiden_id)

  @property
  def host_block_id(self) -> int:
    return self._impl.host_block_id

  @property
  def status(self) -> BlockStatus:
    return BlockStatus[self._impl.status.name]

  @property
  def job_name(self) -> str:
    return self.raiden_id.job_name

  @property
  def job_replica_id(self) -> str:
    return self.raiden_id.job_replica_id

  @property
  def data_name(self) -> str:
    return self.raiden_id.data_name

  @property
  def data_replica_idx(self) -> int:
    return self.raiden_id.data_replica_idx

  def __repr__(self) -> str:
    return (
        f"RaidenBlockID(raiden_id={self.raiden_id},"
        f" host_block_id={self.host_block_id}, status={self.status})"
    )


class KVCacheStore:
  """Wrapper around compiled C++ KVCacheStore."""

  def __init__(
      self,
      capacity: int,
      global_registry_address: str = "",
      raiden_id: RaidenId | None = None,
  ):
    raw_raiden_id = _impl.RaidenId()
    if raiden_id is not None:
      raw_raiden_id = raiden_id._impl  # pylint: disable=protected-access
    self._impl = _impl.KVCacheStore(
        capacity=capacity,
        global_registry_address=global_registry_address,
        raiden_id=raw_raiden_id,
    )

  @property
  def raiden_id(self) -> RaidenId:
    """Returns the RaidenId associated with this store."""
    return RaidenId(impl=self._impl.raiden_id)

  def lookup(
      self,
      block_hashes: list[bytes],
      enable_global: bool = False,
  ) -> list[tuple[bytes, RaidenBlockID]]:
    """Checks the LRU directory for cached block hashes.

    Args:
      block_hashes: Incoming block hashes to check.
      enable_global: Whether to fallback to global registry on miss.

    Returns:
      A list of tuples containing the block hash and the matching
      RaidenBlockID, halting immediately upon the first cache miss.
    """
    raw_res = self._impl.lookup(block_hashes, enable_global)
    final_res = []
    for hash_val, raw_block in raw_res:
      final_res.append((hash_val, RaidenBlockID(impl=raw_block)))
    return final_res

  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, RaidenBlockID]]]:
    """Caches sharded buffers into host-RAM/HBM backing store.

    Args:
      block_hashes: Incoming block hashes to insert.
      slices: List of buffer metadata corresponding to each block hash.
      on_host: Whether the slices are located in host memory.

    Returns:
      A tuple containing:
      - bool: whether all blocks were successfully inserted (i.e. none already
        existed).
      - list: list of entries evicted from the LRU cache during insertion.
    """
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockID(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    all_inserted, raw_evicted = self._impl.insert(
        block_hashes, raw_slices, on_host
    )
    wrapped_evicted = []
    for hash_val, raw_block in raw_evicted:
      wrapped_evicted.append((hash_val, RaidenBlockID(impl=raw_block)))
    return all_inserted, wrapped_evicted

  def insert_and_pin(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, RaidenBlockID]]]:
    """Pins existing block hashes, and inserts/pins new block hashes.

    Pins all existing block hashes, and inserts and pins new block hashes if
    there is sufficient available space in the LRU cache.

    Args:
      block_hashes: Incoming block hashes to insert and pin.
      slices: List of buffer metadata corresponding to each block hash.
      on_host: Whether the slices are located in host memory.

    Returns:
      A tuple containing:
      - bool: whether the entire insert_and_pin operation succeeded (i.e. all
        existing keys were pinned, all new keys inserted and pinned).
      - list: list of entries evicted during insertion.
    """
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockID(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    all_inserted, raw_evicted = self._impl.insert_and_pin(
        block_hashes, raw_slices, on_host
    )
    wrapped_evicted = []
    for hash_val, raw_block in raw_evicted:
      wrapped_evicted.append((hash_val, RaidenBlockID(impl=raw_block)))
    return all_inserted, wrapped_evicted

  def release_and_delete(
      self,
      block_hashes: list[bytes],
      pending_evict_entries: list[tuple[bytes, RaidenBlockID]] | None = None,
  ) -> tuple[int, list[tuple[bytes, RaidenBlockID]]]:
    """Reverts an insert_and_pin operation.

    Unpins all block_hashes in the LRU cache, deletes any block_hash in REMOTE
    status whose pin count is 0, and puts back evicted entries in reverse order
    for each deleted remote block.

    Args:
      block_hashes: Incoming block hashes to unpin and check for deletion.
      pending_evict_entries: List of entries previously evicted during
        insert_and_pin.

    Returns:
      A tuple containing:
      - int: number of remote blocks deleted.
      - list: remaining evicted entries that were not restored.
    """
    if pending_evict_entries is None:
      pending_evict_entries = []
    raw_evicted_in = []
    for hash_val, s in pending_evict_entries:
      if isinstance(s, RaidenId):
        s = RaidenBlockID(raiden_id=s)
      raw_evicted_in.append((hash_val, s._impl))  # pylint: disable=protected-access
    del_count, raw_evicted_out = self._impl.release_and_delete(
        block_hashes, raw_evicted_in
    )
    wrapped_evicted_out = []
    for hash_val, raw_block in raw_evicted_out:
      wrapped_evicted_out.append((hash_val, RaidenBlockID(impl=raw_block)))
    return del_count, wrapped_evicted_out

  def delete(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
  ) -> None:
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockID(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    self._impl.delete(block_hashes, raw_slices)

  def capacity(self) -> int:
    return self._impl.capacity()

  def pin(self, block_hashes: list[bytes]) -> bool:
    """Pins cached block hashes in memory, protecting them against LRU eviction while in active use."""
    return self._impl.pin(block_hashes)

  def release(self, block_hashes: list[bytes]) -> None:
    """Releases previously pinned block hashes, making them eligible for LRU eviction when capacity is exceeded."""
    self._impl.release(block_hashes)
