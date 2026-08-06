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

"""Raiden KV Cache Store API for PyTorch."""

import enum
from typing import Any

from tpu_raiden.api.torch import torch_tpu_common_loader

torch_tpu_common_loader.load_torch_tpu_common()

# pylint: disable=g-import-not-at-top
from tpu_raiden.api.torch import torch_abi

_impl = torch_abi.load_extension(
    "tpu_raiden.frameworks.torch", "_tpu_raiden_torch"
)
# pylint: enable=g-import-not-at-top


class BlockStatus(enum.Enum):
  INIT = 0
  REMOTE = 1
  HOST = 2
  HBM = 3
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

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, RaidenId):
      return False
    return (
        self.job_name == other.job_name
        and self.job_replica_id == other.job_replica_id
        and self.data_name == other.data_name
        and self.data_replica_idx == other.data_replica_idx
    )


class RaidenBlockID:
  """Wrapper around compiled C++ RaidenBlockID."""

  def __init__(
      self,
      raiden_id: RaidenId | None = None,
      host_block_id: int = -1,
      status: BlockStatus = BlockStatus.INIT,
      device_block_id: int = -1,
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
          raiden_id._impl,
          host_block_id,
          device_block_id,
          status_val,  # pylint: disable=protected-access
      )

  @property
  def raiden_id(self) -> RaidenId:
    return RaidenId(impl=self._impl.raiden_id)

  @property
  def host_block_id(self) -> int:
    return self._impl.host_block_id

  @property
  def device_block_id(self) -> int:
    return self._impl.device_block_id

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
        f" host_block_id={self.host_block_id},"
        f" device_block_id={self.device_block_id}, status={self.status})"
    )


class KVCacheStore:
  """Wrapper around compiled C++ KVCacheStore."""

  def __init__(
      self,
      capacity: int,
      global_registry_address: str = "",
      raiden_id: RaidenId | None = None,
      *,
      num_shards: int,
      store_server_ip: str,
      shard_size_bytes: int = 0,
      raiden_orchestrator_address: str = "",
      raiden_controller_port: int = 0,
  ):
    """Creates a KVCacheStore.

    Every store has a RaidenController (num_shards >= 1) and a
    store_server_ip; there is no controller-less or unpublished
    configuration. Same-host use: "127.0.0.1".
    num_shards and store_server_ip have no default and must be passed by
    keyword -- there is no value for either that is valid to fall into by
    omission.

    Args:
      num_shards: Shard count for this store's RaidenController; must be
        >= 1.
      store_server_ip: IP that peers use to reach this store. Bind-and-
        advertise: this store's KVCacheStoreService and RaidenController both
        bind it, and it is the host published to the global registry. Must be
        an IP (or hostname) this process can bind -- not empty, not a
        wildcard.
      raiden_controller_port: Port for this store's RaidenController; 0 lets
        gRPC choose. Note that the IP address of the controller reuses
        store_server_ip.
    """
    raw_raiden_id = _impl.RaidenId()
    if raiden_id is not None:
      raw_raiden_id = raiden_id._impl  # pylint: disable=protected-access
    self._impl = _impl.KVCacheStore(
        capacity=capacity,
        global_registry_address=global_registry_address,
        raiden_id=raw_raiden_id,
        num_shards=num_shards,
        shard_size_bytes=shard_size_bytes,
        raiden_orchestrator_address=raiden_orchestrator_address,
        store_server_ip=store_server_ip,
        raiden_controller_port=raiden_controller_port,
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
      RaidenBlockID replica, halting immediately upon the first cache miss.
    """
    raw_res = self._impl.lookup(block_hashes, enable_global)
    final_res = []
    for hash_val, raw_slice in raw_res:
      final_res.append((hash_val, RaidenBlockID(impl=raw_slice)))
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
      slices: List of RaidenBlockID, one for each block hash.
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
    for hash_val, raw_slice in raw_evicted:
      wrapped_evicted.append((hash_val, RaidenBlockID(impl=raw_slice)))
    return all_inserted, wrapped_evicted

  def insert_and_lock(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      on_host: bool,
  ) -> bool:
    """Locks existing block hashes, and inserts/locks new block hashes.

    Locks all existing block hashes, and inserts and locks new block hashes if
    there is sufficient available space in the LRU cache.

    Args:
      block_hashes: Incoming block hashes to insert and lock.
      slices: List of RaidenBlockID, one for each block hash.
      on_host: Whether the slices are located in host memory.

    Returns:
      bool: whether the entire insert_and_lock operation succeeded (i.e. all
        existing keys were locked, all new keys inserted and locked).
    """
    raw_slices = []
    for s in slices:
      if isinstance(s, RaidenId):
        s = RaidenBlockID(raiden_id=s)
      raw_slices.append(s._impl)  # pylint: disable=protected-access
    return self._impl.insert_and_lock(block_hashes, raw_slices, on_host)

  def release_and_delete(
      self,
      block_hashes: list[bytes],
  ) -> int:
    """Reverts an insert_and_lock operation.

    Unpins all block_hashes in the LRU cache, deletes any block_hash in REMOTE
    status whose pin count is 0.

    Args:
      block_hashes: Incoming block hashes to unpin and check for deletion.

    Returns:
      int: number of remote blocks deleted.
    """
    return self._impl.release_and_delete(block_hashes)

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

  def register_kv_transfer_spec(
      self,
      block_array_bytes: list[int],
      num_kv_shards: int,
      num_workers: int,
  ) -> None:
    """Registers the deployment's KVTransferSpec with the global registry.

    The first registered spec wins; later calls validate against it
    (idempotent). Raises RuntimeError on mismatch or when the store has no
    global registry configured.
    """
    self._impl.register_kv_transfer_spec(
        block_array_bytes, num_kv_shards, num_workers
    )

  def read_remote(
      self,
      block_hashes: list[bytes],
      device_block_ids: list[int] | None = None,
  ) -> bool:
    """Launches an async receiver-initiated read of REMOTE blocks from peers.

    Returns as soon as the reads are issued; poll with
    poll_remote_read_status().

    Args:
      block_hashes: Block hashes to read. Each must already be pinned, and must
        stay pinned until poll_remote_read_status() reports it terminal.
        Releasing early makes the entry deletable mid-read, in which case the
        read is discarded and the WHOLE batch is reported failed.
      device_block_ids: Optional. Omit (or pass None) to read into host DRAM,
        leaving the entries HOST. Pass one device block id per hash to read
        straight into HBM, leaving the entries HOST_AND_HBM (the host landing
        blocks act as the staging hop, so a later load() can reuse them).
        On FAILURE the contents of these device blocks are UNDEFINED: they are
        written before the source's verdict is known. Treat them as scratch
        until the read reports success -- nothing in the cache points at them
        unless it does.

    Returns:
      True if successfully launched.
    """
    return self._impl.read_remote(block_hashes, device_block_ids or [])

  def poll_remote_read_status(
      self,
  ) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active remote reads.

    Returns:
      A tuple of (done, failed, pending), where:
        done: List of block hashes whose remote read successfully completed.
        failed: List of block hashes whose remote read failed.
        pending: List of block hashes whose remote read is still in progress.
    """
    return self._impl.poll_remote_read_status()

  def save(self, block_hashes: list[bytes]) -> bool:
    """Saves blocks from device (HBM) to host (DRAM) asynchronously."""
    return self._impl.save(block_hashes)

  def load(
      self, block_hashes: list[bytes], device_block_ids: list[int]
  ) -> bool:
    """Loads blocks from host (DRAM) to device (HBM) asynchronously."""
    return self._impl.load(block_hashes, device_block_ids)

  def poll_save_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active asynchronous Save operations."""
    return self._impl.poll_save_status()

  def poll_load_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active asynchronous Load operations."""
    return self._impl.poll_load_status()

  def write_remote(
      self,
      block_hashes: list[bytes],
      dst_raiden_id: RaidenId,
  ) -> bool:
    """Offers blocks to a peer, which takes its own copy.

    The use case is proactive eviction: a node under memory pressure hands cold
    blocks to a peer and then frees them locally. The DESTINATION pulls the
    bytes; this node never pushes.

    Returns as soon as the destination has decided; poll with
    poll_remote_write_status().

    Requires a global registry at both ends -- that is how the destination is
    resolved, and how the blocks become reachable once they land.

    IMPORTANT: the destination allocates landing blocks from FREE blocks only
    and never evicts to make room, so a peer whose cache is warm refuses every
    offer. Destination-side eviction is separate, unimplemented work.

    Args:
      block_hashes: Block hashes to offer. Each must already be resident in this
        node's host DRAM. The store takes its own internal pin for the life of
        the operation; the caller's pin is separate and the caller releases it
        itself, once poll_remote_write_status() reports the hashes terminal.
      dst_raiden_id: The peer to offer them to.

    Returns:
      True if the offer was accepted or answered; False if it could not be
      made at all (no registry, unresolvable peer, blocks not held here).
      A True return does NOT mean the peer has the blocks -- poll for that.
    """
    return self._impl.write_remote(
        block_hashes, dst_raiden_id._impl  # pylint: disable=protected-access
    )

  def poll_remote_write_status(
      self,
  ) -> tuple[list[bytes], list[bytes], list[bytes], list[bytes], list[bytes]]:
    """Polls the status of all active remote writes.

    Returns:
      A tuple of (done, failed, pending, existing, unregistered) -- five
      elements, unlike the save/load/read pollers' three. The last two annotate
      failures rather than being outcomes of their own, because this store
      never decides on the caller's behalf:
        done: Hashes the destination now holds AND has published, so peers can
          find them. Includes the case where it already had them: hashes are
          content-addressed, so a peer having them is the post-condition the
          caller wanted. This is the only result that makes it safe to drop a
          local copy.
        failed: Hashes whose remote write did not succeed.
        pending: Hashes whose remote write is still in progress.
        existing: Hashes a destination already held when it refused a PARTIAL
          batch. This store does not reissue with the remainder -- doing so is
          the caller's decision, and this is the list it needs to make it.
        unregistered: The destination HAS these -- the transfer succeeded --
          but could not publish them, so no lookup will find them there. They
          are reported failed because the safe default is to keep your own
          copy: freeing it would move the block from findable-here to
          findable-nowhere. A caller that only needed the peer to hold the
          bytes may treat this as success; a caller that was about to free its
          copy must not.

    A hash in `existing` or `unregistered` also appears in `failed`, not
    instead of it.
    """
    return self._impl.poll_remote_write_status()
