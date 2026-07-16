import enum
from typing import Any

class BlockStatus(enum.Enum):
  INIT = ...
  REMOTE = ...
  HOST = ...
  HBM = ...

class RaidenId:
  job_name: str
  job_replica_id: str
  data_name: str
  data_replica_idx: int
  def __init__(
      self,
      job_name: str,
      job_replica_id: str = '',
      data_name: str = '',
      data_replica_idx: int = 0,
  ) -> None: ...

class RaidenBlockID:
  raiden_id: RaidenId
  host_block_id: int
  status: BlockStatus
  def __init__(
      self,
      raiden_id: RaidenId = ...,
      host_block_id: int = ...,
      status: BlockStatus = ...,
  ) -> None: ...

class KVCacheStore:
  def __init__(
      self,
      capacity: int,
      global_registry_address: str = '',
      raiden_id: RaidenId = ...,
  ) -> None: ...

  @property
  def raiden_id(self) -> RaidenId: ...

  def lookup(
      self,
      block_hashes: list[bytes],
      enable_global: bool = False,
  ) -> list[tuple[bytes, list[RaidenBlockID]]]:
    """Checks the LRU directory for cached block hashes. Returns a list of all matched replica pairs prior to the first miss."""
    ...
  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenBlockID]],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, list[RaidenBlockID]]]]:
    """Caches sharded buffers into host-RAM/HBM backing store."""
    ...
  def insert_and_pin(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenBlockID]],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, list[RaidenBlockID]]]]:
    """Pins existing block hashes, and inserts/pins new block hashes.

    Pins all existing block hashes, and inserts and pins new block hashes if
    there is sufficient available space in the LRU cache.

    Returns:
      - bool: whether the entire insert_and_pin operation succeeded (i.e. all
        existing keys were pinned, all new keys inserted and pinned).
      - list: list of entries evicted during insertion.
    """
    ...
  def release_and_delete(
      self,
      block_hashes: list[bytes],
      pending_evict_entries: list[tuple[bytes, list[RaidenBlockID]]]
      | None = ...,
  ) -> tuple[int, list[tuple[bytes, list[RaidenBlockID]]]]:
    """Reverts an insert_and_pin operation.

    Unpins all block_hashes in the LRU cache, deletes any block_hash in REMOTE
    status whose pin count is 0, and puts back evicted entries in reverse order
    for each deleted remote block.

    Returns:
      - int: number of remote blocks deleted.
      - list: remaining evicted entries that were not restored.
    """
    ...
  def delete(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenBlockID]],
  ) -> None:
    """Deletes cached sharded buffers from host-RAM/HBM backing store entirely."""
    ...
  def capacity(self) -> int:
    """Returns the total manageable pool block capacity of the prefix cache hierarchy."""
    ...
  def pin(self, block_hashes: list[bytes]) -> bool:
    """Pins cached block hashes in memory, protecting them against LRU eviction while in active use."""
    ...
  def release(self, block_hashes: list[bytes]) -> None:
    """Releases previously pinned block hashes, making them eligible for LRU eviction when capacity is exceeded."""
    ...
  def save(self, block_hashes: list[bytes]) -> bool:
    """Saves blocks from device (HBM) to host (DRAM) asynchronously."""
    ...
  def load(self, block_hashes: list[bytes], device_block_ids: list[int]) -> bool:
    """Loads blocks from host (DRAM) to device (HBM) asynchronously."""
    ...
  def poll_save_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of asynchronous Save operations."""
    ...
  def poll_load_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of asynchronous Load operations."""
    ...
