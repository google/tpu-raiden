import enum
from typing import Any

class BlockStatus(enum.Enum):
  INIT = ...
  REMOTE = ...
  HBM = ...
  HOST = ...
  HOST_AND_HBM = ...

class RaidenId:
  job_name: str
  job_replica_id: str
  data_name: str
  data_replica_idx: int
  def __init__(
      self,
      job_name: str = ...,
      job_replica_id: str = ...,
      data_name: str = ...,
      data_replica_idx: int = ...,
      impl: Any = ...,
  ) -> None: ...

class RaidenBlockID:
  raiden_id: RaidenId
  host_block_id: int
  status: BlockStatus
  def __init__(
      self,
      raiden_id: RaidenId | None = ...,
      host_block_id: int = ...,
      status: BlockStatus = ...,
      impl: Any = ...,
  ) -> None: ...

class RemoteFetchConfig:
  orchestrator_address: str
  controller_port: int
  local_worker_port: int
  bytes_per_block: int
  num_shards: int
  def __init__(self) -> None: ...

class FetchFuture:
  def Await(self) -> None: ...
  def IsDone(self) -> bool: ...

class KVCacheStore:
  def __init__(
      self,
      capacity: int,
      global_registry_address: str = ...,
      raiden_id: RaidenId | None = ...,
      remote_config: RemoteFetchConfig | None = ...,
  ) -> None: ...

  @property
  def raiden_id(self) -> RaidenId: ...

  def lookup(
      self,
      block_hashes: list[bytes],
      enable_global: bool = ...,
  ) -> list[tuple[bytes, list[RaidenBlockID]]]:
    """Checks the LRU directory for cached block hashes.

    Args:
      block_hashes: Incoming block hashes to check.

    Returns:
      A list of tuples containing the block hash and a list of matching RaidenBlockID
      replicas, halting immediately upon the first cache miss.
    """
    ...
  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenBlockID]],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, list[RaidenBlockID]]]]:
    """Inserts sharded buffers into the cache."""
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
    """Returns the capacity of the cache."""
    ...
  def pin(self, block_hashes: list[bytes]) -> bool:
    """Pins cached block hashes in memory, protecting them against LRU eviction while in active use."""
    ...
  def release(self, block_hashes: list[bytes]) -> None:
    """Releases previously pinned block hashes, making them eligible for LRU eviction when capacity is exceeded."""
    ...
  def fetch_remote(self, block_hashes: list[bytes]) -> dict[bytes, FetchFuture]:
    """Initiates remote fetch for the given block hashes."""
    ...
  def poll_fetch_remote_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls the status of remote fetches. Returns (done, failed, pending) block hashes."""
    ...
