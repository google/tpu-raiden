from typing import Any

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

class KVCacheStore:
  def __init__(self, capacity: int) -> None: ...
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
    ...
  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenId]],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, list[RaidenId]]]]:
    """Inserts sharded buffers into the cache."""
    ...
  def delete(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenId]],
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
