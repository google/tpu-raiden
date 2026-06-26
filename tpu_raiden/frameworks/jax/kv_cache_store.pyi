from typing import Any

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

class KVCacheStore:
  def __init__(
      self,
      capacity: int,
  ) -> None: ...
  def lookup(
      self,
      block_hashes: list[bytes],
  ) -> list[tuple[bytes, list[RaidenId]]]:
    """Checks the LRU directory for cached block hashes. Returns a list of all matched replica pairs prior to the first miss."""
    ...
  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenId]],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, list[RaidenId]]]]:
    """Caches sharded buffers into host-RAM/HBM backing store."""
    ...
  def delete(
      self,
      block_hashes: list[bytes],
      slices: list[list[RaidenId]],
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
