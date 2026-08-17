import enum
from typing import Any, overload

class BlockStatus(enum.Enum):
  INIT = ...
  REMOTE = ...
  HOST = ...
  HBM = ...
  # A load that sourced from local DRAM ends here; one that sourced from a
  # peer ends at HBM instead, because its landing blocks are freed and no host
  # copy is kept.
  HOST_AND_HBM = ...

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
  # The local host block when status is HOST or HOST_AND_HBM; the OWNING
  # PEER's host block when status is REMOTE, where it means nothing locally.
  host_block_id: int
  device_block_id: int
  status: BlockStatus
  @overload
  def __init__(
      self,
      raiden_id: RaidenId = ...,
      host_block_id: int = ...,
      status: BlockStatus = ...,
  ) -> None: ...
  @overload
  def __init__(
      self,
      raiden_id: RaidenId = ...,
      host_block_id: int = ...,
      device_block_id: int = ...,
      status: BlockStatus = ...,
  ) -> None: ...

class KVCacheStore:
  def __init__(
      self,
      capacity: int,
      global_registry_address: str = '',
      raiden_id: RaidenId = ...,
      *,
      num_shards: int,
      shard_size_bytes: int = 0,
      store_server_ip: str,
      raiden_controller_port: int = 0,
      expected_worker_count: int = 0,
      kv_pool_group: str = '',
  ) -> None: ...

  @property
  def raiden_id(self) -> RaidenId: ...

  @property
  def raiden_controller_address(self) -> str: ...

  @property
  def store_server_address(self) -> str: ...

  def lookup(
      self,
      block_hashes: list[bytes],
      enable_global: bool = False,
  ) -> list[tuple[bytes, RaidenBlockID]]:
    """Checks the LRU directory for cached block hashes. Returns a list of all matched replica pairs prior to the first miss."""
    ...
  def insert(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      on_host: bool,
  ) -> tuple[bool, list[tuple[bytes, RaidenBlockID]]]:
    """Caches sharded buffers into host-RAM/HBM backing store."""
    ...
  def insert_and_lock(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      on_host: bool,
  ) -> bool:
    """Locks existing block hashes, and inserts/locks new block hashes.

    Locks all existing block hashes, and inserts and locks new block hashes if
    there is sufficient available space in the LRU cache.

    Returns:
      - bool: whether the entire insert_and_lock operation succeeded (i.e. all
        existing keys were locked, all new keys inserted and locked).
    """
    ...
  def release_and_delete(
      self,
      block_hashes: list[bytes],
  ) -> int:
    """Reverts an insert_and_lock operation.

    Unpins all block_hashes in the LRU cache, deletes any block_hash in REMOTE
    status whose pin count is 0.

    Returns:
      - int: number of remote blocks deleted.
    """
    ...
  def delete(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
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
    """Saves blocks from device (HBM) to host (DRAM) asynchronously.

    NOTE: The block_hashes must be pinned in the LRU cache (e.g. via
    insert_and_lock) before calling save. Once the operation is complete (as
    reported by poll_save_status), the caller must manually release/unpin them
    via release so they can be evicted if needed.

    Recommended usage flow:
      0. [optional for save] lookup block hashes
      1. insert_and_lock(block_hashes)
      2. save(block_hashes)
      3. poll_save_status() -> wait for completion
      4. release(completed_block_hashes)
    """
    ...
  @overload
  def load(self, block_hashes: list[bytes], device_block_ids: list[int]) -> bool:
    """Asynchronously loads KV cache blocks to device (HBM) from local host DRAM.

    `device_block_ids` is the destination and must name one device block per hash.

    NOTE: The block_hashes must be pinned in the LRU cache before calling load.
    Once the operation is complete (as reported by poll_load_status), the caller
    must manually release/unpin them via release.
    """
    ...
  @overload
  def load(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      device_block_ids: list[int],
  ) -> bool:
    """Asynchronously loads KV cache blocks to device (HBM), either from local
    host DRAM or from a peer.

    ONE CALL IS ONE SOURCE. Every block in a batch must carry the same status,
    and remote blocks must all refer to the same peer.

    `device_block_ids` is the destination and must name one device block per hash.

    If `slices` is provided, the caller's pre-looked up RaidenBlockIDs are
    used directly. Note that blocks in `slices` must be already pinned
    externally (when Loading from local host), and remote loads will re-resolve
    hashes at the peer, ignoring `slices`.
    """
    ...
  def poll_save_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of asynchronous Save operations."""
    ...
  def poll_load_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of asynchronous Load operations."""
    ...
  def read_remote(
      self,
      block_hashes: list[bytes],
      slices: list[RaidenBlockID],
      device_block_ids: list[int],
  ) -> bool:
    """Reads REMOTE blocks from their owning peers into local HBM."""
    ...
  def poll_remote_read_status(self) -> tuple[list[bytes], list[bytes], list[bytes]]:
    """Polls status of active remote reads."""
    ...
  def write_remote(
      self,
      block_hashes: list[bytes],
      dst_raiden_id: RaidenId,
  ) -> bool:
    """Offers blocks to a peer, which takes its own copy.

    The destination pulls the bytes; this node never pushes. Returns as soon
    as the offer is issued -- poll with poll_remote_write_status.
    """
    ...
  def poll_remote_write_status(
      self,
  ) -> tuple[list[bytes], list[bytes], list[bytes], list[bytes], list[bytes]]:
    """Polls status of active remote writes.

    Five lists, unlike the three-list save/load/read pollers:
    (done, failed, pending, existing, unregistered). The last two annotate
    failures rather than being outcomes of their own -- a hash in `existing`
    or `unregistered` also appears in `failed`, not instead of it. `existing`
    is what a destination already held when it refused a partial batch;
    `unregistered` is a destination that HAS the bytes but could not publish
    them, so no lookup will find them there.
    """
    ...
