from typing import Any

class KVCacheStore:
  def __init__(self, block_size: int, capacity: int) -> None: ...
  def lookup_and_fetch(
      self,
      block_hashes: list[int],
      device_arrays: Any,
      dst_offsets_major_dim: list[int],
      copy_sizes_major_dim: list[int],
  ) -> tuple[list[bool], Any]:
    """Looks up block hashes in the cache and pulls them to device.

    Upon a cache hit, triggers asynchronous Host-to-Device (H2D) copy transfers
    from internal host cache buffers to the target device_arrays at the
    specified
    destination offsets.

    Note: The batch lookup will early-terminate on the first cache miss. That
    is, once a miss is encountered, all subsequent hashes in the lookup batch
    are treated as cache misses.

    Args:
      block_hashes: A list of 64-bit integer hashes representing the block keys
        to lookup in the cache.
      device_arrays: A list of target JAX/IFRT device arrays to copy block data
        to.
      dst_offsets_major_dim: Starting offset indexes along the major dimension
        in the destination arrays for copying cache-hit blocks.
      copy_sizes_major_dim: Block segment sizes to copy along the major
        dimension. Must be multiples of the cache's block size.

    Returns:
      A tuple of `(hits, copy_future)`:
        hits: A list of booleans indicating whether each block hash in
          `block_hashes` had a cache hit. Length matches `block_hashes`.
        copy_future: An asynchronous `PjRtCopyFuture` object representing the
        H2D copy transfers, or `()` if there are no cache hits. Callers must
        invoke `.wait()` on this future before accessing the underlying data in
        `device_arrays`.
    """
    ...
  def insert(
      self,
      block_hashes: list[int],
      device_arrays: Any,
      src_offsets_major_dim: list[int],
      copy_sizes_major_dim: list[int],
  ) -> None:
    """Inserts device block data into the cache.

    Triggers asynchronous Device-to-Host (D2H) copy transfers from target
    device_arrays to allocated internal host buffers, and registers the block
    hashes
    into the cache LRU map with their allocated block IDs.

    Args:
      block_hashes: A list of 64-bit integer hashes representing the keys to
        store in the cache.
      device_arrays: A list of source JAX/IFRT device arrays containing raw KV
        cache data.
      src_offsets_major_dim: Starting offset indexes along the major dimension
        in source arrays to copy cache block data from.
      copy_sizes_major_dim: Block segment sizes to copy along the major
        dimension. Must be multiples of the cache's block size.
    """
    ...
