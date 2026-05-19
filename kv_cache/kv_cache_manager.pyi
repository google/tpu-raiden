from typing import Any

class KVCacheManager:
  def __init__(
      self,
      device_arrays: list[Any],
      block_size: int = 1,
      local_port: int | None = None,
      host_blocks_to_allocate: int = 64,
      unsafe_skip_buffer_lock: bool = False,
      parallelism: int = 1,
  ) -> None: ...

  def h2d(
      self,
      src_offsets_major_dim: list[int] = ...,
      dst_offsets_major_dim: list[int] = ...,
      copy_sizes_major_dim: list[int] = ...,
  ) -> Any: ...

  def d2h(
      self,
      src_offsets_major_dim: list[int] = ...,
      dst_offsets_major_dim: list[int] = ...,
      copy_sizes_major_dim: list[int] = ...,
  ) -> Any: ...

  def d2h_auto_allocate(
      self,
      src_offsets_major_dim: list[int] = ...,
      copy_sizes_major_dim: list[int] = ...,
      entity_id: int = 0,
  ) -> tuple[list[int], Any]: ...

  def h2h_write(
      self,
      peer: str,
      src_block_ids: list[int],
      entity_id: int = 0,
  ) -> tuple[list[int], Any]: ...

  def h2h_read(
      self,
      peer: str,
      src_block_ids: list[int],
      entity_id: int = 0,
  ) -> tuple[list[int], Any]: ...

  def local_port(self) -> int | None: ...
