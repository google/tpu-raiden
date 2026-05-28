class RaidenTransferEngine:
  def __init__(
      self,
      kv_caches: list,
      tp_rank: int,
      local_control_port: int,
      max_blocks: int,
      num_slots: int,
      timeout_s: float = ...,
      unsafe_skip_buffer_lock: bool = ...,
  ) -> None: ...
  def register_send(
      self,
      req_id: str,
      uuid: int,
      block_ids: list[int],
  ) -> int: ...
  def submit_load(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: str,
      remote_block_ids: list[int],
      local_block_ids: list[int],
  ) -> int: ...
  def poll_finished(self) -> tuple[list[str], list[str], list[str]]: ...
