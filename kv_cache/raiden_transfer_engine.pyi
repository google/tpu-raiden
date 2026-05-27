class RaidenTransferFuture:
  def Await(self) -> None: ...
  def wait(self) -> None: ...
  def IsReady(self) -> bool: ...
  def is_ready(self) -> bool: ...

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
  @property
  def uses_prepared_tpu_buffers(self) -> bool: ...
  @property
  def local_control_port(self) -> int: ...
  @property
  def local_data_port(self) -> int: ...
  def register_kv_cache(self, kv_caches: list) -> list[int]: ...
  def register_host_buffers(self, host_pool, tp_rank: int) -> None: ...
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
  def stage_d2h(
      self,
      *,
      slot_idx: int,
      num_blocks: int,
      block_ids: list[int],
  ) -> tuple[RaidenTransferFuture, list, list, int]: ...
  def stage_d2h_sync(
      self,
      *,
      slot_idx: int,
      num_blocks: int,
      block_ids: list[int],
  ) -> None: ...
  def commit_h2d(
      self,
      *,
      slot_idx: int,
      num_blocks: int,
      local_block_ids: list[int],
  ) -> tuple[float, float, float, int]: ...
  def rank_layer_views(self, slot_idx: int, rank: int, num_blocks: int): ...
  def unpack_rank_layers(
      self,
      slot_idx: int,
      rank: int,
      num_blocks: int,
      layer_buffers,
  ) -> None: ...
  def submit_d2h(
      self,
      *,
      slot_idx: int,
      num_blocks: int,
      block_ids: list[int],
  ) -> int: ...
  def submit_h2d(
      self,
      *,
      slot_idx: int,
      num_blocks: int,
      local_block_ids: list[int],
  ) -> int: ...
  def poll_finished(self) -> tuple[list[str], list[str], list[str]]: ...
  def poll_transfer_ops(self) -> list[int]: ...
  def wait_transfer(self, op_id: int) -> None: ...
  def _count_copy_segments_for_testing(self, block_ids: list[int]) -> int: ...
  def _send_copy_plan_for_testing(self, block_ids: list[int]) -> dict: ...
  def _load_copy_plan_for_testing(
      self,
      remote_block_ids: list[int],
      local_block_ids: list[int],
  ) -> dict: ...

def await_all(
    futures: RaidenTransferFuture | list[RaidenTransferFuture],
) -> None: ...
def is_ready(
    futures: RaidenTransferFuture | list[RaidenTransferFuture],
) -> bool: ...
