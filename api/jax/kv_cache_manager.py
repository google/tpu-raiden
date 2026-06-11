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

"""High-performance JAX KV Cache Manager (repurposed as TransferEngine)."""

from typing import Any, List, Tuple
from frameworks.jax import _transfer_engine as _impl


class KVCacheManager:
  """Wrapper around compiled C++ TransferEngine.

  This class has been repurposed to expose the TransferEngine API interface
  for distributed disaggregated KV-cache movement on JAX TPUs.
  """

  def __init__(
      self,
      kv_caches: List[Any],
      local_control_port: int,
      max_blocks: int,
      num_slots: int,
      timeout_s: float = 120.0,
      unsafe_skip_buffer_lock: bool = True,
  ):
    """Instantiates the TransferEngine-based KVCacheManager.

    Args:
      kv_caches: List of device-placed contiguous Tensors representing the
        sharded KV caches.
      local_control_port: TCP socket server port for control plane coordination.
      max_blocks: Maximum number of blocks in the host pool.
      num_slots: Number of transfer slots to allocate.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
    """
    self._impl = _impl.TransferEngine(
        kv_caches=kv_caches,
        local_control_port=local_control_port,
        max_blocks=max_blocks,
        num_slots=num_slots,
        timeout_s=timeout_s,
        unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
    )

  @property
  def uses_prepared_tpu_buffers(self) -> bool:
    """Returns whether the engine uses pre-allocated TPU buffers."""
    return self._impl.uses_prepared_tpu_buffers

  @property
  def local_control_port(self) -> int:
    """Returns the active control plane listener port."""
    return self._impl.local_control_port

  @property
  def local_data_port(self) -> int:
    """Returns the active data plane listener port."""
    return self._impl.local_data_port

  def register_kv_cache(self, kv_caches: List[Any]) -> None:
    """Registers a new set of JAX TPU KV-cache device arrays."""
    self._impl.register_kv_cache(kv_caches)

  def register_host_buffers(self, host_pool: Any, tp_rank: int) -> None:
    """Registers a shared CPU host memory pool for staging transfers."""
    self._impl.register_host_buffers(host_pool, tp_rank)

  def notify_for_read(self, req_id: str, uuid: int, block_ids: List[int]) -> None:
    """Producer node notifies the registry/peer that blocks are ready for read."""
    self._impl.notify_for_read(req_id, uuid, block_ids)

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: str,
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> None:
    """Consumer node initiates an asynchronous pull of blocks from a remote peer."""
    self._impl.start_read(
        req_id,
        uuid,
        remote_endpoint,
        remote_block_ids,
        local_block_ids,
        parallelism,
    )

  def stage_d2h(
      self, slot_idx: int, num_blocks: int, block_ids: List[int]
  ) -> Tuple[Any, Any, Any, int]:
    """Stages an asynchronous Device-to-Host (D2H) copy of cache slices."""
    return self._impl.stage_d2h(
        slot_idx=slot_idx, num_blocks=num_blocks, block_ids=block_ids
    )

  def stage_d2h_sync(
      self, slot_idx: int, num_blocks: int, block_ids: List[int]
  ) -> None:
    """Stages a synchronous Device-to-Host (D2H) copy of cache slices."""
    self._impl.stage_d2h_sync(
        slot_idx=slot_idx, num_blocks=num_blocks, block_ids=block_ids
    )

  def stage_h2d(
      self, slot_idx: int, num_blocks: int, block_ids: List[int]
  ) -> Tuple[Any, Any, Any, int]:
    """Stages an asynchronous Host-to-Device (H2D) copy of cache slices."""
    return self._impl.stage_h2d(
        slot_idx=slot_idx, num_blocks=num_blocks, block_ids=block_ids
    )

  def commit_h2d(
      self, slot_idx: int, num_blocks: int, local_block_ids: List[int]
  ) -> Tuple[float, float, float, int]:
    """Commits and waits for completion of Host-to-Device (H2D) copies."""
    return self._impl.commit_h2d(
        slot_idx=slot_idx,
        num_blocks=num_blocks,
        local_block_ids=local_block_ids,
    )

  def rank_layer_views(self, slot_idx: int, rank: int, num_blocks: int) -> Any:
    """Returns memory views of the host staging buffers for a specific rank."""
    return self._impl.rank_layer_views(slot_idx, rank, num_blocks)

  def unpack_rank_layers(
      self, slot_idx: int, rank: int, num_blocks: int, layer_buffers: Any
  ) -> None:
    """Copies external numpy/host buffers directly into the staging spans."""
    self._impl.unpack_rank_layers(slot_idx, rank, num_blocks, layer_buffers)

  def submit_d2h(
      self, slot_idx: int, num_blocks: int, block_ids: List[int]
  ) -> None:
    """Submits a previously staged D2H copy to the TPU hardware queue."""
    self._impl.submit_d2h(
        slot_idx=slot_idx, num_blocks=num_blocks, block_ids=block_ids
    )

  def submit_h2d(
      self, slot_idx: int, num_blocks: int, local_block_ids: List[int]
  ) -> None:
    """Submits a previously staged H2D copy to the TPU hardware queue."""
    self._impl.submit_h2d(
        slot_idx=slot_idx,
        num_blocks=num_blocks,
        local_block_ids=local_block_ids,
    )

  def complete_read(self) -> Tuple[bool, bool, bool]:
    """Waits for and completes all active asynchronous read operations."""
    return self._impl.complete_read()

  def poll_transfer_ops(self) -> List[Any]:
    """Polls the status of active background transfer operations."""
    return self._impl.poll_transfer_ops()

  def wait_transfer(self, op_id: Any) -> None:
    """Waits for a specific background transfer operation to complete."""
    self._impl.wait_transfer(op_id)

  def _count_copy_segments_for_testing(self, block_ids: List[int]) -> int:
    return self._impl._count_copy_segments_for_testing(block_ids)

  def _send_copy_plan_for_testing(self, block_ids: List[int]) -> Any:
    return self._impl._send_copy_plan_for_testing(block_ids)

  def _load_copy_plan_for_testing(
      self, remote_block_ids: List[int], local_block_ids: List[int]
  ) -> Any:
    return self._impl._load_copy_plan_for_testing(
        remote_block_ids, local_block_ids
    )
