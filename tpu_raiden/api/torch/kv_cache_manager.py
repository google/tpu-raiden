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

"""High-performance PyTorch KV Cache Manager (repurposed as TransferEngine)."""

import os
import sys
from typing import Any, List, Tuple

import torch
import torch_tpu  # noqa: F401  # Load torch shared libraries before the extension.
from torch_tpu import _loader as _torch_tpu_loader


def _load_torch_tpu_common() -> None:
  # Load torch_tpu's shared libraries so libpywrap_torch_tpu_common.so is in
  # the process. Deliberately do NOT `ctypes.CDLL(..., RTLD_GLOBAL)` it: the
  # extension statically links its own XLA, and globalizing libpywrap's XLA
  # makes raiden's allocator-factory registry interpose onto libpywrap's,
  # aborting with a duplicate `DefaultCPUAllocator` registration. The
  # extension NEEDED-links libpywrap (see build.sh patchelf step), so its
  # torch_tpu symbols resolve through that dependency in local scope.
  _torch_tpu_loader.load()


_load_torch_tpu_common()

# Import the extension in private (RTLD_LOCAL) scope so its embedded XLA stays
# separate from libpywrap's (mirrors how jax extensions load). A global-scope
# load would merge the two XLA copies' allocator registries and abort.
_prev_dlopen_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_LOCAL | os.RTLD_NOW)
try:
  from tpu_raiden.frameworks.torch import _tpu_raiden_torch as _impl
finally:
  sys.setdlopenflags(_prev_dlopen_flags)


class KVCacheManager:
  """Wrapper around compiled C++ KV Cache Manager.

  This class has been repurposed to expose the TransferEngine API interface
  for distributed disaggregated KV-cache movement on PyTorch TPUs.
  """

  def __init__(
      self,
      kv_caches: List[Any],
      node_id: int,
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
      node_id: Worker or Shard ID (e.g., Tensor Parallel rank).
      local_control_port: TCP socket server port for control plane coordination.
      max_blocks: Maximum number of blocks in the host pool.
      num_slots: Number of transfer slots to allocate.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
    """
    self._impl = _impl.KVCacheManager(
        kv_caches=kv_caches,
        node_id=node_id,
        local_control_port=local_control_port,
        max_blocks=max_blocks,
        num_slots=num_slots,
        timeout_s=timeout_s,
        unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
    )

  @property
  def node_id(self) -> int:
    """Returns the active Worker or Shard ID."""
    return self._impl.node_id()

  @property
  def local_control_port(self) -> int:
    """Returns the active control plane listener port."""
    return self._impl.local_control_port

  def register_read(
      self, req_id: str, uuid: int, block_ids: List[int]
  ) -> bool:
    """Producer node notifies the registry/peer that blocks are ready for read.

    Args:
      req_id: The request ID of the transfer operation.
      uuid: The UUID of the request.
      block_ids: The list of block IDs to be read.

    Returns:
      True if a transfer is indeed needed; False if there is nothing to be
      transferred.
    """
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: str,
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> int:
    """Consumer node initiates an asynchronous pull of blocks from a remote peer."""
    return self._impl.start_read(
        req_id,
        uuid,
        remote_endpoint,
        remote_block_ids,
        local_block_ids,
        parallelism,
    )

  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]:
    """Polls the status of all active background transfer operations.

    Returns:
      A tuple of (done_sending, done_recving, failed_recving) lists of request
      IDs.
    """
    return self._impl.complete_read()

  def d2h(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
  ) -> Any:
    """Device-to-Host (D2H) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.D2h(src_offsets, dst_offsets, copy_sizes)

  def h2d(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
  ) -> Any:
    """Host-to-Device (H2D) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.H2d(src_offsets, dst_offsets, copy_sizes)
