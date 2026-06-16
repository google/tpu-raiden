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

import ctypes
import os
import pathlib
from typing import Any, List, Tuple

import torch

import torch_tpu  # noqa: F401  # Load torch shared libraries before the extension.
from torch_tpu import _loader as _torch_tpu_loader


Path = pathlib.Path


def _load_torch_tpu_common() -> None:
  _torch_tpu_loader.load()
  common = Path(torch_tpu.__file__).resolve().parent / "common"
  lib = common / "libpywrap_torch_tpu_common.so"
  if lib.exists():
    ctypes.CDLL(str(lib), mode=os.RTLD_GLOBAL | os.RTLD_NOW)


_load_torch_tpu_common()

from tpu_raiden.frameworks.torch import _tpu_raiden_torch as _impl


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
