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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch_tpu
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
      parallelism: int = 4,
  ):
    """Instantiates the TransferEngine-based KVCacheManager.

    Args:
      kv_caches: List of device-placed contiguous Tensors representing the
        sharded KV caches.
      node_id: Unique identifier for this host/node in the distributed mesh
        (e.g., Tensor Parallel rank).
      local_control_port: TCP socket server port for control plane coordination
        (use -1 to disable the server).
      max_blocks: Maximum number of blocks in the host pool.
      num_slots: Number of transfer slots to allocate.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
      parallelism: Number of parallel network copies per layer.
    """
    self._impl = _impl.KVCacheManager(
        kv_caches=kv_caches,
        node_id=node_id,
        local_control_port=local_control_port,
        max_blocks=max_blocks,
        num_slots=num_slots,
        timeout_s=timeout_s,
        unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
        parallelism=parallelism,
    )

  @property
  def node_id(self) -> int:
    """Returns the active Worker or Shard ID."""
    return self._impl.node_id()

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the active Raiden endpoint descriptors."""
    return self._impl.get_local_endpoints()

  @property
  def local_control_port(self) -> int:
    """Returns the active control plane listener port."""
    return self._impl.local_control_port

  @property
  def local_port(self) -> int:
    """Returns the active data port."""
    return self._impl.local_port

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
      remote_endpoint: Union[str, List[Dict[str, Any]]],
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

  @property
  def listener_addresses(self) -> List[str]:
    """Returns the full network addresses (IP:port) of all active local control listeners."""
    return self._impl.listener_addresses

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ KVCacheListener is actively running."""
    return self._impl.is_listener_active
