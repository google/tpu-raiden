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

"""High-performance PyTorch Weight Synchronizer for Trainer-Inference Pipelines."""

from typing import List, Optional

import torch

# Import Pybind11 dynamic binary extension E2E!
from tpu_raiden.frameworks.torch import _tpu_raiden_torch as _weight_synchronizer


class WeightSynchronizer:
  """Zero-copy PyTorch Weight Synchronizer."""

  def __init__(
      self,
      device_tensors: List[List[torch.Tensor]],
      local_port: Optional[int] = None,
      parallelism: int = 1,
  ):
    """Instantiates the PyTorch Weight Synchronizer shims.

    Args:
      device_tensors: List of list of device-placed contiguous Tensors
        representing the sharded model weights [layers, shards].
      local_port: Sockets listener port assigned for incoming pulls (inference
        mode).
      parallelism: Parallel TCP sockets workers count.
    """
    self._impl = _weight_synchronizer.WeightSynchronizer(
        device_tensors, local_port, parallelism
    )

  def push_weights(self, peers: List[str]) -> None:
    """Trainer pushing current model weights to peer inference server coordinates E2E."""
    self._impl.PushWeights(peers)

  def pull_weights(self, source: str) -> None:
    """Inference server pulling current weights from the source peer coordinate E2E."""
    self._impl.PullWeights(source)

  def d2h(self) -> None:
    """Triggers asynchronous Device-to-Host (D2H) copy of current weights to Host buffer."""
    self._impl.D2h()

  def pull_weights_chunk(
      self,
      source: str,
      src_shard_idx: int,
      src_offset_bytes: int,
      dst_shard_idx: int,
      dst_offset_bytes: int,
      size_bytes: int,
  ) -> None:
    """Inference server pulling a specific byte range directly from a source worker peer.

    Args:
      source: "host:port" coordinate of the source peer.
      src_shard_idx: Target source device shard index to read.
      src_offset_bytes: Offset in bytes inside source shard staging buffer.
      dst_shard_idx: Local destination device shard index to write.
      dst_offset_bytes: Offset in bytes inside local destination staging buffer.
      size_bytes: Number of bytes to transfer.
    """
    self._impl.PullWeightsChunk(
        source,
        src_shard_idx,
        src_offset_bytes,
        dst_shard_idx,
        dst_offset_bytes,
        size_bytes,
    )

  def h2d_chunk(
      self,
      shard_idx: int,
      host_offset_bytes: int,
      device_offset_bytes: int,
      size_bytes: int,
  ) -> None:
    """Triggers asynchronous Host-to-Device (H2D) chunk copy directly to Device HBM.

    Args:
      shard_idx: Target shard index.
      host_offset_bytes: Source offset in Host staging buffer.
      device_offset_bytes: Destination offset in Device memory.
      size_bytes: Number of bytes to copy.
    """
    self._impl.H2dChunk(
        shard_idx, host_offset_bytes, device_offset_bytes, size_bytes
    )

  def get_host_buffer(
      self, layer_idx: int = 0, shard_idx: int = 0
  ) -> torch.Tensor:
    """Returns a zero-copy Host-side CPU PyTorch Tensor view of the C++ staging buffer.

    Args:
      layer_idx: Target layer index to fetch.
      shard_idx: Target shard index to fetch.
    """
    return self._impl.get_host_buffer(layer_idx, shard_idx)

  @property
  def local_port(self) -> Optional[int]:
    """Returns assigned ephemeral listener port coordinates."""
    return self._impl.local_port

  @property
  def num_layers(self) -> int:
    """Returns total layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns physical devices shards count."""
    return self._impl.num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns individual major dimension slice byte size capacity."""
    return self._impl.slice_byte_size
