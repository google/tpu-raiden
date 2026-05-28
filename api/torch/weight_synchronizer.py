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
from api.torch import _weight_synchronizer


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
