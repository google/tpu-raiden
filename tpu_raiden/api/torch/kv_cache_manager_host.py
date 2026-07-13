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
"""Host-only Python adapter for Raiden KV-cache tests."""

from __future__ import annotations

from typing import Any, Optional


class HostKVCacheManager:
  """Python adapter around the CPU-only nanobind manager."""

  def __init__(
      self,
      *,
      num_layers: int,
      num_shards: int = 1,
      slice_byte_size: int,
      node_id: int,
      local_port: Optional[int] = None,
      host_blocks: int,
      parallelism: int = 4,
  ):
    # Import lazily so importing the host-only wrapper never initializes the
    # torch_tpu runtime.
    # pylint: disable=g-import-not-at-top
    from tpu_raiden.frameworks.torch import _tpu_raiden_host
    # pylint: enable=g-import-not-at-top

    self._impl = _tpu_raiden_host.KVCacheManager(
        num_layers=num_layers,
        num_shards=num_shards,
        slice_byte_size=slice_byte_size,
        node_id=node_id,
        local_port=local_port,
        host_blocks_to_allocate=host_blocks,
        parallelism=parallelism,
    )

  def __getattr__(self, name: str) -> Any:
    """Delegates the native host manager surface."""
    return getattr(self._impl, name)

  def register_active_plan(
      self, uuid: int, request: Any, is_sender: bool
  ) -> None:
    """Registers a protobuf or serialized StartTransferRequest."""
    if hasattr(request, "SerializeToString"):
      request = request.SerializeToString()
    if not isinstance(request, (bytes, bytearray)):
      raise TypeError("request must be bytes or a protobuf message")
    self._impl.register_active_plan(uuid, bytes(request), is_sender)
