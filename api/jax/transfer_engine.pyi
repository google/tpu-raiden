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

"""Interface definitions for JAX TransferEngine."""

class TransferEngine:
  def __init__(
      self,
      kv_caches: list,
      local_control_port: int,
      max_blocks: int,
      num_slots: int,
      timeout_s: float = ...,
      unsafe_skip_buffer_lock: bool = ...,
  ) -> None: ...
  def notify_for_read(
      self,
      req_id: str,
      uuid: int,
      block_ids: list[int],
  ) -> int: ...
  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: str,
      remote_block_ids: list[int],
      local_block_ids: list[int],
  ) -> int: ...
  def complete_read(self) -> tuple[list[str], list[str], list[str]]: ...

RaidenTransferEngine = TransferEngine
