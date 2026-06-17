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

from typing import Any, List, Optional, Tuple

class CompositeFuture:
  def Await(self) -> None: ...
  def wait(self) -> None: ...
  def IsReady(self) -> bool: ...
  def is_ready(self) -> bool: ...



class KVCacheManager:
  def __init__(
      self,
      kv_caches: List[Any],
      local_control_port: int,
      max_blocks: int,
      num_slots: int,
      timeout_s: float = ...,
      unsafe_skip_buffer_lock: bool = ...,
  ) -> None: ...
  @property
  def local_ports(self) -> List[int]: ...
  def local_port(self) -> int: ...
  @property
  def local_control_port(self) -> int: ...
  @property
  def local_ips(self) -> List[str]: ...
  def register_read(
      self,
      req_id: str,
      uuid: int,
      block_ids: List[int],
  ) -> bool: ...
  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: str,
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = ...,
  ) -> CompositeFuture: ...
  def h2h_read(
      self,
      peer: str,
      src_block_ids: List[int],
      entity_id: int = ...,
  ) -> Tuple[List[int], CompositeFuture]: ...
  def h2d(
      self,
      src_offsets: Optional[List[int]] = ...,
      dst_offsets: Optional[List[int]] = ...,
      copy_sizes: Optional[List[int]] = ...,
  ) -> CompositeFuture: ...
  def d2h(
      self,
      src_offsets: Optional[List[int]] = ...,
      dst_offsets: Optional[List[int]] = ...,
      copy_sizes: Optional[List[int]] = ...,
  ) -> CompositeFuture: ...
  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]: ...
