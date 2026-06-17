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

import logging
import os

# import random
# import time
from typing import Any, List, Optional, Tuple

# import uuid

from tpu_raiden.frameworks.jax import _tpu_raiden_jax as _impl


class CompositeFuture:
  """A future that aggregates multiple underlying futures."""

  def __init__(self, futures):
    self._futures = [f for f in futures if f is not None]

  def Await(self) -> None:
    for f in self._futures:
      f.Await()

  def wait(self) -> None:
    self.Await()

  def IsReady(self) -> bool:
    return all(f.IsReady() for f in self._futures)

  def is_ready(self) -> bool:
    return self.IsReady()


def partition_copy_spec(
    src_offsets, dst_offsets, copy_sizes, split_boundary, partition_by_src=True
):
  mgr0_src, mgr0_dst, mgr0_sz = [], [], []
  mgr1_src, mgr1_dst, mgr1_sz = [], [], []

  for s, d, sz in zip(src_offsets, dst_offsets, copy_sizes):
    val = s if partition_by_src else d

    if val + sz <= split_boundary:
      mgr0_src.append(s)
      mgr0_dst.append(d)
      mgr0_sz.append(sz)
    elif val >= split_boundary:
      mgr1_src.append(s)
      mgr1_dst.append(d)
      mgr1_sz.append(sz)
    else:
      sz0 = split_boundary - val
      sz1 = sz - sz0

      mgr0_src.append(s)
      mgr0_dst.append(d)
      mgr0_sz.append(sz0)

      mgr1_src.append(s + sz0)
      mgr1_dst.append(d + sz0)
      mgr1_sz.append(sz1)

  return (mgr0_src, mgr0_dst, mgr0_sz), (mgr1_src, mgr1_dst, mgr1_sz)


class KVCacheManager:
  """Wrapper around compiled C++ KV Cache Manager.

  Manages C++ TransferEngine.
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
    self._impl = _impl.KVCacheManager(
        kv_caches=kv_caches,
        node_id=0,
        local_control_port=local_control_port,
        max_blocks=max_blocks,
        num_slots=num_slots,
        timeout_s=timeout_s,
        unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
    )

  @property
  def local_ports(self) -> List[int]:
    """Returns the ports that this manager is listening on."""
    return self._impl.local_control_ports()

  def local_port(self) -> int:
    """Returns the first port for backward compatibility."""
    return self.local_ports[0]

  @property
  def local_control_port(self) -> int:
    """Returns the first port for backward compatibility."""
    return self.local_ports[0]

  @property
  def local_ips(self) -> List[str]:
    """Returns the resolved IPs."""
    return self._impl.local_ips()

  def register_read(self, req_id: str, uuid: int, block_ids: List[int]) -> bool:
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: str,
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> Any:
    return self._impl.start_read(
        req_id,
        uuid,
        remote_endpoint,
        remote_block_ids,
        local_block_ids,
        parallelism,
    )

  def h2h_read(
      self, peer: str, src_block_ids: List[int], entity_id: int = 0
  ) -> Tuple[List[int], Any]:
    """Compatibility method for tests/runners."""
    import random
    import uuid

    req_id = f"h2h_read_{uuid.uuid4()}"
    uuid_val = random.randint(0, 2**63 - 1)

    future = self.start_read(
        req_id=req_id,
        uuid=uuid_val,
        remote_endpoint=peer,
        remote_block_ids=src_block_ids,
        local_block_ids=src_block_ids,
    )
    return src_block_ids, future

  def h2d(
      self,
      src_offsets: Optional[List[int]] = None,
      dst_offsets: Optional[List[int]] = None,
      copy_sizes: Optional[List[int]] = None,
  ) -> Any:
    if not src_offsets or not dst_offsets or not copy_sizes:
      return self._impl.h2d()
    return self._impl.h2d(src_offsets, dst_offsets, copy_sizes)

  def d2h(
      self,
      src_offsets: Optional[List[int]] = None,
      dst_offsets: Optional[List[int]] = None,
      copy_sizes: Optional[List[int]] = None,
  ) -> Any:
    if not src_offsets or not dst_offsets or not copy_sizes:
      return self._impl.d2h()
    return self._impl.d2h(src_offsets, dst_offsets, copy_sizes)

  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]:
    return self._impl.complete_read()

  def close(self) -> None:
    self._impl = None
