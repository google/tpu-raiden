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

# Copyright 2026 The TPU Raiden Authors. All Rights Reserved.
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
"""Raiden Controller providing high-level transfer API and resharding plans."""

from dataclasses import dataclass
import threading
from typing import Optional


class RaidenMemoryType:
  """Raiden memory type."""

  DRAM = 1
  HBM = 2


@dataclass(frozen=True)
class RaidenId:
  """Identifier for the work unit in Raiden owning a sharded set of data."""

  # The name of the job, e.g., 'trainer', 'sampler', 'inference_server'
  job_name: str
  # Identifier of replicated jobs, combined with job name to identify the job,
  # int or uuid, e.g., sampler 0, inference_server 215, etc.
  job_replica_id: str = ""
  # Name of the array/tensor, e.g., 'kv_cache', 'model.weights' etc.
  data_name: str = ""
  # Identifier for replicated data within the job, mostly for DP within a job
  # replica
  data_replica_idx: int = 0


NDSlice = list[tuple[int, int]]


@dataclass
class TransferPlan:
  """A detailed plan for data transfer with resharding if needed."""

  src_units: list[RaidenId]
  dst_units: list[RaidenId]

  # For push model, maps each source's `RaidenId` to its specific shard push
  # schedule, i.e. shard index to a list of destination's `RaidenId`, shard
  # index, and the n-dimensional slice offsets for the shard index.
  plan: dict[RaidenId, list[list[tuple[RaidenId, int, list[NDSlice]]]]]


class RaidenFuture:
  """Future representing an asynchronous transfer execution."""

  session_id: int

  def __init__(self, session_id: int = 0, transfer_task=None):
    self.session_id = session_id
    self._transfer_task = transfer_task
    self._completed = False

  async def wait(self) -> None:
    """Waits asynchronously for the transfer operation to complete."""
    if self._transfer_task:
      await self._transfer_task
    self._completed = True

  def done(self) -> bool:
    """Returns True if the transfer operation has completed."""
    return self._completed


class RaidenController:
  """High-level transfer controller managing active transfers and generating transfer plans."""

  def __init__(self, port: int):
    self.port = port
    self._active_transfers: dict[str, TransferPlan] = {}
    self._registered_shards: dict[RaidenId, list[str]] = {}
    self._lock = threading.Lock()

  def register_work_unit(self, unit: RaidenId, shards: list[str]) -> None:
    """Registers the physical worker shard addresses for a given semantic RaidenId."""
    with self._lock:
      self._registered_shards[unit] = shards

  def _resolve_shards(self, unit: RaidenId) -> list[str]:
    with self._lock:
      return self._registered_shards.get(unit, ["10.0.0.1:8000"])

  def get_plan(self, req_id: str) -> Optional[TransferPlan]:
    """Returns the generated TransferPlan for a given transfer request ID."""
    with self._lock:
      return self._active_transfers.get(req_id)

  def kickoff_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      src_block_ids: Optional[list[int]] = None,
      dst_device_block_ids: Optional[list[int]] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
  ) -> RaidenFuture:
    """For a requested data transfer, generates a transfer plan for the work units to carry out and kick it off.

    Args:
      src_units: list of source work units containing the data.
      dst_units: All destination work units that need the data.
      req_id: Unique identifier for the active transfer entity.
      src_block_ids: list of source block IDs to be transferred.
      dst_device_block_ids: list of destination device block IDs to receive the
        data. This is only needed when the destination memory type is HBM.
      dst_mem_type: The dst memory type of the data written to.

    Returns:
      A Future for the call site to wait for the transfer to complete.
    """
    if not src_units:
      raise ValueError("src_units must not be empty.")
    if not dst_units:
      raise ValueError("dst_units must not be empty.")
    if src_block_ids:
      raise NotImplementedError("src_block_ids are not supported yet.")
    if dst_device_block_ids:
      raise NotImplementedError("dst_device_block_ids are not supported yet.")

    # Select only one source unit for now.
    with self._lock:
      active_src_counts = {}
      active_dst_counts = {}
      for existing_plan in self._active_transfers.values():
        for s in existing_plan.src_units:
          active_src_counts[s.job_replica_id] = (
              active_src_counts.get(s.job_replica_id, 0) + 1
          )

        for t in existing_plan.dst_units:
          active_dst_counts[t.job_replica_id] = (
              active_dst_counts.get(t.job_replica_id, 0) + 1
          )

    selected_src = min(
        src_units, key=lambda s: active_src_counts.get(s.job_replica_id, 0)
    )

    num_src = len(self._resolve_shards(selected_src))
    plan: dict[
        RaidenId, list[list[tuple[RaidenId, int, list[NDSlice]]]]
    ] = {}

    src_plan: list[list[tuple[RaidenId, int, list[NDSlice]]]] = [
        [] for _ in range(num_src)
    ]

    for dst_unit in dst_units:
      num_dst = len(self._resolve_shards(dst_unit))

      for i in range(num_src):
        src_start = i * num_dst
        src_end = (i + 1) * num_dst

        for j in range(num_dst):
          dst_start = j * num_src
          dst_end = (j + 1) * num_src

          intersect_start = max(src_start, dst_start)
          intersect_end = min(src_end, dst_end)

          if intersect_start < intersect_end:
            local_start = intersect_start - src_start
            local_end = intersect_end - src_start
            nd_slice: NDSlice = [(local_start, local_end)]
            src_plan[i].append((dst_unit, j, [nd_slice]))

    plan[selected_src] = src_plan

    plan = TransferPlan(
        src_units=[selected_src],
        dst_units=dst_units,
        plan=plan,
    )

    with self._lock:
      session_id = len(self._active_transfers)
      if not req_id:
        req_id = f"req_{session_id}"
      self._active_transfers[req_id] = plan

    async def _execute_transfer() -> None:
      pass

    transfer_task = _execute_transfer()
    return RaidenFuture(session_id=session_id, transfer_task=transfer_task)



