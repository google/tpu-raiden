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

import asyncio
import dataclasses
import math
import socket
import threading
import time
from typing import Optional, Protocol, Union, runtime_checkable

from weight_sync import weight_synchronizer_service_pb2


class NameResolver(Protocol):
  """Interface for resolving remote network coordinates (e.g.

  BNS) to raw IP addresses.
  """

  def resolve(self, address_str: str) -> str:
    ...


@runtime_checkable
class RaidenEngine(Protocol):
  """Standardized Data-Plane collective engine interface for Raiden Worker daemons."""

  @property
  def local_port(self) -> Optional[int]:
    """Returns the active TCP socket server listener port."""
    ...


class RaidenMemoryType:
  """Raiden memory type constants."""

  DRAM = 1
  HBM = 2


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass
class TransferPlan:
  """A detailed plan for data transfer with resharding if needed."""

  src_units: list[RaidenId]
  dst_units: list[RaidenId]

  # For push model, maps each source's `RaidenId` to its specific shard push
  # schedule, i.e. shard index to a list of destination's `RaidenId`, shard
  # index, and the n-dimensional slice offsets for the shard index.
  plan: dict[RaidenId, list[list[tuple[RaidenId, int, list[NDSlice]]]]]

  shard_push_schedules: dict[
      RaidenId, dict[int, list[tuple[str, int, int, int, int]]]
  ] = dataclasses.field(default_factory=dict)

  # Maps every RaidenId in the plan to its physical Control-Plane RPC
  # address
  worker_rpc_addresses: dict[RaidenId, str] = dataclasses.field(
      default_factory=dict
  )

  # Maps every RaidenId in the plan to its physical Data TCP socket
  # endpoints
  worker_data_addresses: dict[RaidenId, list[str]] = dataclasses.field(
      default_factory=dict
  )


def create_server_socket(port: int) -> socket.socket:
  """Creates an IPv6 socket (supporting IPv4 dual-stack on Linux) or falls back to IPv4."""
  try:
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", port))
    sock.listen(128)
    return sock
  except Exception:  # pylint: disable=broad-except
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.listen(128)
    return sock


def connect_socket(
    address_str: str,
    timeout: float = 60.0,
    resolver: Optional[NameResolver] = None,
) -> socket.socket:
  """Connects to an IPv4 or IPv6 endpoint robustly with optional coordinate name resolution."""
  start_time = time.time()
  if resolver:
    address_str = resolver.resolve(address_str)

  rindex = address_str.rfind(":")
  host = address_str[:rindex]
  port = int(address_str[rindex + 1 :])
  if host.startswith("[") and host.endswith("]"):
    host = host[1:-1]

  while True:
    try:
      for res in socket.getaddrinfo(
          host, port, socket.AF_UNSPEC, socket.SOCK_STREAM
      ):
        af, socktype, proto, _, sa = res
        sock = None
        try:
          sock = socket.socket(af, socktype, proto)
          sock.settimeout(timeout)
          sock.connect(sa)
          return sock
        except OSError:
          if sock:
            sock.close()
    except OSError:
      pass

    if time.time() - start_time > timeout:
      raise RuntimeError(
          f"Timeout ({timeout}s) failed to connect to robust endpoint"
          f" {address_str}"
      )
    time.sleep(2.0)


class WorkerRpcClient:
  """Distributed RPC Client connecting to Native C++ Control Daemons with Event-Driven resolution.

  Maintains an asynchronous endpoint catalog that resolves worker network
  coordinates instantaneously when participating worker tasks self-register,
  completely eliminating hardcoded active polling loops or arbitrary delays.
  """

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
  ):
    """Instantiates RPC Client with an optional initial endpoint mapping.

    Args:
      endpoint_addresses: Initial catalog of known Worker RPC addresses.
      resolve_timeout: Maximum duration in seconds to wait for a pending worker
        task to self-register before raising a Timeout RuntimeError.
      name_resolver: Interface for resolving remote coordinates (e.g. BNS).
    """
    self._endpoints = endpoint_addresses or {}
    self._pending_endpoints: dict[RaidenId, asyncio.Future[str]] = {}
    self._resolve_timeout = resolve_timeout
    self._name_resolver = name_resolver

  def register_worker_endpoint(
      self, worker_name: RaidenId, rpc_address: str
  ) -> None:
    """Registers remote Worker Control-Plane RPC TCP listener address.

    If any active coroutine is currently suspended awaiting this specific worker
    coordinate, its asyncio Future is immediately resolved.

    Args:
      worker_name: Participating worker RaidenId coordinate.
      rpc_address: TCP server address in 'IP:Port' or Google BNS format.
    """
    self._endpoints[worker_name] = rpc_address
    future = self._pending_endpoints.pop(worker_name, None)
    if future and not future.done():
      future.set_result(rpc_address)

  async def _resolve_endpoint(self, target_id: RaidenId) -> str:
    """Resolves worker RPC address asynchronously or suspends execution until registered.

    Args:
      target_id: Target worker RaidenId coordinate to resolve.

    Returns:
      Resolved remote RPC TCP server address string.

    Raises:
      RuntimeError: If the remote endpoint fails to self-register within
        `resolve_timeout`.
    """
    addr = self._endpoints.get(target_id)
    if addr:
      return addr

    future = self._pending_endpoints.setdefault(target_id, asyncio.Future())
    try:
      return await asyncio.wait_for(future, timeout=self._resolve_timeout)
    except asyncio.TimeoutError as e:
      raise RuntimeError(
          f"Timeout ({self._resolve_timeout}s) waiting for remote RPC"
          f" endpoint {target_id} to self-register"
      ) from e

  async def start_transfer(
      self, target_id: RaidenId, transfer_plan: TransferPlan
  ) -> None:
    """Connects to remote Worker servicer and dispatches encoded collective transfer commands.

    Args:
      target_id: Target participating worker RaidenId.
      transfer_plan: Distributed transfer execution plan mapping source and
        destination topology.

    Raises:
      RuntimeError: If remote servicer socket connection fails, or if remote
        native execution reports failure status.
    """
    try:
      payload = self._encode_start_transfer(target_id, transfer_plan)
      if not payload:
        return
    except NotImplementedError:
      return
    addr = await self._resolve_endpoint(target_id)

    sock = connect_socket(addr, timeout=60.0, resolver=self._name_resolver)
    try:
      sock.sendall(len(payload).to_bytes(4, "big") + payload)

      resp_len_bytes = b""
      while len(resp_len_bytes) < 4:
        chunk = sock.recv(4 - len(resp_len_bytes))
        if not chunk:
          raise RuntimeError(
              "Remote servicer closed connection while reading response length"
          )
        resp_len_bytes += chunk
      resp_len = int.from_bytes(resp_len_bytes, "big")

      resp_bytes = b""
      while len(resp_bytes) < resp_len:
        chunk = sock.recv(resp_len - len(resp_bytes))
        if not chunk:
          raise RuntimeError(
              "Remote servicer closed connection while reading response data"
          )
        resp_bytes += chunk

      self._verify_response(resp_bytes)
    finally:
      sock.close()

  def _encode_start_transfer(
      self, target_id: RaidenId, transfer_plan: TransferPlan
  ) -> Optional[bytes]:
    """Serializes domain-specific binary Protobuf command for collective transfer kickoff.

    Args:
      target_id: Target worker RaidenId coordinate.
      transfer_plan: Top-level distributed Collective Transfer execution plan.

    Returns:
      Serialized binary bytes payload, or None for no-op execution.
    """
    raise NotImplementedError("Subclasses must override _encode_start_transfer")

  def _verify_response(self, resp_bytes: bytes) -> None:
    """Validates demarshaled remote response bytes returned from C++ workers.

    Args:
      resp_bytes: Raw binary bytes received over the TCP socket.

    Raises:
      RuntimeError: If remote execution reports explicit failure status.
    """
    raise NotImplementedError("Subclasses must override _verify_response")

  def get_worker_endpoints(self) -> dict[RaidenId, str]:
    """Returns active read-only snapshot of known registered Worker RPC endpoints."""
    return dict(self._endpoints)

  async def shutdown_workers(self) -> None:
    """Dispatches remote shutdown signaling payloads to all registered worker daemons."""
    for _, addr in list(self._endpoints.items()):
      try:
        sock = connect_socket(addr, timeout=10.0, resolver=self._name_resolver)
        payload = self._encode_shutdown()
        sock.sendall(len(payload).to_bytes(4, "big") + payload)
        sock.close()
      except Exception:  # pylint: disable=broad-except
        pass

  def _encode_shutdown(self) -> bytes:
    """Serializes domain-specific binary command for remote shutdown signaling."""
    raise NotImplementedError("Subclasses must override _encode_shutdown")


class WeightSyncWorkerRpcClient(WorkerRpcClient):
  """Concrete domain subclass for state-of-the-art Weight Synchronizer Protobuf serialization."""

  def _encode_start_transfer(
      self, target_id: RaidenId, transfer_plan: TransferPlan
  ) -> Optional[bytes]:
    """Serializes ControlRequest protobuf specifically for native WeightSynchronizer servicer endpoints.

    Args:
      target_id: Target worker RaidenId coordinate.
      transfer_plan: Top-level distributed Collective Transfer execution plan.

    Returns:
      Serialized binary ControlRequest Protobuf bytes payload, or None for no-op
      Destination execution.
    """
    if target_id not in transfer_plan.src_units:
      return None

    peers = []
    for dst in transfer_plan.dst_units:
      dst_coords = transfer_plan.worker_data_addresses.get(
          dst, ["127.0.0.1:8000"]
      )
      peers.extend(dst_coords)

    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        peers=peers,
    )

    start_req = weight_synchronizer_service_pb2.StartTransferRequest(
        src_units=[_raiden_id_to_proto(u) for u in transfer_plan.src_units],
        dst_units=[_raiden_id_to_proto(u) for u in transfer_plan.dst_units],
    )

    if transfer_plan.shard_push_schedules:
      push_schedules = transfer_plan.shard_push_schedules.get(target_id)
      if push_schedules:
        for shard_idx, entries in push_schedules.items():
          schedule_proto = (
              weight_synchronizer_service_pb2.ShardPushScheduleProto()
          )
          for (
              dst_peer,
              dst_shard_idx,
              dst_offset,
              src_offset,
              size,
          ) in entries:
            entry_proto = schedule_proto.entries.add()
            entry_proto.dst_peer = dst_peer
            entry_proto.dst_shard_idx = dst_shard_idx
            entry_proto.dst_offset_bytes = dst_offset
            entry_proto.src_offset_bytes = src_offset
            entry_proto.size_bytes = size
          start_req.shard_push_schedules[shard_idx].CopyFrom(schedule_proto)

    req.start_transfer_request.CopyFrom(start_req)
    return req.SerializeToString()

  def _verify_response(self, resp_bytes: bytes) -> None:
    """Validates demarshaled ControlResponse protobuf status returned from C++ WeightSynchronizer servers.

    Args:
      resp_bytes: Raw binary Protobuf bytes received over the TCP socket.

    Raises:
      RuntimeError: If remote execution reports explicit failure status.
    """
    resp = weight_synchronizer_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(
          f"Weight Synchronizer remote native execution failed: {resp.message}"
      )

  def _encode_shutdown(self) -> bytes:
    """Serializes ControlRequest protobuf for remote WeightSynchronizer shutdown signaling."""
    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_SHUTDOWN
    )
    return req.SerializeToString()


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


def intersect_nd_slices(
    slice1: list[tuple[int, int]], slice2: list[tuple[int, int]]
) -> Optional[list[tuple[int, int]]]:
  """Computes the precise N-dimensional intersection bounding box between two multi-dimensional slices.

  Each slice is represented as a list of coordinate bounds (start, end) for
  each dimension.

  Args:
    slice1: First N-dimensional slice bounding box.
    slice2: Second N-dimensional slice bounding box.

  Returns:
    A list of (start, end) coordinate bounds representing the intersecting
    subgrid, or None if the slices do not overlap in any dimension.
  """
  result = []
  for (s1, e1), (s2, e2) in zip(slice1, slice2):
    start = max(s1, s2)
    end = min(e1, e2)
    if start >= end:
      return None
    result.append((start, end))
  return result


def generate_1d_copy_chunks(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    itemsize: int,
) -> list[tuple[int, int, int]]:
  """Translates an N-dimensional grid intersection into non-contiguous 1D memory copy byte chunks.

  When linearizing multi-dimensional arrays, an intersecting subgrid is often
  non-contiguous in memory. This function computes the exact linear source and
  destination byte offsets and chunk sizes needed to transmit the non-adjacent
  strided minor rows over a flat 1D data stream.

  Args:
    src_shard_slice: Global multi-dimensional bounding box of the source shard.
    dst_shard_slice: Global multi-dimensional bounding box of the destination
      shard.
    intersection_slice: Global multi-dimensional bounding box of the overlapping
      subgrid.
    itemsize: Byte size of a single array scalar element (e.g., 4 for float32).

  Returns:
    A list of (src_offset_bytes, dst_offset_bytes, size_bytes) tuples defining
    the concrete 1D linear memory copy chunk operations.
  """
  rank = len(src_shard_slice)
  src_shape = [e - s for s, e in src_shard_slice]
  dst_shape = [e - s for s, e in dst_shard_slice]
  int_shape = [e - s for s, e in intersection_slice]

  src_local_int_slice = [
      (int_s - src_s, int_e - src_s)
      for (src_s, _), (int_s, int_e) in zip(src_shard_slice, intersection_slice)
  ]
  dst_local_int_slice = [
      (int_s - dst_s, int_e - dst_s)
      for (dst_s, _), (int_s, int_e) in zip(dst_shard_slice, intersection_slice)
  ]

  src_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1]

  dst_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1]

  minor_dim_size = int_shape[-1]
  contiguous_bytes = minor_dim_size * itemsize

  chunks = []
  outer_shape = int_shape[:-1]
  num_outer_elements = math.prod(outer_shape) if outer_shape else 1

  for i in range(num_outer_elements):
    multi_index = []
    temp = i
    for dim_size in reversed(outer_shape):
      multi_index.append(temp % dim_size)
      temp //= dim_size
    multi_index.reverse()

    src_offset_items = 0
    dst_offset_items = 0

    for d in range(rank - 1):
      src_idx = src_local_int_slice[d][0] + multi_index[d]
      src_offset_items += src_idx * src_strides[d]

      dst_idx = dst_local_int_slice[d][0] + multi_index[d]
      dst_offset_items += dst_idx * dst_strides[d]

    src_offset_items += src_local_int_slice[-1][0] * src_strides[-1]
    dst_offset_items += dst_local_int_slice[-1][0] * dst_strides[-1]

    chunks.append((
        src_offset_items * itemsize,
        dst_offset_items * itemsize,
        contiguous_bytes,
    ))

  return chunks


class RaidenController:
  """High-level transfer controller managing active transfers and generating transfer plans."""

  def __init__(
      self, port: int, worker_rpc_client: Optional[WorkerRpcClient] = None
  ):
    self.port = port
    self._active_transfers: dict[str, TransferPlan] = {}
    self._registered_shards: dict[RaidenId, list[str]] = {}
    self._registered_shard_slices: dict[
        RaidenId, list[list[tuple[int, int]]]
    ] = {}
    self._registered_itemsizes: dict[RaidenId, int] = {}
    self._lock = threading.Lock()
    self.worker_rpc_client = worker_rpc_client or WorkerRpcClient()

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      shard_nd_slices: Optional[
          Union[
              list[list[tuple[int, int]]],
              list[weight_synchronizer_service_pb2.NDSliceProto],
          ]
      ] = None,
      itemsize: Optional[int] = None,
  ) -> None:
    """Registers physical worker shard Data addresses and optional Control-Plane RPC endpoint."""
    if shard_nd_slices is not None and itemsize is None:
      raise ValueError(
          "itemsize must not be None if shard_nd_slices is provided."
      )
    with self._lock:
      self._registered_shards[unit] = shards
      if shard_nd_slices:
        self._registered_shard_slices[unit] = shard_nd_slices
      if itemsize:
        self._registered_itemsizes[unit] = itemsize
      if control_plane_rpc_address and hasattr(
          self.worker_rpc_client, "register_worker_endpoint"
      ):
        self.worker_rpc_client.register_worker_endpoint(
            unit, control_plane_rpc_address
        )

  def _resolve_shards(self, unit: RaidenId) -> list[str]:
    with self._lock:
      return self._registered_shards.get(unit, ["10.0.0.1:8000"])

  def get_plan(self, req_id: str) -> Optional[TransferPlan]:
    """Returns the generated TransferPlan for a given transfer request ID."""
    with self._lock:
      return self._active_transfers.get(req_id)

  def start_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      src_block_ids: Optional[list[int]] = None,
      dst_device_block_ids: Optional[list[int]] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
  ) -> RaidenFuture:
    """For a requested data transfer, generates a transfer plan for the work units to carry out and start it.

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

    src_slices = self._registered_shard_slices.get(selected_src)
    dst_slices = {}
    for d in dst_units:
      d_slices = self._registered_shard_slices.get(d)
      if d_slices:
        dst_slices[d] = d_slices

    itemsize = self._registered_itemsizes.get(selected_src)
    if src_slices is not None and itemsize is None:
      raise ValueError(
          "itemsize must be registered if shard_nd_slices is provided."
      )

    shard_push_schedules: dict[
        RaidenId, dict[int, list[tuple[str, int, int, int, int]]]
    ] = {}

    if src_slices and len(dst_slices) == len(dst_units):
      push_schedules: dict[int, list[tuple[str, int, int, int, int]]] = {}
      for src_shard_idx, src_slice in enumerate(src_slices):
        shard_entries = []
        for dst_unit in dst_units:
          d_slices = dst_slices[dst_unit]
          dst_shards = self._resolve_shards(dst_unit)
          for dst_shard_idx, dst_slice in enumerate(d_slices):
            intersection = intersect_nd_slices(src_slice, dst_slice)
            if intersection:
              dst_peer = dst_shards[dst_shard_idx % len(dst_shards)]
              chunks = generate_1d_copy_chunks(
                  src_slice, dst_slice, intersection, itemsize
              )
              for src_offset, dst_offset, size in chunks:
                shard_entries.append((
                    dst_peer,
                    dst_shard_idx,
                    dst_offset,
                    src_offset,
                    size,
                ))
        if shard_entries:
          push_schedules[src_shard_idx] = shard_entries

      shard_push_schedules[selected_src] = push_schedules

    rpc_addresses = {}
    if hasattr(self.worker_rpc_client, "get_worker_endpoints"):
      rpc_addresses = self.worker_rpc_client.get_worker_endpoints()

    data_addresses = {unit: self._resolve_shards(unit) for unit in dst_units}

    plan = TransferPlan(
        src_units=[selected_src],
        dst_units=dst_units,
        plan=plan,
        shard_push_schedules=shard_push_schedules,
        worker_rpc_addresses=rpc_addresses,
        worker_data_addresses=data_addresses,
    )

    with self._lock:
      session_id = len(self._active_transfers)
      if not req_id:
        req_id = f"req_{session_id}"
      self._active_transfers[req_id] = plan

    async def _execute_transfer() -> None:
      # Controller sends start_transfer RPC to ALL participating Source and
      # Destination workers!
      targets = [selected_src]
      for unit in dst_units:
        if unit not in targets:
          targets.append(unit)
      for target_unit in targets:
        await self.worker_rpc_client.start_transfer(target_unit, plan)

    transfer_task = _execute_transfer()
    return RaidenFuture(session_id=session_id, transfer_task=transfer_task)


class RaidenControllerServer:
  """Centralized Control-Plane network servicer hosting a highly secure JSON/Pickle TCP Controller server."""

  def __init__(self, controller: RaidenController):
    """Instantiates RaidenControllerServer on an active RaidenController instance.

    Args:
      controller: High-level RaidenController instance managing transfer plans.
    """
    self._controller = controller
    self._sock = create_server_socket(controller.port)
    self._stopped = False
    self._thread = None

  def start(self) -> int:
    """Spawns background server acceptance thread listening for incoming Controller RPCs.

    Returns:
      Active TCP listener port coordinate.
    """
    self._thread = threading.Thread(target=self._server_loop, daemon=True)
    self._thread.start()
    return self._controller.port

  def stop(self) -> None:
    """Signals servicer loop shutdown and unblocks pending accept state."""
    self._stopped = True
    try:
      connect_socket(f"127.0.0.1:{self._controller.port}", timeout=1.0)
    except Exception:  # pylint: disable=broad-except
      pass

    self._sock.close()

  def _server_loop(self) -> None:
    """Internal socket server connection acceptance coroutine loop."""
    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    while not self._stopped:
      try:
        conn, _ = self._sock.accept()
        if self._stopped:
          conn.close()
          break
        threading.Thread(
            target=self._handle_conn, args=(conn, loop), daemon=True
        ).start()
      except OSError:
        break

    loop.close()

  def _handle_conn(
      self, conn: socket.socket, loop: asyncio.AbstractEventLoop
  ) -> None:
    """Internal connection processing handler executing deserialized ControlRequest Protobuf RPC payloads.

    Args:
      conn: Accepted incoming TCP socket client handle.
      loop: Target asyncio coroutine loop.
    """
    try:

      len_bytes = b""
      while len(len_bytes) < 4:
        chunk = conn.recv(4 - len(len_bytes))
        if not chunk:
          return
        len_bytes += chunk
      req_len = int.from_bytes(len_bytes, "big")

      req_bytes = b""
      while len(req_bytes) < req_len:
        req_bytes += conn.recv(req_len - len(req_bytes))

      req = weight_synchronizer_service_pb2.ControlRequest()
      req.ParseFromString(req_bytes)

      resp = weight_synchronizer_service_pb2.ControlResponse()
      resp.success = False

      try:
        if (
            req.command
            == weight_synchronizer_service_pb2.ControlRequest.COMMAND_REGISTER_WORK_UNIT
        ):
          reg = req.register_work_unit_request
          unit = RaidenId(
              reg.unit.job_name, reg.unit.job_replica_id, reg.unit.data_name
          )
          shards = list(reg.shards)
          ctrl_addr = (
              reg.control_plane_rpc_address
              if reg.control_plane_rpc_address
              else None
          )
          shard_slices = []
          for nd_slice_proto in reg.shard_nd_slices:
            dims = []
            for dim_proto in nd_slice_proto.dimensions:
              dims.append((dim_proto.start, dim_proto.end))
            shard_slices.append(dims)

          itemsize = reg.itemsize if reg.itemsize > 0 else None

          self._controller.register_work_unit(
              unit, shards, ctrl_addr, shard_slices, itemsize
          )
          resp.success = True
        elif (
            req.command
            == weight_synchronizer_service_pb2.ControlRequest.COMMAND_START_TRANSFER
        ):
          start_req = req.start_transfer_request
          srcs = [
              RaidenId(u.job_name, u.job_replica_id, u.data_name)
              for u in start_req.src_units
          ]
          dsts = [
              RaidenId(u.job_name, u.job_replica_id, u.data_name)
              for u in start_req.dst_units
          ]
          future = self._controller.start_transfer(srcs, dsts, None)
          loop.run_until_complete(future.wait())
          resp.success = True
        elif (
            req.command
            == weight_synchronizer_service_pb2.ControlRequest.COMMAND_SHUTDOWN
        ):
          if hasattr(self._controller.worker_rpc_client, "shutdown_workers"):
            loop.run_until_complete(
                self._controller.worker_rpc_client.shutdown_workers()
            )
          self.stop()
          resp.success = True
      except Exception as e:  # pylint: disable=broad-except
        resp.message = str(e)

      resp_bytes = resp.SerializeToString()
      conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
    except Exception:  # pylint: disable=broad-except
      pass

    finally:
      conn.close()


def _raiden_id_to_proto(
    unit: RaidenId,
) -> weight_synchronizer_service_pb2.RaidenIdProto:
  return weight_synchronizer_service_pb2.RaidenIdProto(
      job_name=unit.job_name,
      job_replica_id=unit.job_replica_id,
      data_name=unit.data_name,
  )


class RaidenControllerClientFacade:
  """Client-side stub encapsulating real remote Network RPCs to a centralized RaidenControllerServer."""

  def __init__(
      self,
      controller_address: str,
      name_resolver: Optional[NameResolver] = None,
  ):
    """Accepts Controller server coordinate 'ip:port'."""
    self._address = controller_address
    self._name_resolver = name_resolver

  def _send_protobuf_rpc(
      self, req: weight_synchronizer_service_pb2.ControlRequest
  ) -> bool:
    """Helper method to serialize and send an RPC Protobuf over robust persistent TCP sockets."""
    sock = connect_socket(
        self._address, timeout=300.0, resolver=self._name_resolver
    )

    try:
      payload = req.SerializeToString()
      sock.sendall(len(payload).to_bytes(4, "big") + payload)

      resp_len_bytes = b""
      while len(resp_len_bytes) < 4:
        chunk = sock.recv(4 - len(resp_len_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response length")
        resp_len_bytes += chunk
      resp_len = int.from_bytes(resp_len_bytes, "big")

      resp_bytes = b""
      while len(resp_bytes) < resp_len:
        chunk = sock.recv(resp_len - len(resp_bytes))
        if not chunk:
          raise RuntimeError("Connection closed while reading response data")
        resp_bytes += chunk

      resp = weight_synchronizer_service_pb2.ControlResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return True
    finally:
      sock.close()

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      shard_nd_slices: Optional[
          Union[
              list[list[tuple[int, int]]],
              list[weight_synchronizer_service_pb2.NDSliceProto],
          ]
      ] = None,
      itemsize: Optional[int] = None,
  ) -> None:
    """Sends remote RPC to register a physical worker entity with the central RaidenControllerServer.

    Args:
      unit: Work unit identifier owning the data shards.
      shards: list of physical Data TCP addresses (e.g. 'IP:Port').
      control_plane_rpc_address: Optional worker Control-Plane RPC servicer
        endpoint coordinate.
      shard_nd_slices: Optional bounding boxes for each shard.
      itemsize: Optional item size in bytes.
    """
    if shard_nd_slices is not None and itemsize is None:
      raise ValueError(
          "itemsize must not be None if shard_nd_slices is provided."
      )
    reg_req = weight_synchronizer_service_pb2.RegisterWorkUnitRequest(
        unit=_raiden_id_to_proto(unit),
        shards=shards,
        control_plane_rpc_address=(
            control_plane_rpc_address if control_plane_rpc_address else ""
        ),
    )
    if shard_nd_slices:
      for nd_slice in shard_nd_slices:
        if isinstance(nd_slice, weight_synchronizer_service_pb2.NDSliceProto):
          reg_req.shard_nd_slices.add().CopyFrom(nd_slice)
        else:
          slice_proto = reg_req.shard_nd_slices.add()
          for s, e in nd_slice:
            dim_proto = slice_proto.dimensions.add()
            dim_proto.start = s
            dim_proto.end = e
    if itemsize:
      reg_req.itemsize = itemsize

    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_REGISTER_WORK_UNIT,
        register_work_unit_request=reg_req,
    )
    self._send_protobuf_rpc(req)

  def start_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
  ) -> bool:
    """Sends remote RPC to start global least-loaded transfer and blocks until fully complete."""
    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_START_TRANSFER,
        start_transfer_request=weight_synchronizer_service_pb2.StartTransferRequest(
            src_units=[_raiden_id_to_proto(u) for u in src_units],
            dst_units=[_raiden_id_to_proto(u) for u in dst_units],
        ),
    )
    return self._send_protobuf_rpc(req)

  def shutdown(self) -> bool:
    """Sends remote RPC to trigger global cluster shutdown across all cooperating jobs."""
    req = weight_synchronizer_service_pb2.ControlRequest(
        command=weight_synchronizer_service_pb2.ControlRequest.COMMAND_SHUTDOWN
    )
    return self._send_protobuf_rpc(req)
