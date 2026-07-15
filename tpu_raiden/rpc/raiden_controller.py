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
import enum
import json
import logging
import math
import os
import random
import socket
import threading
import time
import typing
from typing import Any, Optional, Union

from tpu_raiden.kv_cache import nd_slice_math
from tpu_raiden.rpc import controller_service_pb2
from tpu_raiden.rpc import raiden_service_pb2


def to_physical(logical_shape, logical_mesh_shape, minor_to_major):
  major_to_minor = list(reversed(minor_to_major))
  physical_shape = tuple(logical_shape[d] for d in major_to_minor)
  physical_mesh_shape = tuple(logical_mesh_shape[d] for d in major_to_minor)
  return physical_shape, physical_mesh_shape


class NameResolver(typing.Protocol):
  """Interface for resolving remote network coordinates (e.g.

  BNS) to raw IP addresses.
  """

  def resolve(self, address_str: str) -> str:
    ...


@typing.runtime_checkable
class RaidenEngine(typing.Protocol):
  """Standardized Data-Plane collective engine interface for Raiden Worker daemons."""

  @property
  def local_port(self) -> Optional[int]:
    """Returns the active TCP socket server listener port."""
    ...


class RaidenMemoryType(enum.IntEnum):
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


def _raiden_id_from_proto(unit: Any) -> RaidenId:
  return RaidenId(
      job_name=unit.job_name,
      job_replica_id=unit.job_replica_id,
      data_name=unit.data_name,
      data_replica_idx=unit.data_replica_idx,
  )


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
      RaidenId, dict[int, list[tuple[str, int, int, int, int, int, int]]]
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
  uuid: int = 0
  dst_mem_type: int = RaidenMemoryType.DRAM
  use_block_chunks: bool = False
  is_sender: bool = True
  expected_block_count: int = 0
  req_id: str = ""
  expected_pushes_per_pool: int = 0
  transfer_pool_indices: list[int] = dataclasses.field(default_factory=list)
  pool_dtype_tags: list[str] = dataclasses.field(default_factory=list)
  src_block_ids: dict[RaidenId, list[int]] = dataclasses.field(
      default_factory=dict
  )
  dst_device_block_ids: list[int] = dataclasses.field(default_factory=list)
  src_schedule_keys: dict[RaidenId, int] = dataclasses.field(
      default_factory=dict
  )
  parallelism: int = 1
  num_tokens: int = 0
  skipped_pool_counts: dict[str, int] = dataclasses.field(default_factory=dict)
  request_block_claim_owner: Any = dataclasses.field(
      default=None, repr=False, compare=False
  )


@dataclasses.dataclass(frozen=True)
class RequestBlockRegistration:
  """Producer-owned block IDs for one request and source work unit."""

  uuid: int
  block_ids: tuple[int, ...]
  expires_at: float


@dataclasses.dataclass
class RequestBlockCompletions:
  """Per-rank terminal votes retained until the complete PCP group votes."""

  units: set[RaidenId]
  expires_at: float


_FA_TAG = "fa"
_GDN_CONV_TAG = "gdn.conv"
_GDN_SSM_TAG = "gdn.ssm"
_TP_GATHER_DESIGN = (
    "archive/stage3-pre-controller/" "RESHARD_STAGE3_P0_2_FP8_HEAD_GATHER.md"
)


def _coerce_pool_spec_proto(pool: Any) -> Any:
  """Returns an owned PoolSpecProto from a proto, mapping, or dataclass."""
  result = raiden_service_pb2.PoolSpecProto()
  if isinstance(pool, raiden_service_pb2.PoolSpecProto):
    result.CopyFrom(pool)
    return result

  def value(name: str, default: Any = None) -> Any:
    if isinstance(pool, typing.Mapping):
      return pool.get(name, default)
    return getattr(pool, name, default)

  result.tag = str(value("tag", ""))
  result.storage_index = int(value("storage_index", 0))
  result.base_offset_bytes = int(value("base_offset_bytes", 0))
  result.block_stride_bytes = int(value("block_stride_bytes", 0))
  result.num_blocks = int(value("num_blocks", 0))
  result.dtype_tag = str(value("dtype_tag", ""))
  for region in value("regions", ()):
    if isinstance(region, typing.Mapping):
      region_value = region.get
    else:
      region_value = lambda name, default=None: getattr(region, name, default)
    region_proto = result.regions.add()
    region_proto.name = str(region_value("name", ""))
    region_proto.offset_bytes = int(region_value("offset_bytes", 0))
    region_proto.stride_bytes = int(region_value("stride_bytes", 0))
    region_proto.unit_bytes = int(region_value("unit_bytes", 0))
    region_proto.num_units = int(region_value("num_units", 0))
    region_proto.units_per_stride = int(region_value("units_per_stride", 1))
  return result


def _pool_live_bytes(pool: Any) -> int:
  return sum(
      int(region.unit_bytes)
      * int(region.num_units)
      * int(region.units_per_stride)
      for region in pool.regions
  )


def _pool_geometry_signature(pool: Any) -> tuple[Any, ...]:
  """Returns the registration fields that define one pool's byte geometry."""
  return (
      str(pool.tag),
      int(pool.storage_index),
      int(pool.base_offset_bytes),
      int(pool.block_stride_bytes),
      int(pool.num_blocks),
      str(pool.dtype_tag),
      tuple(
          (
              str(region.name),
              int(region.offset_bytes),
              int(region.stride_bytes),
              int(region.unit_bytes),
              int(region.num_units),
              int(region.units_per_stride),
          )
          for region in pool.regions
      ),
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
      proto_module: Optional[Any] = None,
  ):
    """Instantiates RPC Client with an optional initial endpoint mapping.

    Args:
      endpoint_addresses: Initial catalog of known Worker RPC addresses.
      resolve_timeout: Maximum duration in seconds to wait for a pending worker
        task to self-register before raising a Timeout RuntimeError.
      name_resolver: Interface for resolving remote coordinates (e.g. BNS).
      proto_module: Optional protobuf module to use for ControlRequest/Response.
    """
    self._endpoints = endpoint_addresses or {}
    self._pending_endpoints: dict[RaidenId, asyncio.Future[str]] = {}
    self._resolve_timeout = resolve_timeout
    self._name_resolver = name_resolver
    self._proto_module = proto_module or raiden_service_pb2

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

  def unregister_worker_endpoint(self, worker_name: RaidenId) -> None:
    """Removes a stale worker endpoint during work-unit replacement."""
    self._endpoints.pop(worker_name, None)

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

  async def _send_rpc(self, addr: str, payload: bytes) -> bytes:
    """Connects to remote address, sends payload, and returns the response bytes."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self._send_rpc_sync, addr, payload)

  def _send_rpc_sync(self, addr: str, payload: bytes) -> bytes:
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

      return resp_bytes
    finally:
      sock.close()

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
    resp_bytes = await self._send_rpc(addr, payload)
    self._verify_response(resp_bytes)

  def _raiden_id_to_proto(self, unit: RaidenId) -> Any:
    return self._proto_module.RaidenIdProto(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

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
    if (
        target_id not in transfer_plan.src_units
        and target_id not in transfer_plan.dst_units
    ):
      return None

    peers = []
    for dst in transfer_plan.dst_units:
      dst_coords = transfer_plan.worker_data_addresses.get(dst)
      if not dst_coords:
        raise ValueError(f"No data-plane endpoint registered for {dst}")
      peers.extend(dst_coords)

    req = self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_START_TRANSFER,
        peers=peers,
    )

    is_sender = target_id in transfer_plan.src_units
    start_req = self._proto_module.StartTransferRequest(
        src_units=[
            self._raiden_id_to_proto(u) for u in transfer_plan.src_units
        ],
        dst_units=[
            self._raiden_id_to_proto(u) for u in transfer_plan.dst_units
        ],
        uuid=transfer_plan.uuid,
        is_sender=is_sender,
        dst_mem_type=int(transfer_plan.dst_mem_type),
        use_block_chunks=transfer_plan.use_block_chunks,
        expected_block_count=transfer_plan.expected_block_count,
        req_id=transfer_plan.req_id,
        expected_pushes_per_pool=transfer_plan.expected_pushes_per_pool,
        transfer_pool_indices=transfer_plan.transfer_pool_indices,
        pool_dtype_tags=transfer_plan.pool_dtype_tags,
        src_block_ids=transfer_plan.src_block_ids.get(target_id, []),
        dst_device_block_ids=transfer_plan.dst_device_block_ids,
        parallelism=transfer_plan.parallelism,
        num_tokens=transfer_plan.num_tokens,
    )

    if transfer_plan.shard_push_schedules:
      if target_id in transfer_plan.dst_units:
        # Receiver path: send FILTERED plan, only containing entries for this
        # receiver
        target_endpoints = transfer_plan.worker_data_addresses.get(
            target_id, []
        )
        for (
            src_unit,
            push_schedules,
        ) in transfer_plan.shard_push_schedules.items():
          num_src_shards = len(push_schedules)
          for shard_idx, schedule in push_schedules.items():
            if num_src_shards == 1:
              key_idx = transfer_plan.src_schedule_keys.get(src_unit)
              if key_idx is None:
                if len(transfer_plan.src_units) != 1:
                  raise ValueError(
                      "A schedule key is required for every source in a "
                      "many-to-one transfer"
                  )
                key_idx = 0
            else:
              key_idx = shard_idx
            schedule_proto = self._proto_module.ShardPushScheduleProto()
            for (
                dst_peer,
                dst_shard_idx,
                dst_offset,
                src_offset,
                size,
                src_block_id,
                dst_block_id,
                src_stride,
                dst_stride,
                count,
            ) in schedule:
              if dst_peer in target_endpoints:
                entry_proto = schedule_proto.entries.add()
                entry_proto.dst_peer = dst_peer
                entry_proto.dst_shard_idx = dst_shard_idx
                entry_proto.dst_offset_bytes = dst_offset
                entry_proto.src_offset_bytes = src_offset
                entry_proto.size_bytes = size
                entry_proto.src_block_id = src_block_id
                entry_proto.dst_block_id = dst_block_id
                entry_proto.src_stride_bytes = src_stride
                entry_proto.dst_stride_bytes = dst_stride
                entry_proto.count = count
            if len(schedule_proto.entries) > 0:
              start_req.shard_push_schedules[key_idx].CopyFrom(schedule_proto)
      else:
        # Sender path: only send local schedule
        push_schedules = transfer_plan.shard_push_schedules.get(target_id)
        if push_schedules:
          for shard_idx, entries in push_schedules.items():
            schedule_proto = self._proto_module.ShardPushScheduleProto()
            for (
                dst_peer,
                dst_shard_idx,
                dst_offset,
                src_offset,
                size,
                src_block_id,
                dst_block_id,
                src_stride,
                dst_stride,
                count,
            ) in entries:
              entry_proto = schedule_proto.entries.add()
              entry_proto.dst_peer = dst_peer
              entry_proto.dst_shard_idx = dst_shard_idx
              entry_proto.dst_offset_bytes = dst_offset
              entry_proto.src_offset_bytes = src_offset
              entry_proto.size_bytes = size
              entry_proto.src_block_id = src_block_id
              entry_proto.dst_block_id = dst_block_id
              entry_proto.src_stride_bytes = src_stride
              entry_proto.dst_stride_bytes = dst_stride
              entry_proto.count = count
            start_req.shard_push_schedules[shard_idx].CopyFrom(schedule_proto)

    req.start_transfer_request.CopyFrom(start_req)
    return req.SerializeToString()

  def _verify_response(self, resp_bytes: bytes) -> None:
    """Validates demarshaled remote response bytes returned from C++ workers."""
    resp = self._proto_module.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(
          f"Raiden remote native execution failed: {resp.message}"
      )

  def get_worker_endpoints(self) -> dict[RaidenId, str]:
    """Returns active read-only snapshot of known registered Worker RPC endpoints."""
    return dict(self._endpoints)

  async def shutdown_workers(self) -> None:
    """Dispatches remote shutdown signaling payloads to all registered worker daemons."""
    payload = self._encode_shutdown()
    unique_addrs = set(self._endpoints.values())
    if unique_addrs:
      await asyncio.gather(
          *[self._send_rpc(addr, payload) for addr in unique_addrs],
          return_exceptions=True,
      )

  def _encode_shutdown(self) -> bytes:
    """Serializes domain-specific binary command for remote shutdown signaling."""
    req = self._proto_module.ControlRequest(
        command=self._proto_module.ControlRequest.COMMAND_SHUTDOWN
    )
    return req.SerializeToString()


class WeightSyncWorkerRpcClient(WorkerRpcClient):
  """Concrete domain subclass for state-of-the-art Weight Synchronizer Protobuf serialization."""

  def __init__(
      self,
      endpoint_addresses: Optional[dict[RaidenId, str]] = None,
      resolve_timeout: float = 300.0,
      name_resolver: Optional[NameResolver] = None,
  ):
    super().__init__(
        endpoint_addresses=endpoint_addresses,
        resolve_timeout=resolve_timeout,
        name_resolver=name_resolver,
        proto_module=raiden_service_pb2,
    )


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


def _proto_to_nd_slice(proto_slice: Any) -> list[tuple[int, int]]:
  """Converts an NDSliceProto message to a Python list of (start, end) tuples."""
  return [(dim.start, dim.end) for dim in proto_slice.dimensions]


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


def generate_strided_copy_chunks(
    src_shard_slice: list[tuple[int, int]],
    dst_shard_slice: list[tuple[int, int]],
    intersection_slice: list[tuple[int, int]],
    itemsize: int,
) -> list[tuple[int, int, int, int, int, int]]:
  """Translates an N-dimensional grid intersection into strided memory copy chunks.

  Instead of returning flat 1D chunks, this function groups contiguous dimension
  runs and returns strided chunk descriptors that enable hardware-accelerated
  2D/3D
  transfers without host-side scatter/gather loops.

  Args:
    src_shard_slice: Bounding box slice of the source shard across all N
      dimensions.
    dst_shard_slice: Bounding box slice of the destination shard across all N
      dimensions.
    intersection_slice: The overlapping region between source and destination
      shards.
    itemsize: Byte size of a single element (e.g., 4 for float32, 2 for
      bfloat16).

  Returns:
    A list of strided chunk descriptors where each tuple contains:
      (src_offset, dst_offset, size_bytes, src_stride_bytes, dst_stride_bytes,
      count)
  """
  rank = len(src_shard_slice)
  src_shape = [e - s for s, e in src_shard_slice]
  dst_shape = [e - s for s, e in dst_shard_slice]
  int_shape = [e - s for s, e in intersection_slice]

  # Calculate how many inner dimensions can be merged into a contiguous chunk
  split_dim = -1
  for d in range(rank - 1, -1, -1):
    dim_size = int_shape[d]
    src_full = dim_size == src_shape[d]
    dst_full = dim_size == dst_shape[d]
    if not (src_full and dst_full):
      split_dim = d
      break

  if split_dim != -1:
    contiguous_elements = math.prod(int_shape[max(1, split_dim) :])
    stride_dim = max(0, split_dim - 1)
  else:
    contiguous_elements = math.prod(int_shape)
    stride_dim = -1

  contiguous_bytes = contiguous_elements * itemsize

  src_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1]

  dst_strides = [1] * rank
  for i in range(rank - 2, -1, -1):
    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1]

  if stride_dim >= 0:
    count = int_shape[stride_dim]
    src_stride = src_strides[stride_dim] * itemsize
    dst_stride = dst_strides[stride_dim] * itemsize
    outer_shape = int_shape[:stride_dim]
  else:
    count = 1
    src_stride = 0
    dst_stride = 0
    outer_shape = []

  # Adjacent rows are one physical byte range. Normalizing this common case
  # gives the FA page planner the required count=1 contiguous descriptor even
  # when a destination page covers only part of its larger source page.
  if (
      count > 1
      and src_stride == contiguous_bytes
      and dst_stride == contiguous_bytes
  ):
    contiguous_bytes *= count
    count = 1
    src_stride = 0
    dst_stride = 0

  num_outer_elements = math.prod(outer_shape) if outer_shape else 1

  src_local_int_slice = [
      (int_s - src_s, int_e - src_s)
      for (src_s, _), (int_s, int_e) in zip(src_shard_slice, intersection_slice)
  ]
  dst_local_int_slice = [
      (int_s - dst_s, int_e - dst_s)
      for (dst_s, _), (int_s, int_e) in zip(dst_shard_slice, intersection_slice)
  ]

  chunks = []
  for i in range(num_outer_elements):
    multi_index = []
    temp = i
    for dim_size in reversed(outer_shape):
      multi_index.append(temp % dim_size)
      temp //= dim_size
    multi_index.reverse()

    src_offset_items = 0
    dst_offset_items = 0

    # Calculate offset for outer dimensions
    for d in range(len(outer_shape)):
      src_idx = src_local_int_slice[d][0] + multi_index[d]
      src_offset_items += src_idx * src_strides[d]

      dst_idx = dst_local_int_slice[d][0] + multi_index[d]
      dst_offset_items += dst_idx * dst_strides[d]

    # For merged dimensions (and the stride dim), we use the start of the intersection
    # as the base offset for this chunk
    start_d = len(outer_shape)
    for d in range(start_d, rank):
      src_offset_items += src_local_int_slice[d][0] * src_strides[d]
      dst_offset_items += dst_local_int_slice[d][0] * dst_strides[d]

    chunks.append(
        (
            src_offset_items * itemsize,
            dst_offset_items * itemsize,
            contiguous_bytes,
            src_stride,
            dst_stride,
            count,
        )
    )

  return chunks


class RaidenController:
  """High-level transfer controller managing active transfers and generating transfer plans."""

  def __init__(
      self,
      port: int,
      worker_rpc_client: Optional[WorkerRpcClient] = None,
      request_registry_ttl_s: float = 600.0,
  ):
    self.port = port
    self.broadcast_k = int(os.environ.get("RAIDEN_BROADCAST_K", "2"))
    self._active_transfers: dict[str, TransferPlan] = {}
    self._registered_shards: dict[RaidenId, list[str]] = {}
    self._registered_mesh_shapes: dict[RaidenId, list[int]] = {}
    self._registered_layouts: dict[RaidenId, list[int]] = {}
    self._registered_global_shapes: dict[RaidenId, list[int]] = {}
    self._registered_itemsizes: dict[RaidenId, int] = {}
    self._registered_pool_manifests: dict[RaidenId, list[Any]] = {}
    self._registered_layout_fingerprints: dict[RaidenId, str] = {}
    self._registered_page_tokens: dict[RaidenId, int] = {}
    self._registered_page_slice_tokens: dict[RaidenId, int] = {}
    self._registered_transfer_parallelism: dict[RaidenId, int] = {}
    self._registered_transfer_ranks: dict[RaidenId, int] = {}
    self._request_blocks: dict[
        tuple[str, RaidenId], RequestBlockRegistration
    ] = {}
    # Request-block lifecycle is keyed by the request's generation UUID. A
    # planner claim freezes the D5 snapshot against cancellation; a cancellation
    # tombstone prevents a producer's late D5 registration from resurrecting a
    # request whose consumer has already given up. Both markers expire with the
    # registry TTL so an abandoned request does not leak lifecycle state.
    self._claimed_request_blocks: dict[tuple[str, int], float] = {}
    self._claimed_request_block_units: dict[
        tuple[str, int], frozenset[RaidenId]
    ] = {}
    self._claimed_request_block_owners: dict[tuple[str, int], Any] = {}
    self._completed_request_block_units: dict[
        tuple[str, int], RequestBlockCompletions
    ] = {}
    self._cancelled_request_blocks: dict[tuple[str, int], float] = {}
    if request_registry_ttl_s <= 0:
      raise ValueError("request_registry_ttl_s must be positive")
    self._request_registry_ttl_s = request_registry_ttl_s
    self._lock = threading.Lock()
    self.worker_rpc_client = worker_rpc_client or WorkerRpcClient()

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      mesh_shape: Optional[typing.Sequence[int]] = None,
      layout: Optional[typing.Sequence[int]] = None,
      global_shape: Optional[typing.Sequence[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[typing.Sequence[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      page_slice_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
  ) -> None:
    """Registers physical worker shard Data addresses and optional Control-Plane RPC endpoint."""
    has_metadata = (
        mesh_shape is not None or layout is not None or global_shape is not None
    )
    if has_metadata:
      if mesh_shape is None or layout is None or global_shape is None:
        raise ValueError(
            "If any of mesh_shape, layout, or global_shape is provided, "
            "all of them must be provided to enable centralized slice planning."
        )
      if itemsize is None or itemsize <= 0:
        raise ValueError(
            "itemsize must be provided and must be greater than 0 if resharding"
            " metadata is provided."
        )

    has_reshard_metadata = any(
        value is not None
        for value in (
            pool_manifest,
            layout_fingerprint,
            page_tokens,
            page_slice_tokens,
            transfer_parallelism,
            transfer_rank,
        )
    )
    if has_reshard_metadata:
      if (
          pool_manifest is None
          or layout_fingerprint is None
          or page_tokens is None
          or page_slice_tokens is None
          or transfer_parallelism is None
          or transfer_rank is None
      ):
        raise ValueError(
            "pool_manifest, layout_fingerprint, page_tokens, "
            "page_slice_tokens, "
            "transfer_parallelism, and transfer_rank must be provided "
            "together for FA resharding"
        )
      if not pool_manifest:
        raise ValueError("pool_manifest must not be empty")
      if not layout_fingerprint:
        raise ValueError("layout_fingerprint must not be empty")
      if page_tokens <= 0:
        raise ValueError("page_tokens must be positive")
      if page_slice_tokens <= 0:
        raise ValueError("page_slice_tokens must be positive")
      if page_tokens % page_slice_tokens != 0:
        raise ValueError("page_slice_tokens must divide page_tokens")
      if transfer_parallelism <= 0:
        raise ValueError("transfer_parallelism must be positive")
      if transfer_rank < 0:
        raise ValueError("transfer_rank must be non-negative")
      if transfer_rank >= transfer_parallelism:
        raise ValueError("transfer_rank must be less than transfer_parallelism")
      normalized_pools = [_coerce_pool_spec_proto(p) for p in pool_manifest]
    else:
      normalized_pools = []

    if not shards or any(not shard for shard in shards):
      raise ValueError("shards must contain at least one non-empty endpoint")
    if len(set(shards)) != len(shards):
      raise ValueError("shards must not contain duplicate endpoints")

    with self._lock:
      self._registered_shards[unit] = list(shards)
      # A worker restart invalidates every request-local physical block ID
      # previously reported by that identity.
      for key in [key for key in self._request_blocks if key[1] == unit]:
        del self._request_blocks[key]
      # Registration is replacement, not a patch: stale optional metadata
      # must disappear when a unit restarts with a different payload.
      for registry in (
          self._registered_mesh_shapes,
          self._registered_layouts,
          self._registered_global_shapes,
          self._registered_itemsizes,
          self._registered_pool_manifests,
          self._registered_layout_fingerprints,
          self._registered_page_tokens,
          self._registered_page_slice_tokens,
          self._registered_transfer_parallelism,
          self._registered_transfer_ranks,
      ):
        registry.pop(unit, None)
      if mesh_shape is not None:
        self._registered_mesh_shapes[unit] = list(mesh_shape)
      if layout is not None:
        self._registered_layouts[unit] = list(layout)
      if global_shape is not None:
        self._registered_global_shapes[unit] = list(global_shape)
      if itemsize is not None:
        self._registered_itemsizes[unit] = itemsize
      if has_reshard_metadata:
        self._registered_pool_manifests[unit] = normalized_pools
        self._registered_layout_fingerprints[unit] = layout_fingerprint
        self._registered_page_tokens[unit] = page_tokens
        self._registered_page_slice_tokens[unit] = page_slice_tokens
        self._registered_transfer_parallelism[unit] = transfer_parallelism
        self._registered_transfer_ranks[unit] = transfer_rank
      if control_plane_rpc_address and hasattr(
          self.worker_rpc_client, "register_worker_endpoint"
      ):
        self.worker_rpc_client.register_worker_endpoint(
            unit, control_plane_rpc_address
        )
      elif hasattr(self.worker_rpc_client, "unregister_worker_endpoint"):
        self.worker_rpc_client.unregister_worker_endpoint(unit)

  def _metadata_proto_locked(self, unit: RaidenId) -> Any:
    """Builds an owned registration proto while `_lock` is held."""
    reg_req = raiden_service_pb2.RegisterWorkUnitRequest(
        unit=raiden_service_pb2.RaidenIdProto(
            job_name=unit.job_name,
            job_replica_id=unit.job_replica_id,
            data_name=unit.data_name,
            data_replica_idx=unit.data_replica_idx,
        ),
        shards=self._registered_shards[unit],
        control_plane_rpc_address=(
            self.worker_rpc_client.get_worker_endpoints().get(unit, "")
        ),
        itemsize=self._registered_itemsizes.get(unit, 0),
        layout_fingerprint=self._registered_layout_fingerprints.get(unit, ""),
        page_tokens=self._registered_page_tokens.get(unit, 0),
        page_slice_tokens=self._registered_page_slice_tokens.get(unit, 0),
        transfer_parallelism=self._registered_transfer_parallelism.get(unit, 0),
        transfer_rank=self._registered_transfer_ranks.get(unit, 0),
    )
    reg_req.mesh_shape.extend(self._registered_mesh_shapes.get(unit, ()))
    reg_req.layout.extend(self._registered_layouts.get(unit, ()))
    reg_req.global_shape.extend(self._registered_global_shapes.get(unit, ()))
    for pool in self._registered_pool_manifests.get(unit, ()):
      reg_req.pools.add().CopyFrom(pool)
    return reg_req

  def get_all_metadata(self) -> list[Any]:
    """Returns replacement-safe metadata for all registered work units."""
    with self._lock:
      return [
          self._metadata_proto_locked(unit) for unit in self._registered_shards
      ]

  def _resolve_shards(self, unit: RaidenId) -> list[str]:
    with self._lock:
      shards = self._registered_shards.get(unit)
      if not shards:
        raise ValueError(f"Work unit is not registered: {unit}")
      return list(shards)

  def _purge_expired_request_blocks_locked(self, now: float) -> None:
    expired = [
        key
        for key, registration in self._request_blocks.items()
        if registration.expires_at <= now
    ]
    for key in expired:
      del self._request_blocks[key]

  def _purge_expired_request_block_lifecycle_locked(self, now: float) -> None:
    for lifecycle in (
        self._claimed_request_blocks,
        self._cancelled_request_blocks,
    ):
      expired = [
          key for key, expires_at in lifecycle.items() if expires_at <= now
      ]
      for key in expired:
        del lifecycle[key]
    expired_completions = [
        key
        for key, completion in self._completed_request_block_units.items()
        if completion.expires_at <= now
    ]
    for key in expired_completions:
      del self._completed_request_block_units[key]
    for key in list(self._claimed_request_block_units):
      if key not in self._claimed_request_blocks:
        del self._claimed_request_block_units[key]
        self._claimed_request_block_owners.pop(key, None)

  def _retire_request_blocks_locked(
      self, lifecycle_key: tuple[str, int]
  ) -> int:
    req_id, uuid = lifecycle_key
    keys = [
        key
        for key, registration in self._request_blocks.items()
        if key[0] == req_id and registration.uuid == uuid
    ]
    for key in keys:
      del self._request_blocks[key]
    self._claimed_request_blocks.pop(lifecycle_key, None)
    self._claimed_request_block_units.pop(lifecycle_key, None)
    self._claimed_request_block_owners.pop(lifecycle_key, None)
    self._completed_request_block_units.pop(lifecycle_key, None)
    self._cancelled_request_blocks.pop(lifecycle_key, None)
    return len(keys)

  def register_request_blocks(
      self,
      req_id: str,
      uuid: int,
      unit: RaidenId,
      block_ids: typing.Sequence[int],
  ) -> None:
    """Registers one producer rank's request-local physical block IDs."""
    if not req_id:
      raise ValueError("req_id must not be empty")
    if uuid <= 0:
      raise ValueError("uuid must be positive")
    normalized = tuple(int(block_id) for block_id in block_ids)
    if any(block_id < 0 for block_id in normalized):
      raise ValueError("block_ids must be non-negative")
    if len(set(normalized)) != len(normalized):
      raise ValueError("block_ids must not contain duplicates")

    now = time.monotonic()
    registration = RequestBlockRegistration(
        uuid=uuid,
        block_ids=normalized,
        expires_at=now + self._request_registry_ttl_s,
    )
    key = (req_id, unit)
    lifecycle_key = (req_id, uuid)
    with self._lock:
      self._purge_expired_request_blocks_locked(now)
      self._purge_expired_request_block_lifecycle_locked(now)
      if unit not in self._registered_shards:
        raise ValueError(f"Work unit is not registered: {unit}")
      if lifecycle_key in self._cancelled_request_blocks:
        raise ValueError(
            "Request block registration was cancelled for "
            f"req_id={req_id}, uuid={uuid}"
        )
      if lifecycle_key in self._claimed_request_blocks:
        raise ValueError(
            "Request block registration is already claimed for "
            f"req_id={req_id}, uuid={uuid}"
        )
      existing = self._request_blocks.get(key)
      if existing is not None:
        if existing.uuid != uuid or existing.block_ids != normalized:
          raise ValueError(
              "Conflicting request block registration for "
              f"req_id={req_id}, unit={unit}"
          )
        # Duplicate request-finish notifications are idempotent and refresh
        # the TTL while preserving the exact registered payload.
      self._request_blocks[key] = registration

  def release_request_blocks(self, req_id: str, uuid: int) -> int:
    """Administratively force-retires all state for one generation."""
    if not req_id:
      raise ValueError("req_id must not be empty")
    if uuid <= 0:
      raise ValueError("uuid must be positive")
    now = time.monotonic()
    with self._lock:
      self._purge_expired_request_blocks_locked(now)
      self._purge_expired_request_block_lifecycle_locked(now)
      lifecycle_key = (req_id, uuid)
      return self._retire_request_blocks_locked(lifecycle_key)

  def complete_request_blocks(
      self, req_id: str, uuid: int, unit: RaidenId
  ) -> int:
    """Records one native-terminal vote and retires after all claimed ranks."""
    if not req_id:
      raise ValueError("req_id must not be empty")
    if uuid <= 0:
      raise ValueError("uuid must be positive")
    now = time.monotonic()
    with self._lock:
      self._purge_expired_request_blocks_locked(now)
      self._purge_expired_request_block_lifecycle_locked(now)
      lifecycle_key = (req_id, uuid)
      expected_units = self._claimed_request_block_units.get(lifecycle_key)
      registration = self._request_blocks.get((req_id, unit))
      if (
          expected_units is None
          and registration is not None
          and registration.uuid == uuid
          and registration.block_ids
      ):
        raise ValueError(
            "Only an empty producer registration may complete before the "
            "request block snapshot is claimed"
        )
      if (registration is None or registration.uuid != uuid) and (
          expected_units is None or unit not in expected_units
      ):
        # Aggregate completion is idempotent after the last rank has
        # already retired the generation.
        return 0
      completion = self._completed_request_block_units.get(lifecycle_key)
      if completion is None:
        completion = RequestBlockCompletions(
            units=set(), expires_at=now + self._request_registry_ttl_s
        )
        self._completed_request_block_units[lifecycle_key] = completion
      completion.units.add(unit)
      completion.expires_at = now + self._request_registry_ttl_s
      if expected_units is not None and expected_units <= completion.units:
        return self._retire_request_blocks_locked(lifecycle_key)
      return 0

  def cancel_request_blocks_if_unclaimed(self, req_id: str, uuid: int) -> bool:
    """Atomically cancels an unclaimed D5 snapshot.

    Returns True when cancellation owns the request (including an idempotent
    repeat), or False once a planner lookup has claimed the snapshot. A
    successful cancellation removes every current rank registration and leaves
    a tombstone that rejects late registration and lookup until force release.
    """
    if not req_id:
      raise ValueError("req_id must not be empty")
    if uuid <= 0:
      raise ValueError("uuid must be positive")
    now = time.monotonic()
    lifecycle_key = (req_id, uuid)
    with self._lock:
      self._purge_expired_request_block_lifecycle_locked(now)
      if lifecycle_key in self._cancelled_request_blocks:
        return True
      if lifecycle_key in self._claimed_request_blocks:
        return False

      self._cancelled_request_blocks[lifecycle_key] = (
          now + self._request_registry_ttl_s
      )
      self._completed_request_block_units.pop(lifecycle_key, None)
      self._claimed_request_block_units.pop(lifecycle_key, None)
      self._claimed_request_block_owners.pop(lifecycle_key, None)
      keys = [
          key
          for key, registration in self._request_blocks.items()
          if key[0] == req_id and registration.uuid == uuid
      ]
      for key in keys:
        del self._request_blocks[key]
      return True

  def _lookup_request_blocks(
      self,
      req_id: str,
      uuid: int,
      units: typing.Sequence[RaidenId],
      claim_owner: Any = None,
  ) -> dict[RaidenId, list[int]]:
    now = time.monotonic()
    lifecycle_key = (req_id, uuid)
    with self._lock:
      self._purge_expired_request_blocks_locked(now)
      self._purge_expired_request_block_lifecycle_locked(now)
      if lifecycle_key in self._cancelled_request_blocks:
        raise ValueError(
            "Request block registration was cancelled for "
            f"req_id={req_id}, uuid={uuid}"
        )
      claimed_units = frozenset(units)
      existing_claimed_units = self._claimed_request_block_units.get(
          lifecycle_key
      )
      if existing_claimed_units is not None:
        existing_owner = self._claimed_request_block_owners.get(lifecycle_key)
        if existing_owner is not claim_owner:
          raise ValueError(
              "Request block snapshot is already claimed by "
              "another planning attempt"
          )
        if existing_claimed_units != claimed_units:
          raise ValueError(
              "Request block snapshot was already claimed for a "
              "different source unit set"
          )
      result: dict[RaidenId, list[int]] = {}
      for unit in units:
        registration = self._request_blocks.get((req_id, unit))
        if registration is None or registration.uuid != uuid:
          raise ValueError(
              "Missing producer block registration for "
              f"req_id={req_id}, uuid={uuid}, unit={unit}"
          )
        result[unit] = list(registration.block_ids)
      # This is the claim linearization point: every requested rank has been
      # validated and copied while cancellation is excluded by `_lock`.
      self._claimed_request_blocks[lifecycle_key] = (
          now + self._request_registry_ttl_s
      )
      self._claimed_request_block_units[lifecycle_key] = claimed_units
      self._claimed_request_block_owners[lifecycle_key] = claim_owner
      completion = self._completed_request_block_units.get(lifecycle_key)
      if completion is not None:
        completion.expires_at = now + self._request_registry_ttl_s
        if claimed_units <= completion.units:
          self._retire_request_blocks_locked(lifecycle_key)
      return result

  def _abandon_request_blocks_claim(
      self, req_id: str, uuid: int, claim_owner: Any
  ) -> bool:
    """Rolls back a claim before any sender has been dispatched."""
    lifecycle_key = (req_id, uuid)
    with self._lock:
      if (
          lifecycle_key not in self._claimed_request_blocks
          or self._claimed_request_block_owners.get(lifecycle_key)
          is not claim_owner
      ):
        return False
      self._claimed_request_blocks.pop(lifecycle_key, None)
      self._claimed_request_block_units.pop(lifecycle_key, None)
      self._claimed_request_block_owners.pop(lifecycle_key, None)
      return True

  def get_plan(self, req_id: str) -> Optional[TransferPlan]:
    """Returns the generated TransferPlan for a given transfer request ID."""
    with self._lock:
      return self._active_transfers.get(req_id)

  async def _query_remote_metadata(self, addr: str) -> list[Any]:
    req = raiden_service_pb2.ControlRequest(
        command=raiden_service_pb2.ControlRequest.COMMAND_GET_METADATA
    )
    resp_bytes = await self.worker_rpc_client._send_rpc(
        addr, req.SerializeToString()
    )
    resp = raiden_service_pb2.ControlResponse()
    resp.ParseFromString(resp_bytes)
    if not resp.success:
      raise RuntimeError(f"Failed to query remote metadata: {resp.message}")
    return list(resp.get_metadata_response.metadata)

  def _get_local_metadata(self, units: list[RaidenId]) -> list[Any]:
    with self._lock:
      missing = [unit for unit in units if unit not in self._registered_shards]
      if missing:
        raise ValueError(f"Work units are not registered: {missing}")
      return [self._metadata_proto_locked(unit) for unit in units]

  @staticmethod
  def _metadata_by_unit(
      metadata: typing.Sequence[Any], units: typing.Sequence[RaidenId]
  ) -> dict[RaidenId, Any]:
    """Selects exact requested metadata and rejects duplicate identities."""
    requested = set(units)
    result = {}
    for item in metadata:
      unit = _raiden_id_from_proto(item.unit)
      if unit not in requested:
        continue
      if unit in result:
        raise ValueError(f"Duplicate registration metadata for {unit}")
      result[unit] = item
    missing = [unit for unit in units if unit not in result]
    if missing:
      raise ValueError(f"Missing registration metadata for {missing}")
    return result

  def _build_fa_reshard_plan(
      self,
      *,
      src_units: typing.Sequence[RaidenId],
      dst_units: typing.Sequence[RaidenId],
      src_metadata: typing.Sequence[Any],
      dst_metadata: typing.Sequence[Any],
      req_id: str,
      uuid: int,
      dst_device_block_ids: typing.Sequence[int],
      num_tokens: int,
      parallelism: Optional[int],
  ) -> TransferPlan:
    """Builds a plan and rolls back any claim if planning fails."""
    claim_owner = object()
    try:
      plan = self._build_fa_reshard_plan_claimed(
          src_units=src_units,
          dst_units=dst_units,
          src_metadata=src_metadata,
          dst_metadata=dst_metadata,
          req_id=req_id,
          uuid=uuid,
          dst_device_block_ids=dst_device_block_ids,
          num_tokens=num_tokens,
          parallelism=parallelism,
          claim_owner=claim_owner,
      )
      plan.request_block_claim_owner = claim_owner
      return plan
    except Exception:
      self._abandon_request_blocks_claim(req_id, uuid, claim_owner)
      raise

  def _build_fa_reshard_plan_claimed(
      self,
      *,
      src_units: typing.Sequence[RaidenId],
      dst_units: typing.Sequence[RaidenId],
      src_metadata: typing.Sequence[Any],
      dst_metadata: typing.Sequence[Any],
      req_id: str,
      uuid: int,
      dst_device_block_ids: typing.Sequence[int],
      num_tokens: int,
      parallelism: Optional[int],
      claim_owner: Any,
  ) -> TransferPlan:
    """Builds the fail-closed PCP-interleaved to TP1 FA page plan."""
    if not req_id:
      raise ValueError("req_id must not be empty for FA resharding")
    if uuid <= 0:
      raise ValueError("uuid must be positive for FA resharding")
    if num_tokens <= 0:
      raise ValueError("num_tokens must be positive for FA resharding")
    if len(dst_units) != 1:
      raise ValueError(
          "Stage-3 FA resharding requires exactly one TP1 destination unit"
      )
    if len(set(src_units)) != len(src_units):
      raise ValueError("src_units must not contain duplicates")
    if len(set(dst_units)) != len(dst_units):
      raise ValueError("dst_units must not contain duplicates")

    dst_ids = [int(block_id) for block_id in dst_device_block_ids]
    if not dst_ids:
      raise ValueError("dst_device_block_ids must not be empty")
    if any(block_id < 0 for block_id in dst_ids):
      raise ValueError("dst_device_block_ids must be non-negative")
    if len(set(dst_ids)) != len(dst_ids):
      raise ValueError("dst_device_block_ids must not contain duplicates")

    src_by_unit = self._metadata_by_unit(src_metadata, src_units)
    dst_by_unit = self._metadata_by_unit(dst_metadata, dst_units)
    dst_unit = dst_units[0]
    dst_meta = dst_by_unit[dst_unit]

    all_metadata = [src_by_unit[unit] for unit in src_units] + [dst_meta]
    for meta in all_metadata:
      unit = _raiden_id_from_proto(meta.unit)
      if not meta.layout_fingerprint:
        raise ValueError(f"Missing layout fingerprint for {unit}")
      if not meta.pools:
        raise ValueError(f"Missing explicit pool manifest for {unit}")
      if meta.page_tokens <= 0:
        raise ValueError(f"Missing positive page geometry for {unit}")
      if meta.page_slice_tokens <= 0:
        raise ValueError(f"Missing positive page-slice geometry for {unit}")
      if meta.page_tokens % meta.page_slice_tokens != 0:
        raise ValueError(
            f"page_slice_tokens must divide page_tokens for {unit}"
        )
      if not meta.shards:
        raise ValueError(f"Missing data-plane endpoint for {unit}")
      if len(meta.shards) != 1:
        raise ValueError(
            "Stage-3 PCP-to-TP1 planning requires one data-plane endpoint "
            f"per work unit; {unit} registered {len(meta.shards)}"
        )
      if not meta.control_plane_rpc_address:
        raise ValueError(f"Missing control-plane endpoint for {unit}")

    fingerprints = {meta.layout_fingerprint for meta in all_metadata}
    if len(fingerprints) != 1:
      raise ValueError(
          "FA layout fingerprint mismatch between source and destination"
      )

    dst_identity = tuple(
        (str(pool.tag), str(pool.dtype_tag)) for pool in dst_meta.pools
    )
    if not dst_identity:
      raise ValueError("Destination pool manifest must not be empty")
    for src_unit in src_units:
      src_identity = tuple(
          (str(pool.tag), str(pool.dtype_tag))
          for pool in src_by_unit[src_unit].pools
      )
      if src_identity != dst_identity:
        raise ValueError(
            "Canonical pool manifest mismatch between source and "
            f"destination for {src_unit}"
        )

    reference_src = src_by_unit[src_units[0]]
    reference_geometry = tuple(
        _pool_geometry_signature(pool) for pool in reference_src.pools
    )
    for src_unit in src_units[1:]:
      geometry = tuple(
          _pool_geometry_signature(pool) for pool in src_by_unit[src_unit].pools
      )
      if geometry != reference_geometry:
        raise ValueError(
            f"Source pool geometry differs across PCP ranks at {src_unit}"
        )

    fa_pool_indices = [
        index
        for index, pool in enumerate(dst_meta.pools)
        if pool.tag == _FA_TAG
    ]
    if not fa_pool_indices:
      raise ValueError("Pool manifest contains no FA pools")

    src_page_tokens = int(reference_src.page_tokens)
    src_page_slice_tokens = int(reference_src.page_slice_tokens)
    dst_page_tokens = int(dst_meta.page_tokens)
    dst_page_slice_tokens = int(dst_meta.page_slice_tokens)
    if dst_page_slice_tokens != dst_page_tokens:
      raise ValueError(
          "TP1 destination must explicitly declare contiguous page geometry: "
          "page_slice_tokens must equal page_tokens"
      )
    source_page_tokens = {
        int(src_by_unit[unit].page_tokens) for unit in src_units
    }
    if source_page_tokens != {src_page_tokens}:
      raise ValueError("Source page_tokens differ across PCP ranks")
    source_page_slice_tokens = {
        int(src_by_unit[unit].page_slice_tokens) for unit in src_units
    }
    if source_page_slice_tokens != {src_page_slice_tokens}:
      raise ValueError("Source page_slice_tokens differ across PCP ranks")

    token_bytes_values = set()
    for pool_idx in fa_pool_indices:
      src_pool = reference_src.pools[pool_idx]
      dst_pool = dst_meta.pools[pool_idx]
      src_live = _pool_live_bytes(src_pool)
      dst_live = _pool_live_bytes(dst_pool)
      if src_live != int(src_pool.block_stride_bytes):
        raise ValueError(
            "Stage-3 FA raw transfer requires full-live source pools; "
            f"pool {pool_idx} has live={src_live}, "
            f"stride={src_pool.block_stride_bytes}. See {_TP_GATHER_DESIGN}"
        )
      if dst_live != int(dst_pool.block_stride_bytes):
        raise ValueError(
            "Stage-3 FA raw transfer rejects decode TP>1 or padded pools; "
            f"pool {pool_idx} has live={dst_live}, "
            f"stride={dst_pool.block_stride_bytes}. See {_TP_GATHER_DESIGN}"
        )
      if src_live % src_page_tokens != 0:
        raise ValueError(
            f"Source FA pool {pool_idx} live bytes are not token-aligned"
        )
      if dst_live % dst_page_tokens != 0:
        raise ValueError(
            f"Destination FA pool {pool_idx} live bytes are not token-aligned"
        )
      src_token_bytes = src_live // src_page_tokens
      dst_token_bytes = dst_live // dst_page_tokens
      if src_token_bytes != dst_token_bytes:
        raise ValueError(
            f"FA token byte geometry mismatch for pool {pool_idx}: "
            f"source={src_token_bytes}, destination={dst_token_bytes}"
        )
      token_bytes_values.add(src_token_bytes)
    if len(token_bytes_values) != 1:
      raise ValueError("All FA pools must have the same bytes-per-token")
    token_bytes = next(iter(token_bytes_values))
    if token_bytes <= 0:
      raise ValueError("FA bytes-per-token must be positive")

    admitted_parallelism = {
        int(src_by_unit[unit].transfer_parallelism) for unit in src_units
    }
    if len(admitted_parallelism) != 1 or 0 in admitted_parallelism:
      raise ValueError(
          "All source ranks must declare one consistent positive "
          "transfer_parallelism"
      )
    topology_parallelism = next(iter(admitted_parallelism))
    ranks = {int(src_by_unit[unit].transfer_rank): unit for unit in src_units}
    if len(ranks) != len(src_units):
      raise ValueError("Source transfer_rank values must be unique")
    if sorted(ranks) != list(range(len(src_units))):
      raise ValueError(
          "Source transfer_rank values must be contiguous from zero"
      )
    if topology_parallelism != len(src_units):
      raise ValueError(
          "Source transfer_parallelism must equal the complete PCP rank "
          f"count: admitted={topology_parallelism}, ranks={len(src_units)}"
      )
    requested_parallelism = (
        topology_parallelism if parallelism is None else int(parallelism)
    )
    if requested_parallelism <= 0:
      raise ValueError("parallelism must be positive")
    if requested_parallelism > topology_parallelism:
      raise ValueError(
          "parallelism exceeds source admission: "
          f"requested={requested_parallelism}, "
          f"admitted={topology_parallelism}"
      )

    expected_dst_pages = math.ceil(num_tokens / dst_page_tokens)
    if len(dst_ids) != expected_dst_pages:
      raise ValueError(
          "dst_device_block_ids count does not match destination page count: "
          f"got={len(dst_ids)}, expected={expected_dst_pages}"
      )

    registered_src_ids = self._lookup_request_blocks(
        req_id, uuid, src_units, claim_owner=claim_owner
    )
    chunks_per_src_block = src_page_tokens // src_page_slice_tokens
    interleave_cycle_tokens = src_page_slice_tokens * topology_parallelism
    for rank, src_unit in ranks.items():
      first_rank_token = rank * src_page_slice_tokens
      if num_tokens <= first_rank_token:
        owned_interleave_cycles = 0
      else:
        owned_interleave_cycles = 1 + (
            (num_tokens - 1 - first_rank_token) // interleave_cycle_tokens
        )
      expected_src_blocks = math.ceil(
          owned_interleave_cycles / chunks_per_src_block
      )
      actual_src_blocks = registered_src_ids[src_unit]
      if len(actual_src_blocks) != expected_src_blocks:
        raise ValueError(
            "Source block-id count does not match interleaved page ownership "
            "for "
            f"{src_unit}: got={len(actual_src_blocks)}, "
            f"expected={expected_src_blocks}"
        )

    for pool_idx in fa_pool_indices:
      src_num_blocks = int(reference_src.pools[pool_idx].num_blocks)
      dst_num_blocks = int(dst_meta.pools[pool_idx].num_blocks)
      for src_unit, block_ids in registered_src_ids.items():
        if any(block_id >= src_num_blocks for block_id in block_ids):
          raise ValueError(
              f"Source block id is out of range for FA pool {pool_idx} at "
              f"{src_unit}"
          )
      if any(block_id >= dst_num_blocks for block_id in dst_ids):
        raise ValueError(
            f"Destination block id is out of range for FA pool {pool_idx}"
        )

    dst_peer = str(dst_meta.shards[0])
    schedules: dict[RaidenId, dict[int, list[tuple[Any, ...]]]] = {}
    transfer_pairs_per_sender: dict[RaidenId, set[tuple[str, int, int]]] = {}
    for dst_page_ordinal, dst_block_id in enumerate(dst_ids):
      dst_global_start = dst_page_ordinal * dst_page_tokens
      dst_global_end = min(dst_global_start + dst_page_tokens, num_tokens)
      cursor = dst_global_start
      while cursor < dst_global_end:
        interleave_cycle = cursor // interleave_cycle_tokens
        offset_in_cycle = cursor % interleave_cycle_tokens
        owner_rank = offset_in_cycle // src_page_slice_tokens
        src_unit = ranks[owner_rank]
        owner_slice_start = (
            interleave_cycle * interleave_cycle_tokens
            + owner_rank * src_page_slice_tokens
        )
        owner_slice_end = owner_slice_start + src_page_slice_tokens
        chunk_end = min(owner_slice_end, dst_global_end)
        if chunk_end <= cursor:
          raise AssertionError("Planner failed to advance interleave cursor")

        local_src_block = interleave_cycle // chunks_per_src_block
        src_block_id = registered_src_ids[src_unit][local_src_block]
        chunk_in_src_block = interleave_cycle % chunks_per_src_block
        src_token_offset = (
            chunk_in_src_block * src_page_slice_tokens
            + cursor
            - owner_slice_start
        )
        dst_token_offset = cursor - dst_global_start
        entry = (
            dst_peer,
            0,
            dst_token_offset * token_bytes,
            src_token_offset * token_bytes,
            (chunk_end - cursor) * token_bytes,
            src_block_id,
            dst_block_id,
            0,
            0,
            1,
        )
        schedules.setdefault(src_unit, {}).setdefault(0, []).append(entry)
        transfer_pairs_per_sender.setdefault(src_unit, set()).add(
            (dst_peer, src_block_id, dst_block_id)
        )
        cursor = chunk_end

    active_src_units = [
        ranks[rank] for rank in sorted(ranks) if ranks[rank] in schedules
    ]
    expected_pushes_per_pool = sum(
        min(requested_parallelism, len(transfer_pairs_per_sender[unit]))
        for unit in active_src_units
    )
    if expected_pushes_per_pool <= 0:
      raise ValueError("FA reshard plan contains no source pushes")

    worker_rpc_addresses = {}
    for meta in all_metadata:
      worker_rpc_addresses[_raiden_id_from_proto(meta.unit)] = str(
          meta.control_plane_rpc_address
      )
    worker_data_addresses = {dst_unit: [dst_peer]}
    src_schedule_keys = {
        unit: ordinal for ordinal, unit in enumerate(active_src_units)
    }
    skipped_pool_counts = {}
    for pool in dst_meta.pools:
      if pool.tag != _FA_TAG:
        skipped_pool_counts[pool.tag] = skipped_pool_counts.get(pool.tag, 0) + 1

    return TransferPlan(
        src_units=active_src_units,
        dst_units=[dst_unit],
        plan={},
        shard_push_schedules=schedules,
        worker_rpc_addresses=worker_rpc_addresses,
        worker_data_addresses=worker_data_addresses,
        uuid=uuid,
        dst_mem_type=RaidenMemoryType.HBM,
        use_block_chunks=True,
        is_sender=True,
        expected_block_count=len(dst_ids),
        req_id=req_id,
        expected_pushes_per_pool=expected_pushes_per_pool,
        transfer_pool_indices=fa_pool_indices,
        pool_dtype_tags=[str(pool.dtype_tag) for pool in dst_meta.pools],
        src_block_ids={
            unit: list(registered_src_ids[unit]) for unit in active_src_units
        },
        dst_device_block_ids=dst_ids,
        src_schedule_keys=src_schedule_keys,
        parallelism=requested_parallelism,
        num_tokens=num_tokens,
        skipped_pool_counts=skipped_pool_counts,
    )

  async def _execute_slice_broadcast(
      self,
      key: tuple,
      targets: list[tuple],
      final_plan: TransferPlan,
      fanout_k: int,
      req_id: str,
      uuid: int,
      dst_mem_type: int,
      expected_block_count: int,
      dst_controller_address: Optional[str],
  ) -> None:
    """Executes a decentralized, greedy K-ary tree broadcast for a slice of data across destination workers.

    Schedules tree hops dynamically by assigning idle destination workers that
    have received
    and unpacked their slice as active source nodes for subsequent downstream
    destinations.

    Args:
      key: Tuple identifying the slice: (src_unit, shard_idx, src_block_id,
        src_block_offset, size, src_stride, count).
      targets: List of destination worker target tuples awaiting this slice.
      final_plan: Reference to the global multi-host TransferPlan.
      fanout_k: Maximum number of concurrent child transfers per active source
        node.
      req_id: Unique string identifier for this multi-hop transfer schedule.
      uuid: Numerical transaction ID across all participating workers.
      dst_mem_type: Memory tier destination (e.g. HBM / HOST).
      expected_block_count: Total physical block pushes expected by each
        destination worker.
      dst_controller_address: BNS/IP network address of the destination
        controller.
      src_controller_address: BNS/IP network address of the source controller.
    """
    # key: (src_unit, shard_idx, src_block_id, src_block_offset, size, src_stride, count)
    (
        src_unit,
        shard_idx,
        src_block_id,
        src_block_offset,
        size,
        src_stride,
        count,
    ) = key

    available_sources = [src_unit]
    node_slice_offsets = {
        src_unit: (src_block_id, src_block_offset, src_stride)
    }

    pending_targets = list(targets)
    active_pushes = {u: 0 for u in [src_unit] + [t[0] for t in targets]}
    transfers_in_progress = {}

    while pending_targets or transfers_in_progress:
      # 1. Greedy assignment step
      scheduled_any = False
      for s in list(available_sources):
        while active_pushes[s] < fanout_k and pending_targets:
          t = pending_targets.pop(0)
          (
              dst_unit,
              dst_peer,
              dst_shard_idx,
              dst_block_id,
              dst_block_offset,
              dst_stride,
          ) = t

          active_pushes[s] += 1
          scheduled_any = True

          s_block_id, s_block_offset, s_stride = node_slice_offsets[s]

          entry = (
              dst_peer,
              dst_shard_idx,
              dst_block_offset,
              s_block_offset,
              size,
              s_block_id,
              dst_block_id,
              s_stride,
              dst_stride,
              count,
          )

          sub_schedule = {s: {shard_idx if s == src_unit else 0: [entry]}}
          sub_plan = TransferPlan(
              src_units=[s],
              dst_units=[dst_unit],
              plan=None,
              shard_push_schedules=sub_schedule,
              worker_rpc_addresses=dict(final_plan.worker_rpc_addresses),
              worker_data_addresses=dict(final_plan.worker_data_addresses),
              uuid=uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=True,
              is_sender=True,
              expected_block_count=expected_block_count,
              req_id=req_id,
          )

          async def _run_single_transfer(s_node, d_node, plan):
            if dst_controller_address:
              dst_facade = RaidenControllerClientFacade(
                  dst_controller_address,
                  name_resolver=self.worker_rpc_client._name_resolver,
              )
              loop = asyncio.get_running_loop()
              success = await loop.run_in_executor(
                  None,
                  dst_facade.register_transfer_schedule,
                  [s_node],
                  [d_node],
                  req_id,
                  True,
                  False,
                  expected_block_count,
                  uuid,
                  dst_controller_address,
                  src_controller_address,
                  plan.shard_push_schedules,
                  dst_mem_type,
              )
              if not success:
                raise RuntimeError(
                    "Failed remote prepare in slice tree broadcast"
                )
            else:
              await self.worker_rpc_client.start_transfer(d_node, plan)

            await self.worker_rpc_client.start_transfer(s_node, plan)

          task = asyncio.create_task(
              _run_single_transfer(s, dst_unit, sub_plan)
          )
          transfers_in_progress[t] = (
              s,
              dst_unit,
              dst_block_id,
              dst_block_offset,
              dst_stride,
              task,
          )

      # 2. Wait step
      if transfers_in_progress:
        futures_to_targets = {
            info[5]: t for t, info in transfers_in_progress.items()
        }
        done, _ = await asyncio.wait(
            futures_to_targets.keys(), return_when=asyncio.FIRST_COMPLETED
        )

        # 3. Promotion step
        for fut in done:
          t = futures_to_targets[fut]
          s, dst_unit, dst_block_id, dst_block_offset, dst_stride, _ = (
              transfers_in_progress.pop(t)
          )
          active_pushes[s] -= 1
          available_sources.append(dst_unit)
          node_slice_offsets[dst_unit] = (
              dst_block_id,
              dst_block_offset,
              dst_stride,
          )
      elif not scheduled_any:
        break

  def _start_fa_reshard_transfer(
      self,
      *,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str],
      src_block_ids: Optional[list[int]],
      dst_device_block_ids: Optional[list[int]],
      dst_mem_type: RaidenMemoryType,
      use_block_chunks: bool,
      src_controller_address: Optional[str],
      dst_controller_address: Optional[str],
      uuid: Optional[int],
      is_sender: bool,
      num_tokens: Optional[int],
      parallelism: Optional[int],
  ) -> RaidenFuture:
    """Coordinates the controller-owned Stage-3 FA reshard path."""
    if not src_units:
      raise ValueError("src_units must not be empty")
    if not dst_units:
      raise ValueError("dst_units must not be empty")
    if not req_id:
      raise ValueError("req_id must not be empty for FA resharding")
    if src_block_ids is not None:
      raise ValueError(
          "Stage-3 source block IDs must come from register_request_blocks"
      )
    if dst_device_block_ids is None:
      raise ValueError("dst_device_block_ids are required for FA resharding")
    if num_tokens is None or num_tokens <= 0:
      raise ValueError("num_tokens must be positive for FA resharding")
    if not use_block_chunks:
      raise ValueError("Stage-3 FA resharding requires use_block_chunks=true")
    if dst_mem_type != RaidenMemoryType.HBM:
      raise ValueError("Stage-3 FA resharding requires dst_mem_type=HBM")
    if uuid is None:
      uuid = random.randint(1, 2**63 - 1)
    elif uuid <= 0:
      raise ValueError("uuid must be positive for FA resharding")
    if parallelism is not None and parallelism <= 0:
      raise ValueError("parallelism must be positive")

    with self._lock:
      session_id = len(self._active_transfers)

    async def _execute_fa_reshard() -> None:
      if not is_sender:
        if not src_controller_address:
          raise ValueError(
              "src_controller_address is required for destination-coordinated "
              "FA resharding"
          )
        if not dst_controller_address:
          raise ValueError(
              "dst_controller_address is required so the source controller "
              "can arm the receiver"
          )
        src_facade = RaidenControllerClientFacade(
            src_controller_address,
            name_resolver=self.worker_rpc_client._name_resolver,
        )
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            None,
            lambda: src_facade.coordinate_transfer(
                src_units=src_units,
                dst_units=dst_units,
                req_id=req_id,
                use_block_chunks=True,
                is_sender=True,
                uuid=uuid,
                dst_controller_address=dst_controller_address,
                src_controller_address=src_controller_address,
                dst_mem_type=RaidenMemoryType.HBM,
                dst_device_block_ids=list(dst_device_block_ids),
                num_tokens=num_tokens,
                parallelism=parallelism,
            ),
        )
        if not success:
          raise RuntimeError("Source controller rejected FA reshard request")
        return

      controller_start_ns = time.monotonic_ns()
      plan_build_start_ns = controller_start_ns
      src_metadata = self._get_local_metadata(src_units)
      if dst_controller_address:
        dst_metadata = await self._query_remote_metadata(dst_controller_address)
      else:
        dst_metadata = self._get_local_metadata(dst_units)
      final_plan = self._build_fa_reshard_plan(
          src_units=src_units,
          dst_units=dst_units,
          src_metadata=src_metadata,
          dst_metadata=dst_metadata,
          req_id=req_id,
          uuid=uuid,
          dst_device_block_ids=dst_device_block_ids,
          num_tokens=num_tokens,
          parallelism=parallelism,
      )
      plan_build_end_ns = time.monotonic_ns()

      # Receivers must be armed before any sender can put bytes on the wire.
      receiver_arm_start_ns = time.monotonic_ns()
      if dst_controller_address:
        dst_facade = RaidenControllerClientFacade(
            dst_controller_address,
            name_resolver=self.worker_rpc_client._name_resolver,
        )
        loop = asyncio.get_running_loop()
        try:
          success = await loop.run_in_executor(
              None,
              lambda: dst_facade.register_transfer_schedule(
                  src_units=final_plan.src_units,
                  dst_units=final_plan.dst_units,
                  req_id=final_plan.req_id,
                  use_block_chunks=True,
                  is_sender=False,
                  expected_block_count=final_plan.expected_block_count,
                  uuid=final_plan.uuid,
                  shard_push_schedules=final_plan.shard_push_schedules,
                  dst_mem_type=final_plan.dst_mem_type,
                  expected_pushes_per_pool=(
                      final_plan.expected_pushes_per_pool
                  ),
                  transfer_pool_indices=final_plan.transfer_pool_indices,
                  pool_dtype_tags=final_plan.pool_dtype_tags,
                  dst_device_block_ids=final_plan.dst_device_block_ids,
                  src_schedule_keys=final_plan.src_schedule_keys,
                  parallelism=final_plan.parallelism,
                  num_tokens=final_plan.num_tokens,
              ),
          )
        except Exception:
          self._abandon_request_blocks_claim(
              req_id, uuid, final_plan.request_block_claim_owner
          )
          raise
        if not success:
          self._abandon_request_blocks_claim(
              req_id, uuid, final_plan.request_block_claim_owner
          )
          raise RuntimeError("Destination controller failed to arm receivers")
      else:
        try:
          await asyncio.gather(
              *[
                  self.worker_rpc_client.start_transfer(unit, final_plan)
                  for unit in final_plan.dst_units
              ]
          )
        except Exception:
          self._abandon_request_blocks_claim(
              req_id, uuid, final_plan.request_block_claim_owner
          )
          raise
      receiver_arm_ack_ns = time.monotonic_ns()

      with self._lock:
        self._active_transfers[req_id] = final_plan
      sender_dispatch_ns = time.monotonic_ns()
      await asyncio.gather(
          *[
              self.worker_rpc_client.start_transfer(unit, final_plan)
              for unit in final_plan.src_units
          ]
      )
      sender_dispatch_end_ns = time.monotonic_ns()
      logging.info(
          "%s",
          json.dumps(
              {
                  "event": "raiden_stage3_senders_dispatched",
                  "req_id": final_plan.req_id,
                  "uuid": final_plan.uuid,
                  "num_tokens": final_plan.num_tokens,
                  "receiver_armed_before_sender_dispatch": True,
                  "receiver_arm_ack_monotonic_ns": receiver_arm_ack_ns,
                  "sender_dispatch_monotonic_ns": sender_dispatch_ns,
                  "plan_build_ms": (plan_build_end_ns - plan_build_start_ns)
                  / 1_000_000,
                  "receiver_arm_rpc_ms": (
                      receiver_arm_ack_ns - receiver_arm_start_ns
                  )
                  / 1_000_000,
                  "sender_dispatch_ms": (
                      sender_dispatch_end_ns - sender_dispatch_ns
                  )
                  / 1_000_000,
                  "controller_total_ms": (
                      sender_dispatch_end_ns - controller_start_ns
                  )
                  / 1_000_000,
                  "active_source_ranks": len(final_plan.src_units),
                  "destination_pages": final_plan.expected_block_count,
                  "expected_pushes_per_pool": (
                      final_plan.expected_pushes_per_pool
                  ),
                  "transferred_fa_pools": len(final_plan.transfer_pool_indices),
                  "skipped_gdn_conv_pools": (
                      final_plan.skipped_pool_counts.get(_GDN_CONV_TAG, 0)
                  ),
                  "skipped_gdn_ssm_pools": (
                      final_plan.skipped_pool_counts.get(_GDN_SSM_TAG, 0)
                  ),
              },
              sort_keys=True,
          ),
      )

    return RaidenFuture(
        session_id=session_id, transfer_task=_execute_fa_reshard()
    )

  def register_fa_receiver_plan(
      self,
      *,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: str,
      uuid: int,
      shard_push_schedules: dict,
      expected_pushes_per_pool: int,
      transfer_pool_indices: typing.Sequence[int],
      pool_dtype_tags: typing.Sequence[str],
      dst_device_block_ids: typing.Sequence[int],
      src_schedule_keys: dict[RaidenId, int],
      parallelism: int,
      num_tokens: int,
  ) -> RaidenFuture:
    """Validates and arms a source-controller-generated receiver plan."""
    if not req_id:
      raise ValueError("req_id must not be empty for FA resharding")
    if uuid <= 0:
      raise ValueError("uuid must be positive for FA resharding")
    if len(dst_units) != 1:
      raise ValueError("FA receiver plan requires exactly one destination")
    if not src_units or not shard_push_schedules:
      raise ValueError("FA receiver plan must contain source schedules")
    if expected_pushes_per_pool <= 0:
      raise ValueError("expected_pushes_per_pool must be positive")
    if parallelism <= 0:
      raise ValueError("parallelism must be positive")
    if num_tokens <= 0:
      raise ValueError("num_tokens must be positive")
    dst_ids = [int(block_id) for block_id in dst_device_block_ids]
    if not dst_ids or any(block_id < 0 for block_id in dst_ids):
      raise ValueError(
          "destination block IDs must be non-empty and non-negative"
      )
    if len(set(dst_ids)) != len(dst_ids):
      raise ValueError("destination block IDs must be unique")
    if set(src_schedule_keys) != set(src_units):
      raise ValueError("Every source must have an explicit schedule key")
    if set(src_schedule_keys.values()) != set(range(len(src_units))):
      raise ValueError(
          "Source schedule keys must be contiguous source ordinals"
      )
    if set(shard_push_schedules) != set(src_units):
      raise ValueError("Every source must have one decoded push schedule")

    dst_metadata = self._get_local_metadata(dst_units)
    dst_meta = self._metadata_by_unit(dst_metadata, dst_units)[dst_units[0]]
    if not dst_meta.pools:
      raise ValueError("FA receiver requires an explicit pool manifest")
    expected_dtype_tags = [str(pool.dtype_tag) for pool in dst_meta.pools]
    if list(pool_dtype_tags) != expected_dtype_tags:
      raise ValueError("Receiver pool dtype tags do not match registration")
    expected_fa = [
        index
        for index, pool in enumerate(dst_meta.pools)
        if pool.tag == _FA_TAG
    ]
    if list(transfer_pool_indices) != expected_fa:
      raise ValueError("Receiver plan must name every and only FA pool")
    page_tokens = int(dst_meta.page_tokens)
    page_slice_tokens = int(dst_meta.page_slice_tokens)
    if page_tokens <= 0 or page_slice_tokens <= 0:
      raise ValueError(
          "FA receiver requires explicit positive destination page geometry"
      )
    if page_slice_tokens != page_tokens:
      raise ValueError(
          "FA receiver requires contiguous TP1 destination page geometry"
      )
    expected_pages = math.ceil(num_tokens / page_tokens)
    if len(dst_ids) != expected_pages:
      raise ValueError(
          "Destination block-id count does not match registered page geometry"
      )
    destination_page_bytes = set()
    for pool_idx in expected_fa:
      pool = dst_meta.pools[pool_idx]
      live_bytes = _pool_live_bytes(pool)
      if live_bytes != int(pool.block_stride_bytes):
        raise ValueError(
            "FA receiver pool is not full-live; decode TP>1 is unsupported. "
            f"See {_TP_GATHER_DESIGN}"
        )
      if live_bytes <= 0 or live_bytes % page_tokens:
        raise ValueError(
            f"FA receiver pool {pool_idx} has invalid token byte geometry"
        )
      destination_page_bytes.add(live_bytes)
      if any(block_id >= int(pool.num_blocks) for block_id in dst_ids):
        raise ValueError(
            f"Destination block id is out of range for FA pool {pool_idx}"
        )
    if len(destination_page_bytes) != 1:
      raise ValueError("All receiver FA pools must have one byte geometry")
    page_bytes = next(iter(destination_page_bytes))
    token_bytes = page_bytes // page_tokens

    dst_peers = set(dst_meta.shards)
    entry_count = 0
    calculated_pushes = 0
    coverage_by_dst_id = {block_id: [] for block_id in dst_ids}
    dst_page_ordinals = {
        block_id: ordinal for ordinal, block_id in enumerate(dst_ids)
    }
    for src_unit, unit_schedules in shard_push_schedules.items():
      if set(unit_schedules) != {0}:
        raise ValueError(
            f"FA sender {src_unit} must use exactly local schedule key 0"
        )
      entries = unit_schedules[0]
      transfer_pairs = set()
      for entry in entries:
        entry_count += 1
        if entry[0] not in dst_peers:
          raise ValueError(
              f"FA schedule targets an unregistered destination peer {entry[0]}"
          )
        dst_shard_idx = int(entry[1])
        dst_offset_bytes = int(entry[2])
        src_offset_bytes = int(entry[3])
        size_bytes = int(entry[4])
        src_block_id = int(entry[5])
        dst_block_id = int(entry[6])
        if dst_block_id not in coverage_by_dst_id:
          raise ValueError(
              "FA receiver entry targets an unregistered destination block"
          )
        if (
            dst_shard_idx != 0
            or dst_offset_bytes < 0
            or src_offset_bytes < 0
            or size_bytes <= 0
            or src_block_id < 0
            or entry[7] != 0
            or entry[8] != 0
            or entry[9] != 1
        ):
          raise ValueError(
              "FA receiver entries must be positive contiguous count=1 "
              "writes with non-negative offsets and block IDs"
          )
        if (
            dst_offset_bytes % token_bytes
            or src_offset_bytes % token_bytes
            or size_bytes % token_bytes
        ):
          raise ValueError("FA receiver entries must be token-byte aligned")
        dst_page_ordinal = dst_page_ordinals[dst_block_id]
        page_token_start = dst_page_ordinal * page_tokens
        expected_tokens = min(page_tokens, num_tokens - page_token_start)
        expected_bytes = expected_tokens * token_bytes
        dst_end_bytes = dst_offset_bytes + size_bytes
        if dst_end_bytes > expected_bytes:
          raise ValueError(
              "FA receiver entry exceeds its destination page or live tail"
          )
        coverage_by_dst_id[dst_block_id].append(
            (dst_offset_bytes, dst_end_bytes)
        )
        transfer_pairs.add((entry[0], src_block_id, dst_block_id))
      calculated_pushes += min(parallelism, len(transfer_pairs))
    if entry_count == 0:
      raise ValueError("FA receiver plan contains no entries")
    if calculated_pushes != expected_pushes_per_pool:
      raise ValueError(
          "expected_pushes_per_pool does not match the received schedules"
      )
    for dst_page_ordinal, dst_block_id in enumerate(dst_ids):
      expected_tokens = min(
          page_tokens, num_tokens - dst_page_ordinal * page_tokens
      )
      expected_bytes = expected_tokens * token_bytes
      covered_until = 0
      for start_bytes, end_bytes in sorted(coverage_by_dst_id[dst_block_id]):
        if start_bytes != covered_until:
          relation = "overlap" if start_bytes < covered_until else "gap"
          raise ValueError(
              "FA receiver schedule has a destination coverage "
              f"{relation} for block {dst_block_id}"
          )
        covered_until = end_bytes
      if covered_until != expected_bytes:
        raise ValueError(
            "FA receiver schedule does not cover the exact live bytes for "
            f"destination block {dst_block_id}"
        )

    rpc_addresses = self.worker_rpc_client.get_worker_endpoints()
    for unit in dst_units:
      if not rpc_addresses.get(unit):
        raise ValueError(f"Missing control-plane endpoint for {unit}")
    plan = TransferPlan(
        src_units=list(src_units),
        dst_units=list(dst_units),
        plan={},
        shard_push_schedules=shard_push_schedules,
        worker_rpc_addresses=dict(rpc_addresses),
        worker_data_addresses={dst_units[0]: list(dst_meta.shards)},
        uuid=uuid,
        dst_mem_type=RaidenMemoryType.HBM,
        use_block_chunks=True,
        is_sender=False,
        expected_block_count=len(dst_ids),
        req_id=req_id,
        expected_pushes_per_pool=expected_pushes_per_pool,
        transfer_pool_indices=list(transfer_pool_indices),
        pool_dtype_tags=list(pool_dtype_tags),
        dst_device_block_ids=dst_ids,
        src_schedule_keys=dict(src_schedule_keys),
        parallelism=parallelism,
        num_tokens=num_tokens,
    )
    with self._lock:
      session_id = len(self._active_transfers)

    async def _arm_receivers() -> None:
      receiver_arm_start_ns = time.monotonic_ns()
      await asyncio.gather(
          *[
              self.worker_rpc_client.start_transfer(unit, plan)
              for unit in dst_units
          ]
      )
      receiver_arm_end_ns = time.monotonic_ns()
      with self._lock:
        self._active_transfers[req_id] = plan
      page_tokens = int(dst_meta.page_tokens)
      logging.info(
          "%s",
          json.dumps(
              {
                  "event": "raiden_stage3_receivers_armed",
                  "req_id": req_id,
                  "uuid": uuid,
                  "num_tokens": num_tokens,
                  "page_tokens": page_tokens,
                  "destination_pages": len(dst_ids),
                  "destination_tail_tokens": (
                      num_tokens % page_tokens or page_tokens
                  ),
                  "receiver_arm_rpc_ms": (
                      receiver_arm_end_ns - receiver_arm_start_ns
                  )
                  / 1_000_000,
              },
              sort_keys=True,
          ),
      )

    return RaidenFuture(session_id, _arm_receivers())

  def start_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      src_block_ids: Optional[list[int]] = None,
      dst_device_block_ids: Optional[list[int]] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      use_block_chunks: bool = False,
      src_controller_address: Optional[str] = None,
      dst_controller_address: Optional[str] = None,
      uuid: Optional[int] = None,
      is_sender: bool = True,
      expected_block_count: int = 0,
      shard_push_schedules: Optional[dict] = None,
      num_tokens: Optional[int] = None,
      parallelism: Optional[int] = None,
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
      use_block_chunks: Whether to use chunked transport.
      src_controller_address: Optional address of the source controller.
      dst_controller_address: Optional address of the destination controller.
      uuid: Optional pre-determined UUID for the transfer.
      is_sender: If True, this controller acts as the Sender Coordinator,
        querying destination metadata and triggering the active push on source
        workers. If False, this controller acts as the Destination Coordinator,
        preparing local receiver workers and setting up their expected block
        count.
      expected_block_count: The total number of physical block-pushes expected
        per destination rank (only applicable when is_sender=False).

    Returns:
      A Future for the call site to wait for the transfer to complete.
    """

    if num_tokens is not None or dst_device_block_ids is not None:
      if shard_push_schedules:
        raise ValueError(
            "Prepared FA schedules must use register_fa_receiver_plan"
        )
      return self._start_fa_reshard_transfer(
          src_units=src_units,
          dst_units=dst_units,
          req_id=req_id,
          src_block_ids=src_block_ids,
          dst_device_block_ids=dst_device_block_ids,
          dst_mem_type=dst_mem_type,
          use_block_chunks=use_block_chunks,
          src_controller_address=src_controller_address,
          dst_controller_address=dst_controller_address,
          uuid=uuid,
          is_sender=is_sender,
          num_tokens=num_tokens,
          parallelism=parallelism,
      )

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

    if uuid is None:
      uuid = random.randint(1, 2**63 - 1)

    # Determine session_id and req_id synchronously
    with self._lock:
      session_id = len(self._active_transfers)
      if not req_id:
        req_id = f"req_{session_id}"

    # Generate default plan (always needed as fallback or for old workflow)
    num_src = len(self._resolve_shards(selected_src))
    default_plan_dict = {}
    src_plan = [[] for _ in range(num_src)]
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
            nd_slice = [(local_start, local_end)]
            src_plan[i].append((dst_unit, j, [nd_slice]))
    default_plan_dict[selected_src] = src_plan

    if not use_block_chunks:
      # === OLD WORKFLOW: Fully build and store plan SYNCHRONOUSLY ===
      rpc_addresses = {}
      if hasattr(self.worker_rpc_client, "get_worker_endpoints"):
        rpc_addresses = self.worker_rpc_client.get_worker_endpoints()
      data_addresses = {unit: self._resolve_shards(unit) for unit in dst_units}

      plan = TransferPlan(
          src_units=[selected_src],
          dst_units=dst_units,
          plan=default_plan_dict,
          shard_push_schedules={},
          worker_rpc_addresses=rpc_addresses,
          worker_data_addresses=data_addresses,
          uuid=uuid,
          dst_mem_type=dst_mem_type,
          use_block_chunks=False,
      )
      with self._lock:
        self._active_transfers[req_id] = plan
    else:
      # === NEW WORKFLOW: Store partial plan SYNCHRONOUSLY ===
      plan = TransferPlan(
          src_units=src_units,
          dst_units=dst_units,
          plan=None,
          shard_push_schedules=shard_push_schedules or {},
          worker_rpc_addresses=dict(
              self.worker_rpc_client.get_worker_endpoints()
          ),
          worker_data_addresses=dict(self._registered_shards),
          uuid=uuid,
          dst_mem_type=dst_mem_type,
          use_block_chunks=True,
          is_sender=is_sender,
          expected_block_count=expected_block_count,
          req_id=req_id,
      )
      with self._lock:
        self._active_transfers[req_id] = plan

    async def _execute_transfer() -> None:
      if use_block_chunks:
        # === NEW SYMMETRIC DECENTRALIZED WORKFLOW ===

        if not is_sender:
          # --- ROLE: DESTINATION CONTROLLER (RECEIVER COORDINATOR) ---
          logging.info(
              "RaidenController acting as DESTINATION COORDINATOR"
              " (is_sender=False) for uuid %s",
              uuid,
          )

          # 1. Discover local destination units
          local_dst_units = [
              u for u in dst_units if u in self._registered_shards
          ]
          if not local_dst_units:
            logging.warning("No local destination units found to prepare!")
            return

          # 2. Build rpc_addresses for local destination workers
          rpc_addresses = self.worker_rpc_client.get_worker_endpoints()

          # 3. Construct a lightweight TransferPlan containing receiver parameters
          receiver_plan = TransferPlan(
              src_units=src_units,
              dst_units=dst_units,
              plan=None,
              shard_push_schedules=shard_push_schedules or {},
              worker_rpc_addresses=rpc_addresses,
              worker_data_addresses={
                  u: self._registered_shards[u] for u in local_dst_units
              },
              uuid=uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=True,
              is_sender=False,
              expected_block_count=expected_block_count,
              req_id=req_id,
          )

          # 4. Trigger COMMAND_START_TRANSFER (is_sender=False) on local workers
          logging.info(
              "Triggering preparation RPCs on local destination workers: %s,"
              " expected blocks: %d",
              local_dst_units,
              expected_block_count,
          )
          await asyncio.gather(
              *[
                  self.worker_rpc_client.start_transfer(unit, receiver_plan)
                  for unit in local_dst_units
              ]
          )
          logging.info(
              "Symmetric preparation complete on all local destination workers."
          )

        else:
          # --- ROLE: SENDER CONTROLLER (SENDER COORDINATOR) ---
          logging.info(
              "RaidenController acting as SENDER COORDINATOR (is_sender=True)"
              " for uuid %s",
              uuid,
          )

          # 1. Retrieve destination metadata (either from remote dst_controller or local)
          dst_metadata = []
          if dst_controller_address:
            logging.info(
                "Querying remote destination controller %s for metadata",
                dst_controller_address,
            )
            dst_metadata = await self._query_remote_metadata(
                dst_controller_address
            )
          else:
            logging.info("Using local registration for destination metadata")
            dst_metadata = self._get_local_metadata(dst_units)

          # 2. Compute slices or use pre-computed schedules centrally
          computed_schedules = {}
          data_address_to_unit = {}
          if shard_push_schedules:
            logging.info("Using pre-computed shard_push_schedules")
            computed_schedules = shard_push_schedules
            for meta in dst_metadata:
              unit = _raiden_id_from_proto(meta.unit)
              for shard in meta.shards:
                data_address_to_unit[shard] = unit
          else:
            computed_slices = {}

            # Source slices (always local to sender controller)
            for unit in src_units:
              with self._lock:
                global_shape = self._registered_global_shapes.get(unit)
                mesh_shape = self._registered_mesh_shapes.get(unit)
                layout = self._registered_layouts.get(unit)
              if global_shape and mesh_shape and layout:
                phys_shape, phys_mesh = to_physical(
                    global_shape, mesh_shape, layout
                )
                slices = nd_slice_math.compute_nd_shard_slices(
                    phys_shape, phys_mesh
                )
                computed_slices[unit] = slices
                logging.info("Computed source slices for %s: %s", unit, slices)

            # Destination slices
            for meta in dst_metadata:
              unit = _raiden_id_from_proto(meta.unit)
              for shard in meta.shards:
                data_address_to_unit[shard] = unit
              if meta.global_shape and meta.mesh_shape and meta.layout:
                phys_shape, phys_mesh = to_physical(
                    list(meta.global_shape),
                    list(meta.mesh_shape),
                    list(meta.layout),
                )
                slices = nd_slice_math.compute_nd_shard_slices(
                    phys_shape, phys_mesh
                )
                computed_slices[unit] = slices
                logging.info(
                    "Computed destination slices for %s: %s", unit, slices
                )

            # 3. Generate plan (Intersection)
            for src_unit in src_units:
              src_slices = computed_slices.get(src_unit)
              if not src_slices:
                continue

              # Get itemsize
              with self._lock:
                itemsize = self._registered_itemsizes.get(src_unit)
              if not itemsize:
                itemsize = 4  # default fallback

              src_shards = self._resolve_shards(src_unit)
              unit_schedules = {}

              if len(src_shards) == 1:
                try:
                  src_indices = [(0, int(src_unit.job_replica_id))]
                except ValueError:
                  src_indices = [(0, 0)]
              else:
                src_indices = [(i, i) for i in range(len(src_slices))]

              for local_src_idx, global_src_idx in src_indices:
                if global_src_idx >= len(src_slices):
                  logging.warning(
                      "global_src_idx %d out of range of src_slices (%d)",
                      global_src_idx,
                      len(src_slices),
                  )
                  continue

                src_slice_proto = src_slices[global_src_idx]
                src_slice = _proto_to_nd_slice(src_slice_proto)
                shard_entries = []

                for dst_unit in dst_units:
                  d_slices = computed_slices.get(dst_unit)
                  if not d_slices:
                    continue

                  dst_shards = []
                  for meta in dst_metadata:
                    meta_unit = _raiden_id_from_proto(meta.unit)
                    if meta_unit == dst_unit:
                      dst_shards = list(meta.shards)
                      break
                  if not dst_shards:
                    dst_shards = ["127.0.0.1:8000"]  # fallback

                  if len(dst_shards) == 1:
                    try:
                      dst_indices = [(0, int(dst_unit.job_replica_id))]
                    except ValueError:
                      dst_indices = [(0, 0)]
                  else:
                    dst_indices = [(j, j) for j in range(len(d_slices))]

                  for local_dst_idx, global_dst_idx in dst_indices:
                    if global_dst_idx >= len(d_slices):
                      logging.warning(
                          "global_dst_idx %d out of range of d_slices (%d)",
                          global_dst_idx,
                          len(d_slices),
                      )
                      continue

                    dst_slice_proto = d_slices[global_dst_idx]
                    dst_slice = _proto_to_nd_slice(dst_slice_proto)

                    dst_peer = (
                        dst_shards[local_dst_idx]
                        if local_dst_idx < len(dst_shards)
                        else dst_shards[0]
                    )

                    intersection = intersect_nd_slices(src_slice, dst_slice)
                    if intersection:
                      chunks = generate_strided_copy_chunks(
                          src_slice, dst_slice, intersection, itemsize
                      )
                      for (
                          src_offset,
                          dst_offset,
                          size,
                          src_stride,
                          dst_stride,
                          count,
                      ) in chunks:
                        src_block_bytes = (
                            math.prod([e - s for s, e in src_slice[1:]])
                            * itemsize
                            if len(src_slice) > 1
                            else itemsize
                        )
                        dst_block_bytes = (
                            math.prod([e - s for s, e in dst_slice[1:]])
                            * itemsize
                            if len(dst_slice) > 1
                            else itemsize
                        )
                        src_block_id = src_offset // src_block_bytes
                        dst_block_id = dst_offset // dst_block_bytes

                        # Make offsets block-relative
                        src_block_offset = src_offset % src_block_bytes
                        dst_block_offset = dst_offset % dst_block_bytes

                        shard_entries.append(
                            (
                                dst_peer,
                                local_dst_idx,
                                dst_block_offset,
                                src_block_offset,
                                size,
                                src_block_id,
                                dst_block_id,
                                src_stride,
                                dst_stride,
                                count,
                            )
                        )

                if shard_entries:
                  unit_schedules[local_src_idx] = shard_entries

              if unit_schedules:
                computed_schedules[src_unit] = unit_schedules

          # Build rpc_addresses for local source workers
          rpc_addresses = self.worker_rpc_client.get_worker_endpoints()
          # Merge destination rpc addresses from metadata
          for meta in dst_metadata:
            unit = _raiden_id_from_proto(meta.unit)
            if meta.control_plane_rpc_address:
              rpc_addresses[unit] = meta.control_plane_rpc_address

          data_addresses = {unit: [] for unit in dst_units}
          for meta in dst_metadata:
            unit = _raiden_id_from_proto(meta.unit)
            if unit in data_addresses:
              data_addresses[unit] = list(meta.shards)

          # Build final plan and replace the partial plan
          final_plan = TransferPlan(
              src_units=list(computed_schedules.keys())
              if computed_schedules
              else src_units,
              dst_units=dst_units,
              plan=None,
              shard_push_schedules=computed_schedules,
              worker_rpc_addresses=rpc_addresses,
              worker_data_addresses=data_addresses,
              uuid=uuid,
              dst_mem_type=dst_mem_type,
              use_block_chunks=use_block_chunks,
              is_sender=True,
              expected_block_count=expected_block_count,
              req_id=req_id,
          )
          with self._lock:
            self._active_transfers[req_id] = final_plan

            # Group flat entries into slices for broadcast
            groups = {}
            for src_unit, schedules in computed_schedules.items():
              for shard_idx, entries in schedules.items():
                for entry in entries:
                  (
                      dst_peer,
                      dst_shard_idx,
                      dst_block_offset,
                      src_block_offset,
                      size,
                      src_block_id,
                      dst_block_id,
                      src_stride,
                      dst_stride,
                      count,
                  ) = entry
                  dst_unit = data_address_to_unit.get(dst_peer)
                  if not dst_unit:
                    continue
                  key = (
                      src_unit,
                      shard_idx,
                      src_block_id,
                      src_block_offset,
                      size,
                      src_stride,
                      count,
                  )
                  val = (
                      dst_unit,
                      dst_peer,
                      dst_shard_idx,
                      dst_block_id,
                      dst_block_offset,
                      dst_stride,
                  )
                  groups.setdefault(key, []).append(val)

            # Partition slices into direct transfers and tree-broadcast
            # transfers.
            direct_schedules = {}
            tree_broadcast_tasks = []

            for key, targets in groups.items():
              (
                  src_unit,
                  shard_idx,
                  src_block_id,
                  src_block_offset,
                  size,
                  src_stride,
                  count,
              ) = key
              if len(targets) <= 1:
                # Re-assemble entry for flat schedule
                for (
                    dst_unit,
                    dst_peer,
                    dst_shard_idx,
                    dst_block_id,
                    dst_block_offset,
                    dst_stride,
                ) in targets:
                  entry = (
                      dst_peer,
                      dst_shard_idx,
                      dst_block_offset,
                      src_block_offset,
                      size,
                      src_block_id,
                      dst_block_id,
                      src_stride,
                      dst_stride,
                      count,
                  )
                  direct_schedules.setdefault(src_unit, {}).setdefault(
                      shard_idx, []
                  ).append(entry)
              else:
                # TODO(b/12345678): Optimize early stages with topology-aware
                # branching based on IP/subnet closeness to minimize
                # inter-switch DCN traffic.
                task = self._execute_slice_broadcast(
                    key=key,
                    targets=targets,
                    final_plan=final_plan,
                    fanout_k=self.broadcast_k,
                    req_id=req_id,
                    uuid=uuid,
                    dst_mem_type=dst_mem_type,
                    expected_block_count=expected_block_count,
                    dst_controller_address=dst_controller_address,
                    src_controller_address=src_controller_address,
                )
                tree_broadcast_tasks.append(task)

            # Execute direct schedules (traditional flat route) and tree
            # broadcasts in parallel!
            if direct_schedules:
              direct_plan = TransferPlan(
                  src_units=list(direct_schedules.keys()),
                  dst_units=dst_units,
                  plan=None,
                  shard_push_schedules=direct_schedules,
                  worker_rpc_addresses=dict(final_plan.worker_rpc_addresses),
                  worker_data_addresses=dict(final_plan.worker_data_addresses),
                  uuid=uuid,
                  dst_mem_type=dst_mem_type,
                  use_block_chunks=True,
                  is_sender=True,
                  expected_block_count=expected_block_count,
                  req_id=req_id,
              )

              direct_dsts = []
              for src, scheds in direct_schedules.items():
                for sh, entries in scheds.items():
                  for entry in entries:
                    dst_peer = entry[0]
                    d_node = data_address_to_unit.get(dst_peer)
                    if d_node and d_node not in direct_dsts:
                      direct_dsts.append(d_node)

              if dst_controller_address:
                dst_facade = RaidenControllerClientFacade(
                    dst_controller_address,
                    name_resolver=self.worker_rpc_client._name_resolver,
                )
                loop = asyncio.get_running_loop()
                success = await loop.run_in_executor(
                    None,
                    dst_facade.register_transfer_schedule,
                    list(direct_schedules.keys()),
                    direct_dsts,
                    req_id,
                    True,
                    False,
                    expected_block_count,
                    uuid,
                    dst_controller_address,
                    src_controller_address,
                    direct_schedules,
                    dst_mem_type,
                )
                if not success:
                  raise RuntimeError(
                      "Failed remote prepare for direct schedules"
                  )
              else:
                local_direct_dsts = [
                    u for u in direct_dsts if u in self._registered_shards
                ]
                if local_direct_dsts:
                  await asyncio.gather(
                      *[
                          self.worker_rpc_client.start_transfer(
                              unit, direct_plan
                          )
                          for unit in local_direct_dsts
                      ]
                  )

              local_direct_srcs = [
                  u
                  for u in direct_schedules.keys()
                  if u in self._registered_shards
              ]
              if local_direct_srcs:
                await asyncio.gather(
                    *[
                        self.worker_rpc_client.start_transfer(unit, direct_plan)
                        for unit in local_direct_srcs
                    ]
                )

            if tree_broadcast_tasks:
              await asyncio.gather(*tree_broadcast_tasks)

      else:
        # === OLD PLAN-BASED WORKFLOW (Backward Compatibility) ===
        # Retrieve the plan that was stored synchronously
        with self._lock:
          old_plan = self._active_transfers[req_id]

        # 1. Send start_transfer to Destination workers first to register the
        # plan.
        # This is REQUIRED in the old workflow because they need the plan to
        # unpack!
        for unit in dst_units:
          await self.worker_rpc_client.start_transfer(unit, old_plan)

        # 2. Send start_transfer to Source workers to trigger the actual push.
        await asyncio.gather(
            *[
                self.worker_rpc_client.start_transfer(unit, old_plan)
                for unit in old_plan.src_units
            ]
        )

    transfer_task = _execute_transfer()
    return RaidenFuture(session_id=session_id, transfer_task=transfer_task)


class RaidenControllerServer:
  """Centralized Control-Plane network servicer hosting a highly secure JSON/Pickle TCP Controller server."""

  def __init__(
      self,
      controller: "RaidenController",
      proto_module: Optional[Any] = None,
      raiden_proto_module: Optional[Any] = None,
  ):
    """Instantiates RaidenControllerServer on an active RaidenController instance.

    Args:
      controller: High-level RaidenController instance managing transfer plans.
      proto_module: Optional protobuf module to use for
        ControllerRequest/Response. Defaults to controller_service_pb2.
      raiden_proto_module: Optional protobuf module for raiden service
        primitives.
    """
    self._controller = controller
    self._proto_module = proto_module or controller_service_pb2
    self._raiden_proto_module = raiden_proto_module or raiden_service_pb2
    self._sock = create_server_socket(controller.port)
    # Port 0 is useful for atomic ephemeral-port selection in tests and local
    # harnesses. Publish the kernel-selected port before start()/stop() use it.
    if controller.port == 0:
      controller.port = int(self._sock.getsockname()[1])
    self._stopped = False
    self._thread = None

  @property
  def port(self) -> int:
    """Returns the bound listener port, including for a requested port 0."""
    return self._controller.port

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
    for host in ("[::1]", "127.0.0.1"):
      try:
        wake_socket = connect_socket(
            f"{host}:{self._controller.port}", timeout=0.5
        )
        wake_socket.close()
        break
      except Exception:  # pylint: disable=broad-except
        pass

    try:
      self._sock.shutdown(socket.SHUT_RDWR)
    except Exception:
      pass
    try:
      self._sock.close()
    except Exception:
      pass

  def _server_loop(self) -> None:
    """Accepts connections; every handler owns its asyncio event loop."""
    while not self._stopped:
      try:
        conn, _ = self._sock.accept()
        if self._stopped:
          conn.close()
          break
        threading.Thread(
            target=self._handle_conn, args=(conn,), daemon=True
        ).start()
      except OSError:
        break

  def _handle_conn(self, conn: socket.socket) -> None:
    """Internal connection processing handler executing deserialized ControllerRequest Protobuf RPC payloads.

    Args:
      conn: Accepted incoming TCP socket client handle.
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

      req = self._proto_module.ControllerRequest()
      try:
        req.ParseFromString(req_bytes)
      except Exception:
        req.command = self._proto_module.ControllerRequest.COMMAND_UNSPECIFIED

      if (
          req.command
          == self._proto_module.ControllerRequest.COMMAND_COORDINATE_TRANSFER
          and req.HasField("coordinate_transfer_request")
      ):
        resp = self._proto_module.ControllerResponse()
        resp.success = False
        try:
          if (
              req.command
              == self._proto_module.ControllerRequest.COMMAND_COORDINATE_TRANSFER
          ):
            coord_req = req.coordinate_transfer_request
            srcs = [_raiden_id_from_proto(u) for u in coord_req.src_units]
            dsts = [_raiden_id_from_proto(u) for u in coord_req.dst_units]
            dst_mem_type = RaidenMemoryType.DRAM
            if (
                coord_req.dst_mem_type
                == self._raiden_proto_module.MEMORY_TYPE_HBM
            ):
              dst_mem_type = RaidenMemoryType.HBM

            future = self._controller.start_transfer(
                src_units=srcs,
                dst_units=dsts,
                req_id=coord_req.req_id if coord_req.req_id else None,
                dst_mem_type=dst_mem_type,
                use_block_chunks=coord_req.use_block_chunks,
                src_controller_address=coord_req.src_controller_address
                if coord_req.src_controller_address
                else None,
                dst_controller_address=coord_req.dst_controller_address
                if coord_req.dst_controller_address
                else None,
                uuid=coord_req.uuid if coord_req.uuid > 0 else None,
                is_sender=coord_req.is_sender,
                expected_block_count=coord_req.expected_block_count,
                shard_push_schedules=None,
                dst_device_block_ids=(
                    list(coord_req.dst_device_block_ids)
                    if coord_req.dst_device_block_ids
                    else None
                ),
                num_tokens=(
                    coord_req.num_tokens
                    if coord_req.num_tokens > 0
                    or coord_req.dst_device_block_ids
                    else None
                ),
                parallelism=(
                    coord_req.parallelism if coord_req.parallelism > 0 else None
                ),
            )
            asyncio.run(future.wait())
            resp.success = True
        except Exception as e:
          resp.message = str(e)
        resp_bytes = resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
      else:
        raiden_req = self._raiden_proto_module.ControlRequest()
        raiden_req.ParseFromString(req_bytes)
        raiden_resp = self._raiden_proto_module.ControlResponse()
        raiden_resp.success = False
        try:
          if (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_WORK_UNIT
          ):
            reg = raiden_req.register_work_unit_request
            unit = _raiden_id_from_proto(reg.unit)
            shards = list(reg.shards)
            ctrl_addr = (
                reg.control_plane_rpc_address
                if reg.control_plane_rpc_address
                else None
            )
            mesh_shape = list(reg.mesh_shape) if reg.mesh_shape else None
            layout = list(reg.layout) if reg.layout else None
            global_shape = list(reg.global_shape) if reg.global_shape else None
            itemsize = reg.itemsize if reg.itemsize > 0 else None
            pool_manifest = list(reg.pools) if reg.pools else None
            layout_fingerprint = (
                reg.layout_fingerprint if reg.layout_fingerprint else None
            )
            page_tokens = reg.page_tokens if reg.page_tokens > 0 else None
            page_slice_tokens = (
                reg.page_slice_tokens if reg.page_slice_tokens > 0 else None
            )
            transfer_parallelism = (
                reg.transfer_parallelism
                if reg.transfer_parallelism > 0
                else None
            )
            transfer_rank = (
                reg.transfer_rank if pool_manifest is not None else None
            )

            self._controller.register_work_unit(
                unit,
                shards,
                control_plane_rpc_address=ctrl_addr,
                mesh_shape=mesh_shape,
                layout=layout,
                global_shape=global_shape,
                itemsize=itemsize,
                pool_manifest=pool_manifest,
                layout_fingerprint=layout_fingerprint,
                page_tokens=page_tokens,
                page_slice_tokens=page_slice_tokens,
                transfer_parallelism=transfer_parallelism,
                transfer_rank=transfer_rank,
            )
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_REQUEST_BLOCKS
          ):
            block_req = raiden_req.register_request_blocks_request
            self._controller.register_request_blocks(
                block_req.req_id,
                block_req.uuid,
                _raiden_id_from_proto(block_req.unit),
                list(block_req.block_ids),
            )
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_RELEASE_REQUEST_BLOCKS
          ):
            release_req = raiden_req.release_request_blocks_request
            if not release_req.force:
              raise ValueError(
                  "Legacy rank-local release is unsafe; use the aggregate "
                  "completion-vote RPC"
              )
            released = self._controller.release_request_blocks(
                release_req.req_id,
                release_req.uuid,
            )
            raiden_resp.response_data = str(released)
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_COMPLETE_REQUEST_BLOCKS
          ):
            complete_req = raiden_req.complete_request_blocks_request
            released = self._controller.complete_request_blocks(
                complete_req.req_id,
                complete_req.uuid,
                _raiden_id_from_proto(complete_req.unit),
            )
            raiden_resp.response_data = str(released)
            raiden_resp.success = True
          elif raiden_req.command == (
              self._raiden_proto_module.ControlRequest.COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED
          ):
            cancel_req = raiden_req.cancel_request_blocks_if_unclaimed_request
            cancelled = self._controller.cancel_request_blocks_if_unclaimed(
                cancel_req.req_id, cancel_req.uuid
            )
            raiden_resp.response_data = "true" if cancelled else "false"
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_GET_METADATA
          ):
            metadata_protos = self._controller.get_all_metadata()
            raiden_resp.get_metadata_response.metadata.extend(metadata_protos)
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_TRANSFER_SCHEDULE
          ):
            start_req = raiden_req.start_transfer_request
            srcs = [_raiden_id_from_proto(u) for u in start_req.src_units]
            dsts = [_raiden_id_from_proto(u) for u in start_req.dst_units]
            dst_mem_type = RaidenMemoryType.DRAM
            if (
                start_req.dst_mem_type
                == self._raiden_proto_module.MEMORY_TYPE_HBM
            ):
              dst_mem_type = RaidenMemoryType.HBM

            def decode_entries(schedule_proto):
              return [
                  (
                      entry.dst_peer,
                      entry.dst_shard_idx,
                      entry.dst_offset_bytes,
                      entry.src_offset_bytes,
                      entry.size_bytes,
                      entry.src_block_id,
                      entry.dst_block_id,
                      entry.src_stride_bytes,
                      entry.dst_stride_bytes,
                      entry.count,
                  )
                  for entry in schedule_proto.entries
              ]

            is_fa_reshard = bool(
                start_req.expected_pushes_per_pool
                or start_req.transfer_pool_indices
            )
            if is_fa_reshard:
              if start_req.is_sender:
                raise ValueError(
                    "Inter-controller FA schedule registration must target "
                    "receivers"
                )
              if not start_req.use_block_chunks:
                raise ValueError(
                    "FA receiver schedule requires use_block_chunks=true"
                )
              if dst_mem_type != RaidenMemoryType.HBM:
                raise ValueError(
                    "FA receiver schedule requires dst_mem_type=HBM"
                )
              expected_keys = set(range(len(srcs)))
              actual_keys = set(start_req.shard_push_schedules)
              if actual_keys != expected_keys:
                raise ValueError(
                    "FA schedule map keys must match source list ordinals: "
                    f"got={sorted(actual_keys)}, "
                    f"expected={sorted(expected_keys)}"
                )
              shard_push_schedules = {
                  src_unit: {
                      0: decode_entries(start_req.shard_push_schedules[ordinal])
                  }
                  for ordinal, src_unit in enumerate(srcs)
              }
              future = self._controller.register_fa_receiver_plan(
                  src_units=srcs,
                  dst_units=dsts,
                  req_id=start_req.req_id,
                  uuid=start_req.uuid,
                  shard_push_schedules=shard_push_schedules,
                  expected_pushes_per_pool=(start_req.expected_pushes_per_pool),
                  transfer_pool_indices=list(start_req.transfer_pool_indices),
                  pool_dtype_tags=list(start_req.pool_dtype_tags),
                  dst_device_block_ids=list(start_req.dst_device_block_ids),
                  src_schedule_keys={
                      src_unit: ordinal for ordinal, src_unit in enumerate(srcs)
                  },
                  parallelism=start_req.parallelism,
                  num_tokens=start_req.num_tokens,
              )
            else:
              shard_push_schedules = {}
              if len(srcs) == 1 and len(start_req.shard_push_schedules) > 1:
                unit_schedules = {}
                for (
                    key_idx,
                    schedule_proto,
                ) in start_req.shard_push_schedules.items():
                  entries = decode_entries(schedule_proto)
                  if entries:
                    unit_schedules[key_idx] = entries
                if unit_schedules:
                  shard_push_schedules[srcs[0]] = unit_schedules
              else:
                # Legacy transfer IDs historically used job_replica_id as a
                # schedule key. This compatibility branch is intentionally
                # outside the Stage-3 fail-closed path.
                for src_unit in srcs:
                  src_replica_idx = int(src_unit.job_replica_id)
                  if src_replica_idx in start_req.shard_push_schedules:
                    entries = decode_entries(
                        start_req.shard_push_schedules[src_replica_idx]
                    )
                    if entries:
                      shard_push_schedules[src_unit] = {0: entries}

              future = self._controller.start_transfer(
                  src_units=srcs,
                  dst_units=dsts,
                  req_id=start_req.req_id if start_req.req_id else None,
                  dst_mem_type=dst_mem_type,
                  use_block_chunks=start_req.use_block_chunks,
                  src_controller_address=None,
                  dst_controller_address=None,
                  uuid=start_req.uuid if start_req.uuid > 0 else None,
                  is_sender=start_req.is_sender,
                  expected_block_count=start_req.expected_block_count,
                  shard_push_schedules=shard_push_schedules,
              )
            asyncio.run(future.wait())
            raiden_resp.success = True
          elif (
              raiden_req.command
              == self._raiden_proto_module.ControlRequest.COMMAND_SHUTDOWN
          ):
            if hasattr(self._controller.worker_rpc_client, "shutdown_workers"):
              asyncio.run(self._controller.worker_rpc_client.shutdown_workers())
            self.stop()
            raiden_resp.success = True
        except Exception as e:
          raiden_resp.message = str(e)
        resp_bytes = raiden_resp.SerializeToString()
        conn.sendall(len(resp_bytes).to_bytes(4, "big") + resp_bytes)
    except Exception:  # pylint: disable=broad-except
      pass

    finally:
      conn.close()


class RaidenControllerClientFacade:
  """Client-side stub encapsulating real remote Network RPCs to a centralized RaidenControllerServer."""

  def __init__(
      self,
      controller_address: str,
      name_resolver: Optional[NameResolver] = None,
      proto_module: Optional[Any] = None,
      raiden_proto_module: Optional[Any] = None,
  ):
    """Accepts Controller server coordinate 'ip:port'."""
    self._address = controller_address
    self._name_resolver = name_resolver
    self._proto_module = proto_module or controller_service_pb2
    self._raiden_proto_module = raiden_proto_module or raiden_service_pb2

  def _raiden_id_to_proto(
      self,
      unit: RaidenId,
  ) -> Any:
    return self._raiden_proto_module.RaidenIdProto(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

  def _send_protobuf_rpc(self, req: Any) -> bool:
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

      resp = self._proto_module.ControllerResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return True
    finally:
      sock.close()

  def _send_raiden_protobuf_rpc_response(self, req: Any) -> Any:
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

      resp = self._raiden_proto_module.ControlResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return resp
    finally:
      sock.close()

  def _send_raiden_protobuf_rpc(self, req: Any) -> bool:
    self._send_raiden_protobuf_rpc_response(req)
    return True

  def register_work_unit(
      self,
      unit: RaidenId,
      shards: list[str],
      control_plane_rpc_address: Optional[str] = None,
      mesh_shape: Optional[typing.Sequence[int]] = None,
      layout: Optional[typing.Sequence[int]] = None,
      global_shape: Optional[typing.Sequence[int]] = None,
      itemsize: Optional[int] = None,
      pool_manifest: Optional[typing.Sequence[Any]] = None,
      layout_fingerprint: Optional[str] = None,
      page_tokens: Optional[int] = None,
      page_slice_tokens: Optional[int] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_rank: Optional[int] = None,
  ) -> None:
    """Sends remote RPC to register a physical worker entity with the central RaidenControllerServer.

    Args:
      unit: Work unit identifier owning the data shards.
      shards: list of physical Data TCP addresses (e.g. 'IP:Port').
      control_plane_rpc_address: Optional worker Control-Plane RPC servicer
        endpoint coordinate.
      mesh_shape: Optional logical mesh shape.
      layout: Optional minor_to_major mapping layout.
      global_shape: Optional global array shape.
      itemsize: Optional item size in bytes.
    """
    reg_req = self._raiden_proto_module.RegisterWorkUnitRequest(
        unit=self._raiden_id_to_proto(unit),
        shards=shards,
        control_plane_rpc_address=(
            control_plane_rpc_address if control_plane_rpc_address else ""
        ),
    )
    if mesh_shape:
      reg_req.mesh_shape.extend(mesh_shape)
    if layout:
      reg_req.layout.extend(layout)
    if global_shape:
      reg_req.global_shape.extend(global_shape)
    if itemsize:
      reg_req.itemsize = itemsize
    if pool_manifest is not None:
      for pool in pool_manifest:
        reg_req.pools.add().CopyFrom(_coerce_pool_spec_proto(pool))
    if layout_fingerprint is not None:
      reg_req.layout_fingerprint = layout_fingerprint
    if page_tokens is not None:
      reg_req.page_tokens = page_tokens
    if page_slice_tokens is not None:
      reg_req.page_slice_tokens = page_slice_tokens
    if transfer_parallelism is not None:
      reg_req.transfer_parallelism = transfer_parallelism
    if transfer_rank is not None:
      reg_req.transfer_rank = transfer_rank

    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_WORK_UNIT,
        register_work_unit_request=reg_req,
    )
    self._send_raiden_protobuf_rpc(req)

  def register_request_blocks(
      self,
      req_id: str,
      uuid: int,
      unit: RaidenId,
      block_ids: typing.Sequence[int],
  ) -> None:
    """Registers producer-owned block IDs with its source controller."""
    block_req = self._raiden_proto_module.RegisterRequestBlocksRequest(
        req_id=req_id,
        uuid=uuid,
        unit=self._raiden_id_to_proto(unit),
        block_ids=list(block_ids),
    )
    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_REQUEST_BLOCKS,
        register_request_blocks_request=block_req,
    )
    self._send_raiden_protobuf_rpc(req)

  def release_request_blocks(self, req_id: str, uuid: int) -> None:
    """Explicitly force-releases a complete request lifecycle generation."""
    release_req = self._raiden_proto_module.ReleaseRequestBlocksRequest(
        req_id=req_id, uuid=uuid, force=True
    )
    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_RELEASE_REQUEST_BLOCKS,
        release_request_blocks_request=release_req,
    )
    self._send_raiden_protobuf_rpc(req)

  def complete_request_blocks(
      self, req_id: str, uuid: int, unit: RaidenId
  ) -> None:
    """Votes one producer rank terminal using the aggregate-safe protocol."""
    complete_req = self._raiden_proto_module.CompleteRequestBlocksRequest(
        req_id=req_id,
        uuid=uuid,
        unit=self._raiden_id_to_proto(unit),
    )
    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_COMPLETE_REQUEST_BLOCKS,
        complete_request_blocks_request=complete_req,
    )
    self._send_raiden_protobuf_rpc(req)

  def cancel_request_blocks_if_unclaimed(self, req_id: str, uuid: int) -> bool:
    """Cancels D5 state unless a transfer plan has already claimed it."""
    cancel_req = (
        self._raiden_proto_module.CancelRequestBlocksIfUnclaimedRequest(
            req_id=req_id, uuid=uuid
        )
    )
    req = self._raiden_proto_module.ControlRequest(
        command=(
            self._raiden_proto_module.ControlRequest.COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED
        ),
        cancel_request_blocks_if_unclaimed_request=cancel_req,
    )
    response = self._send_raiden_protobuf_rpc_response(req)
    if response.response_data == "true":
      return True
    if response.response_data == "false":
      return False
    raise RuntimeError(
        "Remote Controller Server returned an invalid cancellation result: "
        f"{response.response_data!r}"
    )

  def coordinate_transfer(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      use_block_chunks: bool = False,
      is_sender: bool = True,
      expected_block_count: int = 0,
      uuid: int = 0,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      shard_push_schedules: Optional[dict] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      dst_device_block_ids: Optional[typing.Sequence[int]] = None,
      num_tokens: Optional[int] = None,
      parallelism: Optional[int] = None,
  ) -> bool:
    """Sends remote RPC to coordinate global least-loaded transfer and blocks until fully complete."""
    coord_req = self._proto_module.CoordinateTransferRequest(
        src_units=[self._raiden_id_to_proto(u) for u in src_units],
        dst_units=[self._raiden_id_to_proto(u) for u in dst_units],
        use_block_chunks=use_block_chunks,
        is_sender=is_sender,
        expected_block_count=expected_block_count,
        uuid=uuid,
        req_id=req_id if req_id else "",
        dst_controller_address=dst_controller_address
        if dst_controller_address
        else "",
        src_controller_address=src_controller_address
        if src_controller_address
        else "",
        dst_mem_type=int(dst_mem_type),
    )
    if dst_device_block_ids is not None:
      coord_req.dst_device_block_ids.extend(dst_device_block_ids)
    if num_tokens is not None:
      coord_req.num_tokens = num_tokens
    if parallelism is not None:
      coord_req.parallelism = parallelism

    req = self._proto_module.ControllerRequest(
        command=self._proto_module.ControllerRequest.COMMAND_COORDINATE_TRANSFER,
        coordinate_transfer_request=coord_req,
    )
    return self._send_protobuf_rpc(req)

  def register_transfer_schedule(
      self,
      src_units: list[RaidenId],
      dst_units: list[RaidenId],
      req_id: Optional[str] = None,
      use_block_chunks: bool = False,
      is_sender: bool = False,
      expected_block_count: int = 0,
      uuid: int = 0,
      dst_controller_address: Optional[str] = None,
      src_controller_address: Optional[str] = None,
      shard_push_schedules: Optional[dict] = None,
      dst_mem_type: RaidenMemoryType = RaidenMemoryType.DRAM,
      expected_pushes_per_pool: int = 0,
      transfer_pool_indices: Optional[typing.Sequence[int]] = None,
      pool_dtype_tags: Optional[typing.Sequence[str]] = None,
      dst_device_block_ids: Optional[typing.Sequence[int]] = None,
      src_schedule_keys: Optional[dict[RaidenId, int]] = None,
      parallelism: int = 0,
      num_tokens: int = 0,
  ) -> bool:
    """Inter-controller RPC to register computed push schedules and prepare receivers."""
    start_req = self._raiden_proto_module.StartTransferRequest(
        src_units=[self._raiden_id_to_proto(u) for u in src_units],
        dst_units=[self._raiden_id_to_proto(u) for u in dst_units],
        use_block_chunks=use_block_chunks,
        is_sender=is_sender,
        expected_block_count=expected_block_count,
        uuid=uuid,
        req_id=req_id if req_id else "",
        dst_mem_type=int(dst_mem_type),
        expected_pushes_per_pool=expected_pushes_per_pool,
        parallelism=parallelism,
        num_tokens=num_tokens,
    )
    if transfer_pool_indices is not None:
      start_req.transfer_pool_indices.extend(transfer_pool_indices)
    if pool_dtype_tags is not None:
      start_req.pool_dtype_tags.extend(pool_dtype_tags)
    if dst_device_block_ids is not None:
      start_req.dst_device_block_ids.extend(dst_device_block_ids)

    if src_schedule_keys is not None:
      if set(src_schedule_keys) != set(src_units):
        raise ValueError("Every source must have an explicit schedule key")
      if len(set(src_schedule_keys.values())) != len(src_schedule_keys):
        raise ValueError("Source schedule keys must be unique")

    if shard_push_schedules:
      for src_unit, push_schedules in shard_push_schedules.items():
        num_src_shards = len(push_schedules)
        for shard_idx, schedule in push_schedules.items():
          if src_schedule_keys is not None:
            if num_src_shards != 1:
              raise ValueError(
                  "Explicit source schedule keys require one local shard"
              )
            key_idx = src_schedule_keys[src_unit]
          else:
            key_idx = (
                int(src_unit.job_replica_id)
                if num_src_shards == 1
                else shard_idx
            )
          schedule_proto = self._raiden_proto_module.ShardPushScheduleProto()
          for (
              dst_peer,
              dst_shard_idx,
              dst_offset,
              src_offset,
              size,
              src_block_id,
              dst_block_id,
              src_stride,
              dst_stride,
              count,
          ) in schedule:
            entry_proto = schedule_proto.entries.add()
            entry_proto.dst_peer = dst_peer
            entry_proto.dst_shard_idx = dst_shard_idx
            entry_proto.dst_offset_bytes = dst_offset
            entry_proto.src_offset_bytes = src_offset
            entry_proto.size_bytes = size
            entry_proto.src_block_id = src_block_id
            entry_proto.dst_block_id = dst_block_id
            entry_proto.src_stride_bytes = src_stride
            entry_proto.dst_stride_bytes = dst_stride
            entry_proto.count = count
          if len(schedule_proto.entries) > 0:
            start_req.shard_push_schedules[key_idx].CopyFrom(schedule_proto)

    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_REGISTER_TRANSFER_SCHEDULE,
        start_transfer_request=start_req,
    )
    return self._send_raiden_protobuf_rpc(req)

  def start_transfer(self, *args, **kwargs) -> bool:
    """Alias for coordinate_transfer for backward compatibility."""
    return self.coordinate_transfer(*args, **kwargs)

  def shutdown(self) -> bool:
    """Sends remote RPC to trigger global cluster shutdown across all cooperating jobs."""
    req = self._raiden_proto_module.ControlRequest(
        command=self._raiden_proto_module.ControlRequest.COMMAND_SHUTDOWN
    )
    return self._send_raiden_protobuf_rpc(req)

  def get_metadata(self) -> list[Any]:
    """Queries the controller for all registered work units' metadata."""
    sock = connect_socket(
        self._address, timeout=300.0, resolver=self._name_resolver
    )
    try:
      req = self._raiden_proto_module.ControlRequest(
          command=self._raiden_proto_module.ControlRequest.COMMAND_GET_METADATA
      )
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

      resp = self._raiden_proto_module.ControlResponse()
      resp.ParseFromString(resp_bytes)
      if not resp.success:
        raise RuntimeError(
            f"Remote Controller Server execution failed: {resp.message}"
        )
      return list(resp.get_metadata_response.metadata)
    finally:
      sock.close()
