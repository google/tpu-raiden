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

"""High-performance JAX Disaggregated KV Cache Manager."""

from typing import Any, Callable, List, Optional

# Import JAX Nanobind binary extension shims
from api.jax import _kv_cache_manager

# Re-export type enum and request structure
DisaggTransferRequestType = _kv_cache_manager.DisaggTransferRequestType
DisaggTransferRequest = _kv_cache_manager.DisaggTransferRequest


class DisaggKVCacheManager:
  """Zero-copy JAX Disaggregated KV Cache Manager."""

  def __init__(
      self,
      device_arrays: List[Any],
      block_size: int = 1,
      local_port: Optional[int] = None,
      host_blocks_to_allocate: Optional[int] = None,
      external_host_ptrs: Optional[List[int]] = None,
      unsafe_skip_buffer_lock: bool = False,
      transport_parallelism: int = 1,
      worker_parallelism: int = 1,
  ):
    """Instantiates the sharded JAX Disaggregated KV Cache Manager.

    Args:
      device_arrays: List of physical sharded JAX device PJRT arrays
        representing the sharded KV caches.
      block_size: Physical tokens count per allocation page block.
      local_port: TCP socket server port for remote pulls/pushes coordinates.
      host_blocks_to_allocate: Max host blocks staging cache size.
      external_host_ptrs: Pinned external CPU host memory addresses list.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
      transport_parallelism: Number of parallel TCP streams a SINGLE H2H
        Push/Pull is striped across (the BlockTransport parallelism).
      worker_parallelism: Number of H2H worker threads draining the transfer
        queue, i.e. how many H2H transfers run concurrently.
    """
    self._impl = _kv_cache_manager.DisaggKVCacheManager(
        device_arrays,
        block_size,
        local_port,
        host_blocks_to_allocate,
        external_host_ptrs,
        unsafe_skip_buffer_lock,
        transport_parallelism,
        worker_parallelism,
    )

  def start(self):
    """Starts the background orchestration, worker and listener threads."""
    self._impl.start()

  def stop(self):
    """Stops the background orchestration, worker and listener threads."""
    self._impl.stop()

  def register_peer(
      self, name: str, ip: str, zmq_port: int, transport_port: int
  ):
    """Manually registers peer control plane (ZMQ) and data plane (TCP) endpoints."""
    self._impl.register_peer(name, ip, zmq_port, transport_port)

  def submit_request(
      self,
      request_id: int,
      req_type: Any,
      src_offsets: List[int] = [],
      dst_offsets: List[int] = [],
      sizes: List[int] = [],
      peer: str = "",
      block_ids: List[int] = [],
      entity_id: int = 0,
      pull: bool = False,
      callback: Optional[Callable[[Any], None]] = None,
  ):
    """Submits a disaggregated transfer request.

    Args:
      request_id: Globally unique identifier of this request.
      req_type: Type of transfer (PREFILL_D2H, DECODE_H2D, H2H_WRITE, H2H_READ).
      src_offsets: Source offsets major dimension (local memory).
      dst_offsets: Target offsets major dimension (local/peer memory).
      sizes: Copy sizes major dimension.
      peer: Target peer name registered via `register_peer`.
      block_ids: Block IDs list (host blocks).
      entity_id: Entity ID (passed to TCP BlockTransport).
      pull: If True, use PULL mode (decode pulls from prefill) instead of the
        default PUSH mode (prefill pushes to decode). Both the prefill
        (PREFILL_D2H) and decode (DECODE_H2D) requests of one transfer must set
        this identically, and in pull mode both must set `peer` to the other
        engine (registered on both sides).
      callback: User Python callback `cb(err)` invoked upon completion.
    """
    req = DisaggTransferRequest()
    req.request_id = request_id
    req.type = req_type
    req.pull_mode = pull
    req.src_offsets = src_offsets
    req.dst_offsets = dst_offsets
    req.sizes = sizes
    req.peer = peer
    req.block_ids = block_ids
    req.entity_id = entity_id
    if callback:
      # nanobind automatically casts C++ absl::Status to None on success,
      # or raises RuntimeError on non-OK status in Python.
      req.callback = callback
    else:
      req.callback = lambda status: None

    self._impl.submit_request(req)

  def local_port(self) -> Optional[int]:
    """Returns assigned ephemeral listener data plane port coordinates."""
    return self._impl.local_port()

  def zmq_control_port(self) -> int:
    """Returns assigned ephemeral ZMQ control plane port coordinates."""
    return self._impl.zmq_control_port()

  @property
  def num_layers(self) -> int:
    """Returns total layers registered."""
    return self._impl.num_layers

  @property
  def num_shards(self) -> int:
    """Returns physical JAX devices shards count."""
    return self._impl.num_shards

  @property
  def slice_byte_size(self) -> int:
    """Returns individual major dimension slice byte size capacity."""
    return self._impl.slice_byte_size
