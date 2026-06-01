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

  def await_pull(
      self,
      uuid: int,
      req_id: int,
      src_offsets: List[int],
      sizes: List[int],
      peer: str,
      entity_id: int = 0,
      callback: Optional[Callable[[Any], None]] = None,
  ):
    """Producer (prefill) side of a pull transfer.

    Stages the source device data into manager-allocated host staging blocks and
    waits for `peer` to pull them. The caller specifies ONLY the source device
    region; the staging host buffer is allocated (and released) by the manager.

    Args:
      uuid: Globally-unique transfer key; must equal the consumer `pull()`'s
        uuid. This is what the handshake uses to pair producer and consumer.
      req_id: Opaque caller bookkeeping tag (e.g. an inference request id);
        carried through unchanged and not used for matching.
      src_offsets: Source offsets (major dim) of the device KV to stage. The
        consumer's pull() must name the same source region (validated).
      sizes: Copy sizes (major dim); each must be a multiple of `block_size`.
      peer: Consumer engine name, registered via `register_peer` (bidirectional).
      entity_id: Entity id passed to the staging allocator / transport.
      callback: `cb(err)` invoked on completion (None on success).
    """
    self._submit(
        uuid=uuid,
        req_id=req_id,
        req_type=DisaggTransferRequestType.PREFILL_D2H,
        src_offsets=src_offsets,
        dst_offsets=[],  # empty -> manager auto-allocates the staging blocks
        sizes=sizes,
        peer=peer,
        entity_id=entity_id,
        callback=callback,
    )

  def pull(
      self,
      uuid: int,
      req_id: int,
      src_offsets: List[int],
      dst_offsets: List[int],
      sizes: List[int],
      peer: str,
      entity_id: int = 0,
      callback: Optional[Callable[[Any], None]] = None,
  ):
    """Consumer (decode) side of a pull transfer.

    Pulls the producer's staged data into manager-allocated local host staging,
    then writes it into the destination device offsets. The caller specifies the
    source region (for identity validation against the producer) and the local
    destination region; staging is allocated by the manager.

    Args:
      uuid: Globally-unique transfer key; must equal the producer
        `await_pull()`'s uuid (the handshake match key).
      req_id: Opaque caller bookkeeping tag; carried through, not used to match.
      src_offsets: Source offsets (major dim) at the producer that this pull
        expects; validated against the producer's await_pull (must match).
      dst_offsets: Destination offsets (major dim) in the local device KV.
      sizes: Copy sizes (major dim); each must be a multiple of `block_size`.
      peer: Producer engine name, registered via `register_peer` (bidirectional).
      entity_id: Entity id passed to the staging allocator / transport.
      callback: `cb(err)` invoked on completion (None on success).
    """
    self._submit(
        uuid=uuid,
        req_id=req_id,
        req_type=DisaggTransferRequestType.DECODE_H2D,
        src_offsets=src_offsets,
        dst_offsets=dst_offsets,
        sizes=sizes,
        peer=peer,
        entity_id=entity_id,
        callback=callback,
    )

  def _submit(
      self,
      uuid: int,
      req_type: Any,
      req_id: int = 0,
      src_offsets: List[int] = [],
      dst_offsets: List[int] = [],
      sizes: List[int] = [],
      peer: str = "",
      block_ids: List[int] = [],
      entity_id: int = 0,
      callback: Optional[Callable[[Any], None]] = None,
  ):
    """Low-level transfer submission. Internal: prefer await_pull() / pull()."""
    req = DisaggTransferRequest()
    req.uuid = uuid
    req.req_id = req_id
    req.type = req_type
    req.src_offsets = src_offsets
    req.dst_offsets = dst_offsets
    req.sizes = sizes
    req.peer = peer
    req.block_ids = block_ids
    req.entity_id = entity_id
    if callback:
      # The callback receives None on success, or the error message string on
      # failure (the C++ side passes std::optional<std::string>).
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
