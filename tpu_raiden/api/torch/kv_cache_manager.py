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

"""High-performance PyTorch KV Cache Manager (repurposed as TransferEngine)."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_HOST_IMPL = None
_TORCH_IMPL = None


def _layout_kind_id(kind: Any) -> int:
  # pylint: disable=g-import-not-at-top
  from tpu_raiden.api.torch import hybrid_layout
  # pylint: enable=g-import-not-at-top

  if kind == hybrid_layout.LayerKind.FULL_ATTENTION:
    return 0
  if kind == hybrid_layout.LayerKind.MAMBA_STATE:
    return 1
  if kind == hybrid_layout.LayerKind.OPAQUE:
    return 2
  raise ValueError(f"unknown layer kind: {kind!r}")


def _layout_to_native_tuple(layout: Any) -> tuple[int, int, list[tuple]]:
  return (
      _layout_kind_id(layout.kind),
      int(layout.slot_bytes),
      [(
          region.name,
          int(region.offset_bytes),
          int(region.stride_bytes),
          int(region.unit_bytes),
          int(region.num_units),
          int(region.units_per_stride),
      ) for region in layout.regions],
  )


def _host_impl():
  """Returns the host-only extension without importing torch_tpu."""
  global _HOST_IMPL
  if _HOST_IMPL is None:
    # pylint: disable=g-import-not-at-top
    from tpu_raiden.frameworks.torch import _tpu_raiden_host as impl
    # pylint: enable=g-import-not-at-top
    _HOST_IMPL = impl
  return _HOST_IMPL


def _torch_impl():
  """Returns the torch-backed extension, loading torch_tpu runtime first."""
  global _TORCH_IMPL
  if _TORCH_IMPL is None:
    # pylint: disable=g-import-not-at-top
    from tpu_raiden.api.torch import torch_tpu_common_loader

    torch_tpu_common_loader.load_torch_tpu_common()
    from tpu_raiden.frameworks.torch import _tpu_raiden_torch as impl
    # pylint: enable=g-import-not-at-top
    _TORCH_IMPL = impl
  return _TORCH_IMPL


class KVCacheManager:
  """Wrapper around compiled C++ KV Cache Manager.

  This class has been repurposed to expose the TransferEngine API interface
  for distributed disaggregated KV-cache movement on PyTorch TPUs.
  """

  def __init__(
      self,
      kv_caches: List[Any],
      local_control_port: int,
      max_blocks: Optional[int] = None,
      num_slots: Optional[int] = None,
      timeout_s: float = 120.0,
      unsafe_skip_buffer_lock: bool = True,
      host_blocks_to_allocate: Optional[int] = None,
      parallelism: int = 4,
      node_id: int = 0,
      listener_port: Optional[int] = None,
  ):
    """Instantiates the TransferEngine-based KVCacheManager.

    Args:
      kv_caches: List of device-placed contiguous Tensors representing the
        sharded KV caches.
      local_control_port: TCP socket server port for control plane coordination.
      max_blocks: Maximum number of blocks per staging slot.
      num_slots: Number of transfer slots to allocate.
      timeout_s: Timeout in seconds for transfer operations.
      unsafe_skip_buffer_lock: Skip dynamic safety locking.
      host_blocks_to_allocate: Legacy/unified total blocks to allocate in host
        pool.
      parallelism: Number of parallel network copies per layer.
      node_id: Unique identifier for this host/node in the distributed mesh.
      listener_port: Sockets server port for incoming C++ KVCacheListener
        commands.
    """
    self._block_layouts = None
    self._admission_summary = None
    impl = _torch_impl()
    if host_blocks_to_allocate is not None:
      self._impl = impl.KVCacheManager(
          kv_caches,
          local_control_port if local_control_port > 0 else None,
          host_blocks_to_allocate,
          unsafe_skip_buffer_lock,
          parallelism,
      )
    else:
      if max_blocks is None or num_slots is None:
        raise ValueError(
            "Must specify either (max_blocks, num_slots) or"
            " host_blocks_to_allocate."
        )
      self._impl = impl.KVCacheManager(
          kv_caches=kv_caches,
          node_id=node_id,
          local_control_port=local_control_port,
          max_blocks=max_blocks,
          num_slots=num_slots,
          timeout_s=timeout_s,
          unsafe_skip_buffer_lock=unsafe_skip_buffer_lock,
          listener_port=listener_port,
          parallelism=parallelism,
      )

  @classmethod
  def create_host_only(
      cls,
      *,
      num_layers: int,
      num_shards: int = 1,
      slice_byte_size: int,
      node_id: int,
      local_port: Optional[int] = None,
      host_blocks: int,
      parallelism: int = 4,
  ) -> "KVCacheManager":
    """Creates a CPU-only manager backed by Raiden-owned host memory."""
    obj = cls.__new__(cls)
    obj._block_layouts = None
    obj._admission_summary = None
    obj._impl = _host_impl().KVCacheManager(
        num_layers=num_layers,
        num_shards=num_shards,
        slice_byte_size=slice_byte_size,
        node_id=node_id,
        local_port=local_port,
        host_blocks_to_allocate=host_blocks,
        parallelism=parallelism,
    )
    return obj

  @property
  def node_id(self) -> int:
    """Returns the active Worker or Shard ID."""
    return self._impl.node_id()

  def get_local_endpoints(self) -> List[Dict[str, Any]]:
    """Returns the active Raiden endpoint descriptors."""
    return self._impl.get_local_endpoints()

  @property
  def local_control_port(self) -> int:
    """Returns the active control plane listener port."""
    return self._impl.local_control_port

  @property
  def local_port(self) -> int:
    """Returns the active data port."""
    return self._impl.local_port

  @property
  def transfer_address(self) -> str:
    """Returns the formatted data transfer endpoint string (host:port)."""
    return self._impl.transfer_address

  @property
  def listener_address(self) -> str:
    """Returns the formatted control listener endpoint string (host:port)."""
    return self._impl.listener_address

  def register_active_plan(
      self, uuid: int, request: Union[bytes, Any], is_sender: bool
  ) -> None:
    """Registers a serialized StartTransferRequest for strided push."""
    if hasattr(request, "SerializeToString"):
      request = request.SerializeToString()
    if not isinstance(request, (bytes, bytearray)):
      raise TypeError("request must be bytes or a protobuf message")
    self._impl.register_active_plan(uuid, bytes(request), is_sender)

  def unregister_active_plan(self, uuid: int) -> None:
    """Removes a previously registered strided push plan."""
    self._impl.unregister_active_plan(uuid)

  def push_registered_plan(
      self,
      uuid: int,
      peer: str,
      src_block_ids: Sequence[int],
      dst_block_ids: Sequence[int],
      layer_idx: int = -1,
      parallelism: int = 1,
  ) -> None:
    """Synchronously pushes host blocks using an already registered plan."""
    self._impl.push_registered_plan(
        uuid,
        peer,
        list(src_block_ids),
        list(dst_block_ids),
        layer_idx,
        parallelism,
    )

  def read_block_bytes(self, layer_idx: int, block_id: int) -> bytes:
    """Returns one host block as bytes."""
    return self._impl.read_block_bytes(layer_idx, block_id)

  def write_block_bytes(
      self, layer_idx: int, block_id: int, payload: bytes
  ) -> None:
    """Overwrites one host block with bytes."""
    self._impl.write_block_bytes(layer_idx, block_id, payload)

  def set_block_layouts(self, layouts: Sequence[Any]) -> None:
    """Registers one hybrid block layout per wrapped layer."""
    # pylint: disable=g-import-not-at-top
    from tpu_raiden.api.torch import hybrid_layout
    # pylint: enable=g-import-not-at-top

    coerced = tuple(hybrid_layout.coerce_layer_layout(layout)
                    for layout in layouts)
    expected_layers = int(self._impl.num_layers)
    if len(coerced) != expected_layers:
      raise ValueError(
          "layout count must match manager layer count: "
          f"got {len(coerced)}, expected {expected_layers}")
    for layer_idx, layout in enumerate(coerced):
      manager_slot_bytes = self._manager_slot_byte_size(layer_idx)
      try:
        layout.validate(manager_slot_bytes)
      except Exception as exc:
        raise ValueError(
            f"invalid block layout for layer {layer_idx}: {exc}") from exc
    if hasattr(self._impl, "set_block_layouts_native"):
      self._impl.set_block_layouts_native(
          [_layout_to_native_tuple(layout) for layout in coerced])
    self._block_layouts = coerced

  def _manager_slot_byte_size(self, layer_idx: int) -> int:
    if hasattr(self._impl, "layer_block_byte_size"):
      return int(self._impl.layer_block_byte_size(layer_idx))
    return int(self._impl.slice_byte_size)

  def get_block_ref(
      self, layer_idx: int, block_id: int, shard_idx: int = 0
  ) -> Dict[str, Any]:
    """Returns a typed reference descriptor for one host-side hybrid block."""
    if self._block_layouts is None:
      raise RuntimeError("block layouts are not set")
    if hasattr(self._impl, "get_hybrid_block_ref_native"):
      return dict(
          self._impl.get_hybrid_block_ref_native(
              layer_idx=layer_idx, shard_idx=shard_idx, block_id=block_id))
    if layer_idx < 0 or layer_idx >= len(self._block_layouts):
      raise IndexError("layer index out of range")
    layout = self._block_layouts[layer_idx]
    ptr = int(
        self._impl.get_block_host_pointer(
            layer_idx=layer_idx, shard_idx=shard_idx, block_id=block_id))
    return {
        "ptr": ptr,
        "slot_bytes": layout.slot_bytes,
        "kind": layout.kind.value,
        "layer_idx": layer_idx,
        "shard_idx": shard_idx,
        "block_id": block_id,
        "regions": [region.to_dict() for region in layout.regions],
    }

  def fa_layer_indices(self) -> List[int]:
    """Returns the registered full-attention layer indices."""
    if self._block_layouts is None:
      return []
    if hasattr(self._impl, "layer_indices_of_kind_native"):
      # 0 is LayerKind::kFullAttention in the native binding.
      return [int(layer_idx)
              for layer_idx in self._impl.layer_indices_of_kind_native(0)]
    # pylint: disable=g-import-not-at-top
    from tpu_raiden.api.torch import hybrid_layout
    # pylint: enable=g-import-not-at-top

    return [
        layer_idx for layer_idx, layout in enumerate(self._block_layouts)
        if layout.kind == hybrid_layout.LayerKind.FULL_ATTENTION
    ]

  def admit_qwen35_kv_cache(self, spec: Any) -> Dict[str, Any]:
    """Admits a Qwen3.5 hybrid KV cache layout into this manager."""
    # pylint: disable=g-import-not-at-top
    from tpu_raiden.api.torch import hybrid_layout
    # pylint: enable=g-import-not-at-top

    if not isinstance(spec, hybrid_layout.Qwen35AdmissionSpec):
      raise TypeError("spec must be a Qwen35AdmissionSpec")
    if int(self._impl.num_layers) != spec.num_layers:
      raise ValueError(
          "Qwen3.5 admission layer count mismatch: "
          f"manager={int(self._impl.num_layers)} spec={spec.num_layers}")
    layouts = tuple(hybrid_layout.coerce_layer_layout(layout)
                    for layout in spec.layouts)
    mismatched_layer_sizes = []
    for layer_idx, layout in enumerate(layouts):
      manager_slot_bytes = self._manager_slot_byte_size(layer_idx)
      allowed_slot_bytes = {int(layout.slot_bytes)}
      if layout.kind == hybrid_layout.LayerKind.MAMBA_STATE:
        allowed_slot_bytes.add(int(spec.gdn_used_bytes))
      if manager_slot_bytes not in allowed_slot_bytes:
        mismatched_layer_sizes.append((layer_idx, manager_slot_bytes))
    if mismatched_layer_sizes:
      layer_idx, manager_slot_bytes = mismatched_layer_sizes[0]
      raise ValueError(
          "Qwen3.5 admission slot byte size mismatch: "
          f"layer={layer_idx} manager={manager_slot_bytes} "
          f"spec={spec.slot_bytes} gdn_used={spec.gdn_used_bytes}")
    self.set_block_layouts(layouts)
    fa_layers = self.fa_layer_indices()
    summary = {
        "admitted": True,
        "topology": spec.topology,
        "model_server_role": spec.model_server_role,
        "num_layers": spec.num_layers,
        "fa_layers": len(fa_layers),
        "gdn_layers": spec.num_layers - len(fa_layers),
        "block_tokens": spec.block_tokens,
        "slot_bytes": spec.slot_bytes,
        "gdn_used_bytes": spec.gdn_used_bytes,
    }
    self._admission_summary = dict(summary)
    return summary

  def admission_summary(self) -> Dict[str, Any]:
    """Returns the last successful Qwen3.5 admission summary."""
    if self._admission_summary is None:
      return {"admitted": False}
    return dict(self._admission_summary)

  def register_recv(
      self, uuid: int, req_id: str, expected_block_count: int
  ) -> None:
    """[EXPERIMENTAL] Registers expected incoming blocks for decentralized push resharding.

    This allocates staging slots in the C++ receiver engine and sets the
    synchronization barrier for the expected physical block-pushes. The
    engine will automatically trigger Host-to-Device (H2D) copy to TPU HBM
    once this count is reached.

    This API is experimental and subject to change.

    Args:
      uuid: Unique identifier for the transfer transaction.
      req_id: Request ID associated with the transfer.
      expected_block_count: The total number of physical block-pushes expected
        from all contributing source ranks.
    """
    self._impl.RegisterRecv(uuid, req_id, expected_block_count)

  def register_read(
      self, req_id: str, uuid: int, block_ids: List[int]
  ) -> bool:
    """Producer node notifies the registry/peer that blocks are ready for read.

    Args:
      req_id: The request ID of the transfer operation.
      uuid: The UUID of the request.
      block_ids: The list of block IDs to be read.

    Returns:
      True if a transfer is indeed needed; False if there is nothing to be
      transferred.
    """
    return bool(self._impl.notify_for_read(req_id, uuid, block_ids))

  def start_read(
      self,
      req_id: str,
      uuid: int,
      remote_endpoint: Union[str, List[Dict[str, Any]]],
      remote_block_ids: List[int],
      local_block_ids: List[int],
      parallelism: int = 1,
  ) -> int:
    """Consumer node initiates an asynchronous pull of blocks from a remote peer."""
    return self._impl.start_read(
        req_id,
        uuid,
        remote_endpoint,
        remote_block_ids,
        local_block_ids,
        parallelism,
    )

  def poll_stats(self) -> Tuple[List[str], List[str], List[str]]:
    """Polls the status of all active background transfer operations.

    Returns:
      A tuple of (done_sending, done_recving, failed_recving) lists of request
      IDs.
    """
    return self._impl.complete_read()

  def d2h(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
  ) -> Any:
    """Device-to-Host (D2H) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.D2h(src_offsets, dst_offsets, copy_sizes)

  def h2d(
      self,
      src_offsets: List[int],
      dst_offsets: List[int],
      copy_sizes: List[int] | None = None,
  ) -> Any:
    """Host-to-Device (H2D) copy transfer.

    Args:
      src_offsets: Source block offsets.
      dst_offsets: Destination block offsets.
      copy_sizes: Optional number of contiguous blocks to copy per segment
        (defaults to 1 block per segment).

    Returns:
      A future representing the asynchronous copy transfer operation.
    """
    if copy_sizes is None:
      copy_sizes = [1] * len(src_offsets)
    return self._impl.H2d(src_offsets, dst_offsets, copy_sizes)

  @property
  def listener_port(self) -> Optional[int]:
    """Returns the active local port assigned to the C++ KVCacheListener."""
    return self._impl.listener_port

  @property
  def is_listener_active(self) -> bool:
    """Returns whether the native C++ KVCacheListener is actively running."""
    return self._impl.is_listener_active
