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

"""Hybrid KV-cache block layout derivation for Raiden.

The Qwen3.5 admission helpers mirror the physical unified-block-pool geometry
used by the TPU vLLM connector. Stage 2 uses these descriptors to teach Raiden
which bytes in each physical page belong to full-attention KV and which belong
to GDN/Mamba state, without performing any reshard transfer.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Literal, Mapping, Sequence


def _cdiv(value: int, divisor: int) -> int:
  if divisor <= 0:
    raise ValueError("divisor must be positive")
  return (value + divisor - 1) // divisor


def _align_to(value: int, alignment: int) -> int:
  if alignment <= 0:
    raise ValueError("alignment must be positive")
  return _cdiv(value, alignment) * alignment


class LayerKind(enum.Enum):
  FULL_ATTENTION = "full_attention"
  MAMBA_STATE = "mamba_state"
  OPAQUE = "opaque"


@dataclasses.dataclass(frozen=True)
class RegionSpec:
  name: str
  offset_bytes: int
  stride_bytes: int
  unit_bytes: int
  num_units: int
  units_per_stride: int = 1

  @property
  def live_bytes(self) -> int:
    return self.unit_bytes * self.num_units * self.units_per_stride

  @property
  def extent_end_bytes(self) -> int:
    if self.num_units == 0:
      return self.offset_bytes
    return (self.offset_bytes + (self.num_units - 1) * self.stride_bytes +
            self.units_per_stride * self.unit_bytes)

  def validate(self, slot_bytes: int) -> None:
    if not self.name:
      raise ValueError("region name must be non-empty")
    for field_name in (
        "offset_bytes",
        "stride_bytes",
        "unit_bytes",
        "num_units",
        "units_per_stride",
    ):
      value = getattr(self, field_name)
      if value < 0:
        raise ValueError(f"region {self.name} {field_name} must be >= 0")
    if self.num_units > 0 and self.unit_bytes <= 0:
      raise ValueError(f"region {self.name} unit_bytes must be positive")
    if self.num_units > 1 and self.stride_bytes <= 0:
      raise ValueError(f"region {self.name} stride_bytes must be positive")
    if self.units_per_stride <= 0:
      raise ValueError(
          f"region {self.name} units_per_stride must be positive")
    if self.num_units > 0 and self.stride_bytes < (
        self.units_per_stride * self.unit_bytes):
      raise ValueError(
          f"region {self.name} stride_bytes is smaller than packed units")
    if self.extent_end_bytes > slot_bytes:
      raise ValueError(
          f"region {self.name} exceeds slot bytes: "
          f"end={self.extent_end_bytes} slot={slot_bytes}")

  def to_dict(self) -> dict[str, int | str]:
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LayerBlockLayout:
  kind: LayerKind
  slot_bytes: int
  regions: tuple[RegionSpec, ...]

  def validate(self, manager_slot_bytes: int | None = None) -> None:
    if self.slot_bytes <= 0:
      raise ValueError("layout slot_bytes must be positive")
    if not self.regions:
      raise ValueError("layout must contain at least one region")
    for region in self.regions:
      region.validate(self.slot_bytes)
    if manager_slot_bytes is not None:
      if manager_slot_bytes <= 0:
        raise ValueError("manager slot byte size must be positive")
      compact_mamba = (
          self.kind == LayerKind.MAMBA_STATE and
          manager_slot_bytes <= self.slot_bytes)
      if self.slot_bytes != manager_slot_bytes and not compact_mamba:
        raise ValueError(
            "layout slot_bytes must match manager slot byte size: "
            f"layout={self.slot_bytes} manager={manager_slot_bytes}")
      if compact_mamba and manager_slot_bytes != self.slot_bytes:
        for region in self.regions:
          region.validate(manager_slot_bytes)

  def to_dict(self) -> dict[str, object]:
    return {
        "kind": self.kind.value,
        "slot_bytes": self.slot_bytes,
        "regions": [region.to_dict() for region in self.regions],
    }


@dataclasses.dataclass(frozen=True)
class ModelKVConfig:
  num_layers: int
  fa_layer_indices: tuple[int, ...]
  num_kv_heads: int
  head_dim: int
  kv_itemsize: int
  linear_num_key_heads: int
  linear_num_value_heads: int
  linear_key_head_dim: int
  linear_value_head_dim: int
  conv_kernel: int
  linear_num_ssm_heads: int | None = None
  conv_itemsize: int = 2
  ssm_itemsize: int = 4
  kv_packing: int = 4
  physical_fa_head_slots: int | None = None
  prefill_workers: int = 1


@dataclasses.dataclass(frozen=True)
class Qwen35AdmissionSpec:
  topology: Literal["pcp8_prefill", "tp2dp4_decode"]
  num_layers: int
  fa_layer_indices: tuple[int, ...]
  requested_tokens: int
  block_tokens: int
  slot_bytes: int
  gdn_used_bytes: int
  model_server_role: Literal["kv_producer", "kv_consumer"]
  pcp_size: int
  tp_size: int
  dp_size: int
  local_kv_heads: int
  physical_fa_head_slots: int
  local_linear_key_heads: int
  local_linear_value_heads: int
  layouts: tuple[LayerBlockLayout, ...]


def _coerce_kind(value: LayerKind | str) -> LayerKind:
  if isinstance(value, LayerKind):
    return value
  normalized = str(value).lower()
  for kind in LayerKind:
    if normalized in (kind.value, kind.name.lower()):
      return kind
  raise ValueError(f"unknown layer kind: {value!r}")


def coerce_region_spec(value: RegionSpec | Mapping[str, object]) -> RegionSpec:
  if isinstance(value, RegionSpec):
    return value
  return RegionSpec(
      name=str(value["name"]),
      offset_bytes=int(value["offset_bytes"]),
      stride_bytes=int(value["stride_bytes"]),
      unit_bytes=int(value["unit_bytes"]),
      num_units=int(value["num_units"]),
      units_per_stride=int(value.get("units_per_stride", 1)),
  )


def coerce_layer_layout(
    value: LayerBlockLayout | Mapping[str, object]) -> LayerBlockLayout:
  if isinstance(value, LayerBlockLayout):
    return value
  regions = value["regions"]
  if not isinstance(regions, Sequence):
    raise TypeError("layout regions must be a sequence")
  return LayerBlockLayout(
      kind=_coerce_kind(value["kind"]),
      slot_bytes=int(value["slot_bytes"]),
      regions=tuple(coerce_region_spec(region) for region in regions),
  )


def derive_gdn_used_bytes(cfg: ModelKVConfig) -> int:
  conv_segments = cfg.conv_kernel - 1
  if conv_segments < 0:
    raise ValueError("conv_kernel must be positive")
  conv_stride_bytes = (
      (cfg.linear_num_key_heads * 2 + cfg.linear_num_value_heads) *
      cfg.linear_key_head_dim * cfg.conv_itemsize)
  recurrent_head_bytes = (
      cfg.linear_key_head_dim * cfg.linear_value_head_dim * cfg.ssm_itemsize)
  return (conv_segments * conv_stride_bytes +
          cfg.linear_num_value_heads * recurrent_head_bytes)


def _physical_fa_head_slots(cfg: ModelKVConfig) -> int:
  if cfg.physical_fa_head_slots is not None:
    return cfg.physical_fa_head_slots
  return _cdiv(cfg.num_kv_heads * 2, cfg.kv_packing) * cfg.kv_packing // 2


def _fa_token_bytes(cfg: ModelKVConfig) -> int:
  return (_physical_fa_head_slots(cfg) * 2 * cfg.head_dim * cfg.kv_itemsize)


def derive_unified_block_size(cfg: ModelKVConfig,
                              requested_tokens: int) -> int:
  """Derives physical page tokens for a unified Qwen3.5-style block.

  `requested_tokens` is kept as a lower bound. For Qwen3.5, the GDN state must
  fit in the same physical page as the FA token payload, and the model-server
  page size is aligned to 16 tokens.
  """
  if requested_tokens <= 0:
    raise ValueError("requested_tokens must be positive")
  gdn_fit_tokens = _align_to(
      _cdiv(derive_gdn_used_bytes(cfg), _fa_token_bytes(cfg)), 16)
  return max(requested_tokens, gdn_fit_tokens)


def derive_slot_bytes(cfg: ModelKVConfig, block_tokens: int) -> int:
  if block_tokens <= 0:
    raise ValueError("block_tokens must be positive")
  return block_tokens * _fa_token_bytes(cfg)


def _fa_region(cfg: ModelKVConfig, block_tokens: int) -> RegionSpec:
  return RegionSpec(
      name="fa_payload",
      offset_bytes=0,
      stride_bytes=_fa_token_bytes(cfg),
      unit_bytes=2 * cfg.head_dim * cfg.kv_itemsize,
      num_units=block_tokens,
      units_per_stride=cfg.num_kv_heads,
  )


def _gdn_regions(cfg: ModelKVConfig) -> tuple[RegionSpec, ...]:
  conv_segments = cfg.conv_kernel - 1
  head_bytes = cfg.linear_key_head_dim * cfg.conv_itemsize
  q_bytes = cfg.linear_num_key_heads * head_bytes
  k_bytes = cfg.linear_num_key_heads * head_bytes
  v_bytes = cfg.linear_num_value_heads * head_bytes
  conv_stride_bytes = q_bytes + k_bytes + v_bytes
  ssm_heads = (cfg.linear_num_ssm_heads
               if cfg.linear_num_ssm_heads is not None
               else cfg.linear_num_value_heads)
  ssm_head_bytes = (
      cfg.linear_key_head_dim * cfg.linear_value_head_dim * cfg.ssm_itemsize)
  return (
      RegionSpec(
          name="gdn_conv_q",
          offset_bytes=0,
          stride_bytes=conv_stride_bytes,
          unit_bytes=head_bytes,
          num_units=conv_segments,
          units_per_stride=cfg.linear_num_key_heads,
      ),
      RegionSpec(
          name="gdn_conv_k",
          offset_bytes=q_bytes,
          stride_bytes=conv_stride_bytes,
          unit_bytes=head_bytes,
          num_units=conv_segments,
          units_per_stride=cfg.linear_num_key_heads,
      ),
      RegionSpec(
          name="gdn_conv_v",
          offset_bytes=q_bytes + k_bytes,
          stride_bytes=conv_stride_bytes,
          unit_bytes=head_bytes,
          num_units=conv_segments,
          units_per_stride=cfg.linear_num_value_heads,
      ),
      RegionSpec(
          name="gdn_ssm",
          offset_bytes=conv_segments * conv_stride_bytes,
          stride_bytes=ssm_head_bytes,
          unit_bytes=ssm_head_bytes,
          num_units=ssm_heads,
          units_per_stride=1,
      ),
  )


def build_layer_layouts(cfg: ModelKVConfig,
                        block_tokens: int,
                        slot_bytes: int | None = None) -> list[LayerBlockLayout]:
  if slot_bytes is None:
    slot_bytes = derive_slot_bytes(cfg, block_tokens)
  fa_indices = set(cfg.fa_layer_indices)
  layouts: list[LayerBlockLayout] = []
  for layer_idx in range(cfg.num_layers):
    if layer_idx in fa_indices:
      layout = LayerBlockLayout(
          kind=LayerKind.FULL_ATTENTION,
          slot_bytes=slot_bytes,
          regions=(_fa_region(cfg, block_tokens),),
      )
    else:
      layout = LayerBlockLayout(
          kind=LayerKind.MAMBA_STATE,
          slot_bytes=slot_bytes,
          regions=_gdn_regions(cfg),
      )
    layout.validate(slot_bytes)
    layouts.append(layout)
  return layouts


def _qwen35_cfg(*, local_kv_heads: int, physical_fa_head_slots: int,
                local_linear_key_heads: int,
                local_linear_value_heads: int,
                local_linear_ssm_heads: int) -> ModelKVConfig:
  return ModelKVConfig(
      num_layers=60,
      fa_layer_indices=tuple(range(3, 60, 4)),
      num_kv_heads=local_kv_heads,
      head_dim=256,
      kv_itemsize=1,
      linear_num_key_heads=local_linear_key_heads,
      linear_num_value_heads=local_linear_value_heads,
      linear_num_ssm_heads=local_linear_ssm_heads,
      linear_key_head_dim=128,
      linear_value_head_dim=128,
      conv_kernel=4,
      physical_fa_head_slots=physical_fa_head_slots,
      prefill_workers=4,
  )


def qwen35_397b_admission_spec(topology: str) -> Qwen35AdmissionSpec:
  """Returns the Qwen3.5-397B FP8 unified-page admission spec.

  Supported topology names match the Stage 2 execution document:
  `pcp8_prefill` for the standalone PCP8/EP8 local view, and
  `tp2dp4_decode` for the standalone TP2*DP4/EP8 local view. The names preserve
  the connector admission selectors used by vLLM; stage 2 launches these as
  independent standalone servers, not as a live disaggregated pair.
  """
  if topology == "pcp8_prefill":
    cfg = _qwen35_cfg(
        local_kv_heads=2,
        physical_fa_head_slots=2,
        local_linear_key_heads=16,
        local_linear_value_heads=64,
        local_linear_ssm_heads=64,
    )
    block_tokens = 4096
    spec_kwargs = {
        "model_server_role": "kv_producer",
        "pcp_size": 8,
        "tp_size": 1,
        "dp_size": 1,
    }
  elif topology == "tp2dp4_decode":
    cfg = _qwen35_cfg(
        local_kv_heads=1,
        physical_fa_head_slots=2,
        local_linear_key_heads=8,
        local_linear_value_heads=32,
        local_linear_ssm_heads=32,
    )
    block_tokens = 2048
    spec_kwargs = {
        "model_server_role": "kv_consumer",
        "pcp_size": 1,
        "tp_size": 2,
        "dp_size": 4,
    }
  else:
    raise ValueError(f"unsupported Qwen3.5 admission topology: {topology!r}")

  slot_bytes = max(derive_slot_bytes(cfg, block_tokens),
                   derive_gdn_used_bytes(cfg))
  layouts = tuple(build_layer_layouts(cfg, block_tokens, slot_bytes))
  return Qwen35AdmissionSpec(
      topology=topology,  # pytype: disable=wrong-arg-types
      num_layers=cfg.num_layers,
      fa_layer_indices=cfg.fa_layer_indices,
      requested_tokens=1024,
      block_tokens=block_tokens,
      slot_bytes=slot_bytes,
      gdn_used_bytes=derive_gdn_used_bytes(cfg),
      local_kv_heads=cfg.num_kv_heads,
      physical_fa_head_slots=_physical_fa_head_slots(cfg),
      local_linear_key_heads=cfg.linear_num_key_heads,
      local_linear_value_heads=cfg.linear_num_value_heads,
      layouts=layouts,
      **spec_kwargs,  # pytype: disable=wrong-keyword-args
  )
