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

"""Generic storage/pool/region descriptors for Raiden block admission.

A *storage* is one whole allocation wrapped by a KVCacheManager constructor
(constructor order gives ``storage_index``). A *pool* is an array of equally
strided blocks inside one storage; *regions* describe the live bytes inside
one block. All descriptors are byte-level and model-agnostic: callers (e.g.
the TPU vLLM connector) derive them from their own model and kernel layouts.
``tag`` and ``dtype_tag`` are opaque to Raiden — stored, filtered, and echoed,
never parsed.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence


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
    return (
        self.offset_bytes
        + (self.num_units - 1) * self.stride_bytes
        + self.units_per_stride * self.unit_bytes
    )

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
      raise ValueError(f"region {self.name} units_per_stride must be positive")
    if self.num_units > 0 and self.stride_bytes < (
        self.units_per_stride * self.unit_bytes
    ):
      raise ValueError(
          f"region {self.name} stride_bytes is smaller than packed units"
      )
    if self.extent_end_bytes > slot_bytes:
      raise ValueError(
          f"region {self.name} exceeds slot bytes: "
          f"end={self.extent_end_bytes} slot={slot_bytes}"
      )

  def to_dict(self) -> dict[str, int | str]:
    return dataclasses.asdict(self)

  def to_native_tuple(self) -> tuple:
    return (
        self.name,
        int(self.offset_bytes),
        int(self.stride_bytes),
        int(self.unit_bytes),
        int(self.num_units),
        int(self.units_per_stride),
    )


@dataclasses.dataclass(frozen=True)
class PoolSpec:
  tag: str
  storage_index: int
  base_offset_bytes: int
  block_stride_bytes: int
  num_blocks: int
  regions: tuple[RegionSpec, ...]
  dtype_tag: str = ""

  def validate(self, storage_bytes: int | None = None) -> None:
    if not self.tag:
      raise ValueError("pool tag must be non-empty")
    if self.storage_index < 0:
      raise ValueError(f"pool {self.tag} storage_index must be >= 0")
    if self.block_stride_bytes <= 0:
      raise ValueError(f"pool {self.tag} block_stride_bytes must be positive")
    if self.num_blocks <= 0:
      raise ValueError(f"pool {self.tag} num_blocks must be positive")
    if self.base_offset_bytes < 0:
      raise ValueError(f"pool {self.tag} base_offset_bytes must be >= 0")
    if not self.regions:
      raise ValueError(f"pool {self.tag} must contain at least one region")
    for region in self.regions:
      region.validate(self.block_stride_bytes)
    if storage_bytes is not None:
      end = self.base_offset_bytes + self.num_blocks * self.block_stride_bytes
      if end > storage_bytes:
        raise ValueError(
            f"pool {self.tag} exceeds storage bytes: "
            f"end={end} storage={storage_bytes}"
        )

  @property
  def live_bytes_per_block(self) -> int:
    return sum(region.live_bytes for region in self.regions)

  def to_dict(self) -> dict[str, object]:
    return {
        "tag": self.tag,
        "storage_index": self.storage_index,
        "base_offset_bytes": self.base_offset_bytes,
        "block_stride_bytes": self.block_stride_bytes,
        "num_blocks": self.num_blocks,
        "regions": [region.to_dict() for region in self.regions],
        "dtype_tag": self.dtype_tag,
    }

  def to_native_tuple(self) -> tuple:
    return (
        self.tag,
        int(self.storage_index),
        int(self.base_offset_bytes),
        int(self.block_stride_bytes),
        int(self.num_blocks),
        [region.to_native_tuple() for region in self.regions],
        self.dtype_tag,
    )


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


def coerce_pool_spec(value: PoolSpec | Mapping[str, object]) -> PoolSpec:
  if isinstance(value, PoolSpec):
    return value
  regions = value["regions"]
  if not isinstance(regions, Sequence):
    raise TypeError("pool regions must be a sequence")
  return PoolSpec(
      tag=str(value["tag"]),
      storage_index=int(value["storage_index"]),
      base_offset_bytes=int(value.get("base_offset_bytes", 0)),
      block_stride_bytes=int(value["block_stride_bytes"]),
      num_blocks=int(value["num_blocks"]),
      regions=tuple(coerce_region_spec(region) for region in regions),
      dtype_tag=str(value.get("dtype_tag", "")),
  )
