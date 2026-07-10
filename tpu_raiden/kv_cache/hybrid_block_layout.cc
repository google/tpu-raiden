// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_raiden/kv_cache/hybrid_block_layout.h"

#include <cstdint>
#include <limits>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

bool MulWillOverflow(int64_t lhs, int64_t rhs) {
  if (lhs == 0 || rhs == 0) return false;
  return lhs > std::numeric_limits<int64_t>::max() / rhs;
}

absl::StatusOr<int64_t> CheckedMul(int64_t lhs, int64_t rhs,
                                   const std::string& field_name) {
  if (lhs < 0 || rhs < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(field_name, " operands must be non-negative"));
  }
  if (MulWillOverflow(lhs, rhs)) {
    return absl::InvalidArgumentError(
        absl::StrCat(field_name, " overflows int64"));
  }
  return lhs * rhs;
}

absl::StatusOr<int64_t> CheckedAdd(int64_t lhs, int64_t rhs,
                                   const std::string& field_name) {
  if (lhs < 0 || rhs < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(field_name, " operands must be non-negative"));
  }
  if (lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return absl::InvalidArgumentError(
        absl::StrCat(field_name, " overflows int64"));
  }
  return lhs + rhs;
}

}  // namespace

int64_t RegionSpec::live_bytes() const {
  if (MulWillOverflow(unit_bytes, num_units)) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t unit_total = unit_bytes * num_units;
  if (MulWillOverflow(unit_total, units_per_stride)) {
    return std::numeric_limits<int64_t>::max();
  }
  return unit_total * units_per_stride;
}

int64_t RegionSpec::extent_end_bytes() const {
  if (num_units == 0) {
    return offset_bytes;
  }
  if (num_units < 0 || stride_bytes < 0 || unit_bytes < 0 ||
      units_per_stride < 0 || offset_bytes < 0) {
    return std::numeric_limits<int64_t>::max();
  }
  if (MulWillOverflow(num_units - 1, stride_bytes)) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t strided_offset = (num_units - 1) * stride_bytes;
  if (MulWillOverflow(units_per_stride, unit_bytes)) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t packed_bytes = units_per_stride * unit_bytes;
  if (offset_bytes > std::numeric_limits<int64_t>::max() - strided_offset) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t start = offset_bytes + strided_offset;
  if (start > std::numeric_limits<int64_t>::max() - packed_bytes) {
    return std::numeric_limits<int64_t>::max();
  }
  return start + packed_bytes;
}

absl::Status RegionSpec::Validate(int64_t slot_bytes) const {
  if (name.empty()) {
    return absl::InvalidArgumentError("region name must be non-empty");
  }
  if (slot_bytes <= 0) {
    return absl::InvalidArgumentError("slot_bytes must be positive");
  }
  if (offset_bytes < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " offset_bytes must be >= 0"));
  }
  if (stride_bytes < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " stride_bytes must be >= 0"));
  }
  if (unit_bytes < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " unit_bytes must be >= 0"));
  }
  if (num_units < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " num_units must be >= 0"));
  }
  if (units_per_stride <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "region ", name, " units_per_stride must be positive"));
  }
  if (num_units > 0 && unit_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " unit_bytes must be positive"));
  }
  if (num_units > 1 && stride_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " stride_bytes must be positive"));
  }
  auto packed_bytes = CheckedMul(units_per_stride, unit_bytes,
                                 absl::StrCat("region ", name,
                                              " packed unit bytes"));
  if (!packed_bytes.ok()) return packed_bytes.status();
  if (num_units > 0 && stride_bytes < *packed_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "region ", name, " stride_bytes is smaller than packed units"));
  }
  auto stride_extent = CheckedMul(num_units > 0 ? num_units - 1 : 0,
                                  stride_bytes,
                                  absl::StrCat("region ", name,
                                               " stride extent"));
  if (!stride_extent.ok()) return stride_extent.status();
  auto extent_start = CheckedAdd(offset_bytes, *stride_extent,
                                 absl::StrCat("region ", name,
                                              " extent start"));
  if (!extent_start.ok()) return extent_start.status();
  auto extent_end = CheckedAdd(*extent_start, *packed_bytes,
                               absl::StrCat("region ", name, " extent end"));
  if (!extent_end.ok()) return extent_end.status();
  if (*extent_end > slot_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "region ", name, " exceeds slot bytes: end=", *extent_end,
        " slot=", slot_bytes));
  }
  return absl::OkStatus();
}

absl::Status LayerBlockLayout::Validate(int64_t manager_slot_bytes) const {
  if (slot_bytes <= 0) {
    return absl::InvalidArgumentError("layout slot_bytes must be positive");
  }
  if (manager_slot_bytes <= 0) {
    return absl::InvalidArgumentError(
        "manager slot byte size must be positive");
  }
  const bool compact_mamba =
      kind == LayerKind::kMambaState && manager_slot_bytes <= slot_bytes;
  if (slot_bytes != manager_slot_bytes && !compact_mamba) {
    return absl::InvalidArgumentError(absl::StrCat(
        "layout slot_bytes must match manager slot byte size: layout=",
        slot_bytes, " manager=", manager_slot_bytes));
  }
  if (regions.empty()) {
    return absl::InvalidArgumentError("layout must contain at least one region");
  }
  for (const RegionSpec& region : regions) {
    absl::Status status = region.Validate(slot_bytes);
    if (!status.ok()) return status;
  }
  if (slot_bytes != manager_slot_bytes && compact_mamba) {
    for (const RegionSpec& region : regions) {
      absl::Status status = region.Validate(manager_slot_bytes);
      if (!status.ok()) return status;
    }
  }
  return absl::OkStatus();
}

tpu_raiden::rpc::LayerKindProto ToProto(LayerKind kind) {
  switch (kind) {
    case LayerKind::kFullAttention:
      return tpu_raiden::rpc::LAYER_KIND_FULL_ATTENTION;
    case LayerKind::kMambaState:
      return tpu_raiden::rpc::LAYER_KIND_MAMBA_STATE;
    case LayerKind::kOpaque:
      return tpu_raiden::rpc::LAYER_KIND_OPAQUE;
  }
  return tpu_raiden::rpc::LAYER_KIND_UNSPECIFIED;
}

absl::StatusOr<LayerKind> LayerKindFromProto(
    tpu_raiden::rpc::LayerKindProto kind) {
  switch (kind) {
    case tpu_raiden::rpc::LAYER_KIND_FULL_ATTENTION:
      return LayerKind::kFullAttention;
    case tpu_raiden::rpc::LAYER_KIND_MAMBA_STATE:
      return LayerKind::kMambaState;
    case tpu_raiden::rpc::LAYER_KIND_OPAQUE:
      return LayerKind::kOpaque;
    case tpu_raiden::rpc::LAYER_KIND_UNSPECIFIED:
      return absl::InvalidArgumentError("layer kind must be specified");
  }
  return absl::InvalidArgumentError("unknown layer kind proto value");
}

tpu_raiden::rpc::RegionSpecProto ToProto(const RegionSpec& region) {
  tpu_raiden::rpc::RegionSpecProto proto;
  proto.set_name(region.name);
  proto.set_offset_bytes(region.offset_bytes);
  proto.set_stride_bytes(region.stride_bytes);
  proto.set_unit_bytes(region.unit_bytes);
  proto.set_num_units(region.num_units);
  proto.set_units_per_stride(region.units_per_stride);
  return proto;
}

absl::StatusOr<RegionSpec> RegionSpecFromProto(
    const tpu_raiden::rpc::RegionSpecProto& proto) {
  return RegionSpec{
      .name = proto.name(),
      .offset_bytes = proto.offset_bytes(),
      .stride_bytes = proto.stride_bytes(),
      .unit_bytes = proto.unit_bytes(),
      .num_units = proto.num_units(),
      .units_per_stride = proto.units_per_stride(),
  };
}

tpu_raiden::rpc::LayerBlockLayoutProto ToProto(
    const LayerBlockLayout& layout) {
  tpu_raiden::rpc::LayerBlockLayoutProto proto;
  proto.set_kind(ToProto(layout.kind));
  proto.set_slot_bytes(layout.slot_bytes);
  for (const RegionSpec& region : layout.regions) {
    *proto.add_regions() = ToProto(region);
  }
  return proto;
}

absl::StatusOr<LayerBlockLayout> LayerBlockLayoutFromProto(
    const tpu_raiden::rpc::LayerBlockLayoutProto& proto) {
  absl::StatusOr<LayerKind> kind = LayerKindFromProto(proto.kind());
  if (!kind.ok()) return kind.status();
  LayerBlockLayout layout;
  layout.kind = *kind;
  layout.slot_bytes = proto.slot_bytes();
  layout.regions.reserve(proto.regions_size());
  for (const auto& region_proto : proto.regions()) {
    absl::StatusOr<RegionSpec> region = RegionSpecFromProto(region_proto);
    if (!region.ok()) return region.status();
    layout.regions.push_back(*region);
  }
  return layout;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
