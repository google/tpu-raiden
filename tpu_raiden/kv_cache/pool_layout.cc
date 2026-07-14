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

#include "tpu_raiden/kv_cache/pool_layout.h"

#include <algorithm>
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

bool RegionCoversOffset(const RegionSpec& region, size_t offset,
                        size_t* covered_end) {
  if (region.num_units <= 0) return false;
  const size_t region_offset = static_cast<size_t>(region.offset_bytes);
  const size_t stride = static_cast<size_t>(region.stride_bytes);
  const size_t packed_bytes = static_cast<size_t>(region.unit_bytes) *
                              static_cast<size_t>(region.units_per_stride);
  const size_t num_units = static_cast<size_t>(region.num_units);
  if (offset < region_offset || packed_bytes == 0) return false;

  size_t unit_idx = 0;
  if (num_units > 1) {
    if (stride == 0) return false;
    unit_idx = (offset - region_offset) / stride;
    if (unit_idx >= num_units) return false;
  }
  const size_t segment_start = region_offset + unit_idx * stride;
  const size_t segment_end = segment_start + packed_bytes;
  if (offset < segment_start || offset >= segment_end) return false;
  *covered_end = segment_end;
  return true;
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
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " units_per_stride must be positive"));
  }
  if (num_units > 0 && unit_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " unit_bytes must be positive"));
  }
  if (num_units > 1 && stride_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " stride_bytes must be positive"));
  }
  auto packed_bytes =
      CheckedMul(units_per_stride, unit_bytes,
                 absl::StrCat("region ", name, " packed unit bytes"));
  if (!packed_bytes.ok()) return packed_bytes.status();
  if (num_units > 0 && stride_bytes < *packed_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "region ", name, " stride_bytes is smaller than packed units"));
  }
  auto stride_extent =
      CheckedMul(num_units > 0 ? num_units - 1 : 0, stride_bytes,
                 absl::StrCat("region ", name, " stride extent"));
  if (!stride_extent.ok()) return stride_extent.status();
  auto extent_start =
      CheckedAdd(offset_bytes, *stride_extent,
                 absl::StrCat("region ", name, " extent start"));
  if (!extent_start.ok()) return extent_start.status();
  auto extent_end = CheckedAdd(*extent_start, *packed_bytes,
                               absl::StrCat("region ", name, " extent end"));
  if (!extent_end.ok()) return extent_end.status();
  if (*extent_end > slot_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat("region ", name, " exceeds slot bytes: end=", *extent_end,
                     " slot=", slot_bytes));
  }
  return absl::OkStatus();
}

absl::Status PoolSpec::Validate(int64_t storage_bytes) const {
  if (tag.empty()) {
    return absl::InvalidArgumentError("pool tag must be non-empty");
  }
  if (block_stride_bytes <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("pool ", tag, " block_stride_bytes must be positive"));
  }
  if (num_blocks <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("pool ", tag, " num_blocks must be positive"));
  }
  if (base_offset_bytes < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("pool ", tag, " base_offset_bytes must be >= 0"));
  }
  if (regions.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("pool ", tag, " must contain at least one region"));
  }
  for (const RegionSpec& region : regions) {
    absl::Status status = region.Validate(block_stride_bytes);
    if (!status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat("pool ", tag, ": ", status.message()));
    }
  }
  if (storage_bytes >= 0) {
    auto array_bytes = CheckedMul(num_blocks, block_stride_bytes,
                                  absl::StrCat("pool ", tag, " array bytes"));
    if (!array_bytes.ok()) return array_bytes.status();
    auto array_end = CheckedAdd(base_offset_bytes, *array_bytes,
                                absl::StrCat("pool ", tag, " array end"));
    if (!array_end.ok()) return array_end.status();
    if (*array_end > storage_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("pool ", tag, " exceeds storage bytes: end=", *array_end,
                       " storage=", storage_bytes));
    }
  }
  return absl::OkStatus();
}

bool RegionsCoverRange(const std::vector<RegionSpec>& regions, size_t start,
                       size_t end) {
  size_t cursor = start;
  while (cursor < end) {
    size_t next = cursor;
    for (const RegionSpec& region : regions) {
      size_t covered_end = 0;
      if (RegionCoversOffset(region, cursor, &covered_end)) {
        next = std::max(next, std::min(covered_end, end));
      }
    }
    if (next == cursor) {
      return false;
    }
    cursor = next;
  }
  return true;
}

absl::StatusOr<std::vector<PoolBlockCopyExtent>> ComputePoolBlockCopyExtents(
    const PoolSpec& pool, absl::Span<const int64_t> block_ids) {
  std::vector<PoolBlockCopyExtent> extents;
  extents.reserve(block_ids.size());
  for (int64_t block_id : block_ids) {
    if (block_id < 0 || block_id >= pool.num_blocks) {
      return absl::OutOfRangeError(absl::StrCat("pool ", pool.tag, " block_id ",
                                                block_id, " out of range [0, ",
                                                pool.num_blocks, ")"));
    }
    const int64_t offset =
        pool.base_offset_bytes + block_id * pool.block_stride_bytes;
    if (!extents.empty() &&
        extents.back().offset_bytes + extents.back().size_bytes == offset) {
      extents.back().size_bytes += pool.block_stride_bytes;
    } else {
      extents.push_back(PoolBlockCopyExtent{
          .offset_bytes = offset,
          .size_bytes = pool.block_stride_bytes,
      });
    }
  }
  return extents;
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

tpu_raiden::rpc::PoolSpecProto ToProto(const PoolSpec& pool) {
  tpu_raiden::rpc::PoolSpecProto proto;
  proto.set_tag(pool.tag);
  proto.set_storage_index(static_cast<int64_t>(pool.storage_index));
  proto.set_base_offset_bytes(pool.base_offset_bytes);
  proto.set_block_stride_bytes(pool.block_stride_bytes);
  proto.set_num_blocks(pool.num_blocks);
  for (const RegionSpec& region : pool.regions) {
    *proto.add_regions() = ToProto(region);
  }
  proto.set_dtype_tag(pool.dtype_tag);
  return proto;
}

absl::StatusOr<PoolSpec> PoolSpecFromProto(
    const tpu_raiden::rpc::PoolSpecProto& proto) {
  if (proto.storage_index() < 0) {
    return absl::InvalidArgumentError("storage_index must be non-negative");
  }
  PoolSpec pool;
  pool.tag = proto.tag();
  pool.storage_index = static_cast<size_t>(proto.storage_index());
  pool.base_offset_bytes = proto.base_offset_bytes();
  pool.block_stride_bytes = proto.block_stride_bytes();
  pool.num_blocks = proto.num_blocks();
  pool.regions.reserve(proto.regions_size());
  for (const auto& region_proto : proto.regions()) {
    absl::StatusOr<RegionSpec> region = RegionSpecFromProto(region_proto);
    if (!region.ok()) return region.status();
    pool.regions.push_back(*region);
  }
  pool.dtype_tag = proto.dtype_tag();
  return pool;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
