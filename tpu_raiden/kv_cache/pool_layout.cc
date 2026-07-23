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
#include <vector>

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
    const int64_t live_end = storage_extent_end_bytes();
    if (live_end == std::numeric_limits<int64_t>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("pool ", tag, " live storage extent overflows int64"));
    }
    if (live_end > storage_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "pool ", tag, " exceeds storage bytes (live regions): end=", live_end,
          " storage=", storage_bytes));
    }
  }
  return absl::OkStatus();
}

int64_t PoolSpec::storage_extent_end_bytes() const {
  if (base_offset_bytes < 0 || block_stride_bytes < 0 || num_blocks <= 0) {
    return std::numeric_limits<int64_t>::max();
  }
  int64_t block_offset = 0;
  if (MulWillOverflow(num_blocks - 1, block_stride_bytes)) {
    return std::numeric_limits<int64_t>::max();
  }
  block_offset = (num_blocks - 1) * block_stride_bytes;
  int64_t block_live_end = 0;
  for (const RegionSpec& region : regions) {
    block_live_end = std::max(block_live_end, region.extent_end_bytes());
  }
  if (base_offset_bytes > std::numeric_limits<int64_t>::max() - block_offset) {
    return std::numeric_limits<int64_t>::max();
  }
  const int64_t last_block = base_offset_bytes + block_offset;
  if (block_live_end < 0 ||
      last_block > std::numeric_limits<int64_t>::max() - block_live_end) {
    return std::numeric_limits<int64_t>::max();
  }
  return last_block + block_live_end;
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
  for (int64_t block_id : block_ids) {
    if (block_id < 0 || block_id >= pool.num_blocks) {
      return absl::OutOfRangeError(absl::StrCat("pool ", pool.tag, " block_id ",
                                                block_id, " out of range [0, ",
                                                pool.num_blocks, ")"));
    }
    auto block_delta_or =
        CheckedMul(block_id, pool.block_stride_bytes,
                   absl::StrCat("pool ", pool.tag, " block offset"));
    if (!block_delta_or.ok()) return block_delta_or.status();
    const int64_t block_delta = *block_delta_or;
    auto block_base_or =
        CheckedAdd(pool.base_offset_bytes, block_delta,
                   absl::StrCat("pool ", pool.tag, " block base"));
    if (!block_base_or.ok()) return block_base_or.status();
    const int64_t block_base = *block_base_or;
    for (const RegionSpec& region : pool.regions) {
      auto packed_bytes_or =
          CheckedMul(region.unit_bytes, region.units_per_stride,
                     absl::StrCat("pool ", pool.tag, " region ", region.name,
                                  " packed bytes"));
      if (!packed_bytes_or.ok()) return packed_bytes_or.status();
      const int64_t packed_bytes = *packed_bytes_or;
      if (region.num_units == 0 || packed_bytes == 0) continue;

      // A tightly packed strided region is one contiguous DMA extent. Sparse
      // regions retain one extent per unit so padding is never staged.
      if (region.stride_bytes == packed_bytes) {
        auto region_bytes_or =
            CheckedMul(region.num_units, packed_bytes,
                       absl::StrCat("pool ", pool.tag, " region ", region.name,
                                    " live bytes"));
        if (!region_bytes_or.ok()) return region_bytes_or.status();
        const int64_t region_bytes = *region_bytes_or;
        auto offset_or = CheckedAdd(block_base, region.offset_bytes,
                                    absl::StrCat("pool ", pool.tag, " region ",
                                                 region.name, " offset"));
        if (!offset_or.ok()) return offset_or.status();
        const int64_t offset = *offset_or;
        extents.push_back({.offset_bytes = offset, .size_bytes = region_bytes});
        continue;
      }
      for (int64_t unit = 0; unit < region.num_units; ++unit) {
        auto unit_delta_or =
            CheckedMul(unit, region.stride_bytes,
                       absl::StrCat("pool ", pool.tag, " region ", region.name,
                                    " unit offset"));
        if (!unit_delta_or.ok()) return unit_delta_or.status();
        const int64_t unit_delta = *unit_delta_or;
        auto region_base_or = CheckedAdd(
            block_base, region.offset_bytes,
            absl::StrCat("pool ", pool.tag, " region ", region.name, " base"));
        if (!region_base_or.ok()) return region_base_or.status();
        const int64_t region_base = *region_base_or;
        auto offset_or = CheckedAdd(region_base, unit_delta,
                                    absl::StrCat("pool ", pool.tag, " region ",
                                                 region.name, " unit address"));
        if (!offset_or.ok()) return offset_or.status();
        const int64_t offset = *offset_or;
        extents.push_back({.offset_bytes = offset, .size_bytes = packed_bytes});
      }
    }
  }

  std::sort(extents.begin(), extents.end(),
            [](const PoolBlockCopyExtent& lhs, const PoolBlockCopyExtent& rhs) {
              if (lhs.offset_bytes != rhs.offset_bytes) {
                return lhs.offset_bytes < rhs.offset_bytes;
              }
              return lhs.size_bytes < rhs.size_bytes;
            });
  std::vector<PoolBlockCopyExtent> merged;
  merged.reserve(extents.size());
  for (const PoolBlockCopyExtent& extent : extents) {
    auto extent_end_or =
        CheckedAdd(extent.offset_bytes, extent.size_bytes,
                   absl::StrCat("pool ", pool.tag, " copy extent end"));
    if (!extent_end_or.ok()) return extent_end_or.status();
    const int64_t extent_end = *extent_end_or;
    if (merged.empty()) {
      merged.push_back(extent);
      continue;
    }
    PoolBlockCopyExtent& previous = merged.back();
    auto previous_end_or = CheckedAdd(
        previous.offset_bytes, previous.size_bytes,
        absl::StrCat("pool ", pool.tag, " previous copy extent end"));
    if (!previous_end_or.ok()) return previous_end_or.status();
    const int64_t previous_end = *previous_end_or;
    if (extent.offset_bytes > previous_end) {
      merged.push_back(extent);
      continue;
    }
    previous.size_bytes =
        std::max(previous_end, extent_end) - previous.offset_bytes;
  }
  return merged;
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
