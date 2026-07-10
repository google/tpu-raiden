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

#ifndef THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_HYBRID_LAYOUT_NANOBIND_H_
#define THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_HYBRID_LAYOUT_NANOBIND_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"
#include "tpu_raiden/kv_cache/hybrid_block_layout.h"

namespace tpu_raiden {
namespace torch_bindings {

namespace nb = nanobind;

using RegionTuple =
    std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>;
using LayoutTuple = std::tuple<int, int64_t, std::vector<RegionTuple>>;

inline void ThrowIfNotOk(const absl::Status& status,
                         const std::string& context) {
  if (!status.ok()) {
    throw std::runtime_error(absl::StrCat(context, ": ", status.message()));
  }
}

inline kv_cache::LayerKind LayerKindFromId(int kind_id) {
  switch (kind_id) {
    case 0:
      return kv_cache::LayerKind::kFullAttention;
    case 1:
      return kv_cache::LayerKind::kMambaState;
    case 2:
      return kv_cache::LayerKind::kOpaque;
    default:
      throw std::invalid_argument(
          absl::StrCat("unknown layer kind id: ", kind_id));
  }
}

inline const char* LayerKindName(kv_cache::LayerKind kind) {
  switch (kind) {
    case kv_cache::LayerKind::kFullAttention:
      return "full_attention";
    case kv_cache::LayerKind::kMambaState:
      return "mamba_state";
    case kv_cache::LayerKind::kOpaque:
      return "opaque";
  }
  return "opaque";
}

inline std::vector<kv_cache::LayerBlockLayout> LayoutsFromTuples(
    const std::vector<LayoutTuple>& entries) {
  std::vector<kv_cache::LayerBlockLayout> layouts;
  layouts.reserve(entries.size());
  for (const auto& [kind_id, slot_bytes, region_entries] : entries) {
    kv_cache::LayerBlockLayout layout;
    layout.kind = LayerKindFromId(kind_id);
    layout.slot_bytes = slot_bytes;
    layout.regions.reserve(region_entries.size());
    for (const auto& [name, offset_bytes, stride_bytes, unit_bytes, num_units,
                      units_per_stride] : region_entries) {
      layout.regions.push_back(kv_cache::RegionSpec{
          .name = name,
          .offset_bytes = offset_bytes,
          .stride_bytes = stride_bytes,
          .unit_bytes = unit_bytes,
          .num_units = num_units,
          .units_per_stride = units_per_stride,
      });
    }
    layouts.push_back(std::move(layout));
  }
  return layouts;
}

inline nb::dict RegionToDict(const kv_cache::RegionSpec& region) {
  nb::dict result;
  result["name"] = region.name;
  result["offset_bytes"] = region.offset_bytes;
  result["stride_bytes"] = region.stride_bytes;
  result["unit_bytes"] = region.unit_bytes;
  result["num_units"] = region.num_units;
  result["units_per_stride"] = region.units_per_stride;
  return result;
}

inline nb::dict HybridBlockRefToDict(const kv_cache::HybridBlockRef& ref) {
  nb::dict result;
  result["ptr"] = reinterpret_cast<uintptr_t>(ref.ptr);
  result["slot_bytes"] = ref.slot_bytes;
  result["kind"] = LayerKindName(ref.layout->kind);
  result["layer_idx"] = ref.layer_idx;
  result["shard_idx"] = ref.shard_idx;
  result["block_id"] = ref.block_id;
  nb::list regions;
  for (const kv_cache::RegionSpec& region : ref.layout->regions) {
    regions.append(RegionToDict(region));
  }
  result["regions"] = regions;
  return result;
}

}  // namespace torch_bindings
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_FRAMEWORKS_TORCH_HYBRID_LAYOUT_NANOBIND_H_
