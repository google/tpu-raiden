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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_HYBRID_BLOCK_LAYOUT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_HYBRID_BLOCK_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

enum class LayerKind : uint8_t {
  kFullAttention = 0,
  kMambaState = 1,
  kOpaque = 2,
};

struct RegionSpec {
  std::string name;
  int64_t offset_bytes = 0;
  int64_t stride_bytes = 0;
  int64_t unit_bytes = 0;
  int64_t num_units = 0;
  int64_t units_per_stride = 1;

  int64_t live_bytes() const;
  int64_t extent_end_bytes() const;
  absl::Status Validate(int64_t slot_bytes) const;
};

struct LayerBlockLayout {
  LayerKind kind = LayerKind::kOpaque;
  int64_t slot_bytes = 0;
  std::vector<RegionSpec> regions;

  absl::Status Validate(int64_t manager_slot_bytes) const;
};

struct HybridBlockRef {
  uint8_t* ptr = nullptr;
  int64_t slot_bytes = 0;
  const LayerBlockLayout* layout = nullptr;
  size_t layer_idx = 0;
  size_t shard_idx = 0;
  int64_t block_id = 0;
};

tpu_raiden::rpc::LayerKindProto ToProto(LayerKind kind);
absl::StatusOr<LayerKind> LayerKindFromProto(
    tpu_raiden::rpc::LayerKindProto kind);

tpu_raiden::rpc::RegionSpecProto ToProto(const RegionSpec& region);
absl::StatusOr<RegionSpec> RegionSpecFromProto(
    const tpu_raiden::rpc::RegionSpecProto& proto);

tpu_raiden::rpc::LayerBlockLayoutProto ToProto(
    const LayerBlockLayout& layout);
absl::StatusOr<LayerBlockLayout> LayerBlockLayoutFromProto(
    const tpu_raiden::rpc::LayerBlockLayoutProto& proto);

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_HYBRID_BLOCK_LAYOUT_H_
