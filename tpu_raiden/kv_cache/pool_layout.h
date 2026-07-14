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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_POOL_LAYOUT_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_POOL_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

// A strided run of live bytes inside one pool block. Byte-level and
// model-agnostic: callers describe interior layout, raiden only stores and
// checks it.
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

// One block pool inside a wrapped storage. A storage is a whole wrapped
// allocation (constructor order gives storage_index); a pool is an array of
// num_blocks equally strided blocks starting at base_offset_bytes, each
// carrying the same regions of live bytes. tag and dtype_tag are opaque:
// raiden stores, filters, and echoes them but never parses them.
struct PoolSpec {
  std::string tag;
  size_t storage_index = 0;
  int64_t base_offset_bytes = 0;
  int64_t block_stride_bytes = 0;
  int64_t num_blocks = 0;
  std::vector<RegionSpec> regions;
  std::string dtype_tag;

  // Validates internal consistency and, when storage_bytes >= 0, that the
  // whole block array fits inside the storage.
  absl::Status Validate(int64_t storage_bytes) const;
};

// Host-side reference to one block of one pool.
struct PoolBlockRef {
  uint8_t* ptr = nullptr;
  int64_t block_stride_bytes = 0;
  const PoolSpec* pool = nullptr;
  size_t pool_idx = 0;
  size_t shard_idx = 0;
  int64_t block_id = 0;
};

// True when regions jointly cover every byte of [start, end) within a block.
bool RegionsCoverRange(const std::vector<RegionSpec>& regions, size_t start,
                       size_t end);

// One contiguous byte range inside a pool's storage. Host mirrors share the
// storage layout, so the same offset addresses both sides of a D2H/H2D copy.
struct PoolBlockCopyExtent {
  int64_t offset_bytes = 0;
  int64_t size_bytes = 0;
};

// Whole-block copy extents for the given block ids of one pool, with runs of
// consecutive ids coalesced into single extents. Rejects out-of-range ids.
absl::StatusOr<std::vector<PoolBlockCopyExtent>> ComputePoolBlockCopyExtents(
    const PoolSpec& pool, absl::Span<const int64_t> block_ids);

tpu_raiden::rpc::RegionSpecProto ToProto(const RegionSpec& region);
absl::StatusOr<RegionSpec> RegionSpecFromProto(
    const tpu_raiden::rpc::RegionSpecProto& proto);

tpu_raiden::rpc::PoolSpecProto ToProto(const PoolSpec& pool);
absl::StatusOr<PoolSpec> PoolSpecFromProto(
    const tpu_raiden::rpc::PoolSpecProto& proto);

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_POOL_LAYOUT_H_
