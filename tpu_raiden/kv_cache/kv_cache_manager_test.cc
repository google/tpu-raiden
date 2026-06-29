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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

// Helper subclass to pretend we are device-backed for validation tests
class TestKVCacheManager : public KVCacheManagerBase {
 public:
  TestKVCacheManager(size_t num_layers, size_t num_shards,
                     size_t slice_byte_size)
      : KVCacheManagerBase(num_layers, num_shards, slice_byte_size) {
    buffer_holds_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      buffer_holds_[l].resize(num_shards);
    }
  }
};

TEST(KVCacheManagerTest, CompilesAndLinksSuccessfully) {
  EXPECT_TRUE(true);
}

TEST(KVCacheManagerTest, D2hFailsWithMismatchedCopySpecLengths) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0, 1};  // Mismatch
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2h(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must have the same length"));
}

TEST(KVCacheManagerTest, H2dFailsWithNegativeOffsets) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {-1};  // Negative
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.H2d(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must be non-negative"));
}

TEST(KVCacheManagerTest, D2hFailsWithCpuOnlyManager) {
  // Use the base class directly to test CPU-only behavior
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2h(src_offsets, dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("requires a device-backed"));
}

TEST(KVCacheManagerTest, H2dFailsWithOutOfRangeLayer) {
  TestKVCacheManager manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<int64_t> src_offsets;
  std::vector<int64_t> dst_offsets;
  std::vector<int64_t> sizes;

  auto status = manager.H2d(src_offsets, dst_offsets, sizes,
                            /*slot_idx=*/std::nullopt,
                            /*layer_idx=*/1, /*shard_idx=*/0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("layer or shard index out of range"));
}

TEST(KVCacheManagerTest, H2hReadExplicitAcceptsParallelism) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  std::vector<uint8_t*> ptrs = {nullptr};
  auto status = manager.H2hReadExplicit("127.0.0.1:8080", {0}, {0}, ptrs,
                                        /*parallelism=*/2);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("Failed to connect to peer"));
}

// Subclass that sets per_layer_physical_size_ to simulate the device-backed
// constructor path where layers have different on-device buffer sizes.
class TestKVCacheManagerHeterogeneous : public KVCacheManagerBase {
 public:
  TestKVCacheManagerHeterogeneous(size_t num_layers, size_t num_shards,
                                  size_t slice_byte_size,
                                  std::optional<int> host_blocks,
                                  const std::vector<size_t>& per_layer_sizes)
      : KVCacheManagerBase(num_layers, num_shards, slice_byte_size,
                           /*local_port=*/std::nullopt, host_blocks) {
    per_layer_physical_size_ = per_layer_sizes;
    buffer_holds_.resize(num_layers);
    for (size_t l = 0; l < num_layers; ++l) {
      buffer_holds_[l].resize(num_shards);
    }
  }
};

TEST(KVCacheManagerPerLayerTest, BytesPerBlockReturnsMaxSliceSize) {
  // Simulate conv (small) and ssm (large) alternating, like a mamba group.
  // bytes_per_block() should return slice_byte_size_ (the max) since it's
  // used for host buffer allocation which must fit the largest layer.
  size_t num_blocks = 126;
  size_t conv_total = num_blocks * 48 * 1024;       // 48 KiB/block
  size_t ssm_total = num_blocks * 2 * 1024 * 1024;  // 2 MiB/block
  size_t max_slice = ssm_total / num_blocks;        // 2 MiB

  TestKVCacheManagerHeterogeneous manager(
      /*num_layers=*/4, /*num_shards=*/1,
      /*slice_byte_size=*/max_slice,
      /*host_blocks=*/8,
      /*per_layer_sizes=*/{conv_total, ssm_total, conv_total, ssm_total});

  EXPECT_EQ(manager.bytes_per_block(), max_slice);
}

TEST(KVCacheManagerPerLayerTest, EmptyPerLayerFallsBackToUniform) {
  // When per_layer_physical_size_ is empty (CPU-only path), all DMA code
  // falls back to using slice_byte_size_ uniformly.
  TestKVCacheManager manager(/*num_layers=*/3, /*num_shards=*/1,
                             /*slice_byte_size=*/256);

  EXPECT_EQ(manager.bytes_per_block(), 256u);
}

TEST(KVCacheManagerPerLayerTest, D2hDirectFailsWithLayerCountMismatch) {
  size_t num_blocks = 10;
  size_t conv_total = num_blocks * 100;
  size_t ssm_total = num_blocks * 1000;

  TestKVCacheManagerHeterogeneous manager(
      /*num_layers=*/2, /*num_shards=*/1,
      /*slice_byte_size=*/1000,
      /*host_blocks=*/static_cast<int>(num_blocks),
      /*per_layer_sizes=*/{conv_total, ssm_total});

  // Pass wrong number of device buffers (3 instead of 2).
  std::vector<uint8_t*> device_buffers = {nullptr, nullptr, nullptr};
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.D2hDirect(nullptr, device_buffers, src_offsets,
                                  dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("must match layer count"));
}

TEST(KVCacheManagerPerLayerTest, H2dDirectFailsWithLayerCountMismatch) {
  size_t num_blocks = 10;
  size_t conv_total = num_blocks * 100;
  size_t ssm_total = num_blocks * 1000;

  TestKVCacheManagerHeterogeneous manager(
      /*num_layers=*/2, /*num_shards=*/1,
      /*slice_byte_size=*/1000,
      /*host_blocks=*/static_cast<int>(num_blocks),
      /*per_layer_sizes=*/{conv_total, ssm_total});

  // Pass wrong number of device buffers (1 instead of 2).
  std::vector<uint8_t*> device_buffers = {nullptr};
  std::vector<int64_t> src_offsets = {0};
  std::vector<int64_t> dst_offsets = {0};
  std::vector<int64_t> sizes = {1};

  auto status = manager.H2dDirect(nullptr, device_buffers, src_offsets,
                                  dst_offsets, sizes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), testing::HasSubstr("must match layer count"));
}

TEST(KVCacheManagerPerLayerTest, UniformPerLayerMatchesOriginal) {
  // When all per-layer sizes are identical, behavior should match the
  // original uniform code path exactly.
  size_t slice = 512;
  size_t num_blocks = 10;
  size_t total = num_blocks * slice;

  TestKVCacheManagerHeterogeneous manager_heterogeneous(
      /*num_layers=*/3, /*num_shards=*/1,
      /*slice_byte_size=*/slice,
      /*host_blocks=*/static_cast<int>(num_blocks),
      /*per_layer_sizes=*/{total, total, total});

  TestKVCacheManager manager_uniform(
      /*num_layers=*/3, /*num_shards=*/1,
      /*slice_byte_size=*/slice);

  // Both should report the same bytes_per_block.
  EXPECT_EQ(manager_heterogeneous.bytes_per_block(),
            manager_uniform.bytes_per_block());
  EXPECT_EQ(manager_heterogeneous.num_layers(), manager_uniform.num_layers());
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
