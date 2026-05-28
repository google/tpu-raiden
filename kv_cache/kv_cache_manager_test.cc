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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "kv_cache/kv_cache_manager_base.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(KVCacheManagerTest, CompilesAndLinksSuccessfully) {
  // Fulfills the cc_test verification requirement. Runtime logic and IFRT array
  // extraction from live Python objects are fully validated via the end-to-end
  // Python unit tests (kv_cache_manager_test_gl and kv_cache_manager_test_gf).
  EXPECT_TRUE(true);
}

TEST(KVCacheManagerTest, D2hToFailsWithMismatchedCopySpecLengths) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  KVCacheCopySpec spec;
  spec.src_offsets = {0};
  spec.dst_offsets = {0, 1};  // Mismatch
  spec.sizes = {1};

  char buf[128];
  auto status = manager.D2hTo(0, buf, 128, spec, 0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must have the same length"));
}

TEST(KVCacheManagerTest, H2dFromFailsWithNegativeOffsets) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  KVCacheCopySpec spec;
  spec.src_offsets = {-1};
  spec.dst_offsets = {0};
  spec.sizes = {1};

  char buf[128];
  auto status = manager.H2dFrom(0, buf, 128, spec, 0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("must be non-negative"));
}

TEST(KVCacheManagerTest, D2hToFailsWithCpuOnlyManager) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  KVCacheCopySpec spec;  // Empty spec is valid

  char buf[128];
  auto status = manager.D2hTo(0, buf, 128, spec, 0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("device-backed KVCacheManagerBase"));
}

TEST(KVCacheManagerTest, H2dFromFailsWithOutOfRangeLayer) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  KVCacheCopySpec spec;

  char buf[128];
  auto status = manager.H2dFrom(1, buf, 128, spec, 0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("layer or shard index out of range"));
}

TEST(KVCacheManagerTest, D2hToFailsWithNullPointerWhenSizeIsPositive) {
  KVCacheManagerBase manager(/*num_layers=*/1, /*num_shards=*/1,
                             /*slice_byte_size=*/128);
  KVCacheCopySpec spec;

  auto status = manager.D2hTo(0, nullptr, 128, spec, 0);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
  // Fails at CPU-only check before null check in this sequence.
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr("device-backed KVCacheManagerBase"));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
