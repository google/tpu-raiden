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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

RegionSpec ValidRegion() {
  return RegionSpec{
      .name = "key",
      .offset_bytes = 0,
      .stride_bytes = 100,
      .unit_bytes = 10,
      .num_units = 5,
      .units_per_stride = 2,
  };
}

PoolSpec ValidPool() {
  return PoolSpec{
      .tag = "pool0",
      .storage_index = 0,
      .base_offset_bytes = 0,
      .block_stride_bytes = 1000,
      .num_blocks = 10,
      .regions = {ValidRegion()},
  };
}

TEST(RegionSpecTest, LiveBytes) {
  RegionSpec region = ValidRegion();
  EXPECT_EQ(region.live_bytes(), 100);
}

TEST(RegionSpecTest, ExtentEndBytes) {
  RegionSpec region = ValidRegion();
  // offset 0 + (5 - 1) * 100 + 2 * 10 = 420
  EXPECT_EQ(region.extent_end_bytes(), 420);
}

TEST(RegionSpecTest, ValidateOk) {
  RegionSpec region = ValidRegion();
  ABSL_EXPECT_OK(region.Validate(1000));
}

TEST(RegionSpecTest, ValidateExceedsSlot) {
  RegionSpec region = ValidRegion();
  EXPECT_THAT(region.Validate(400), StatusIs(absl::StatusCode::kInvalidArgument,
                                             HasSubstr("exceeds slot bytes")));
}

TEST(RegionSpecTest, ValidateNegativeOffset) {
  RegionSpec region = ValidRegion();
  region.offset_bytes = -1;
  EXPECT_THAT(region.Validate(1000),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PoolSpecTest, ValidateOk) {
  PoolSpec pool = ValidPool();
  ABSL_EXPECT_OK(pool.Validate(10000));
}

TEST(PoolSpecTest, ValidateExceedsStorage) {
  PoolSpec pool = ValidPool();
  // The final live region ends at 9 * 1000 + 420. Trailing padding is not
  // addressable and need not exist in the backing allocation.
  EXPECT_EQ(pool.storage_extent_end_bytes(), 9420);
  ABSL_EXPECT_OK(pool.Validate(9420));
  EXPECT_THAT(pool.Validate(9419),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds storage bytes")));
}

TEST(PoolSpecTest, AdmitsGdnSsmWhenLastLiveByteFitsSharedRawStorage) {
  // The logical trailing stride ends at 67,174,400, but the final 2 MiB live
  // payload ends at 65,077,248 and fits in the shared 64 MiB raw allocation.
  PoolSpec pool{
      .tag = "gdn.ssm.g0",
      .storage_index = 0,
      .base_offset_bytes = 65536,
      .block_stride_bytes = 4194304,
      .num_blocks = 16,
      .regions = {RegionSpec{
          .name = "ssm",
          .offset_bytes = 0,
          .stride_bytes = 2097152,
          .unit_bytes = 2097152,
          .num_units = 1,
          .units_per_stride = 1,
      }},
  };

  EXPECT_EQ(pool.storage_extent_end_bytes(), 65077248);
  ABSL_EXPECT_OK(pool.Validate(/*storage_bytes=*/67108864));
}

TEST(PoolSpecTest, ProtoRoundTrip) {
  PoolSpec pool = ValidPool();
  tpu_raiden::rpc::PoolSpecProto proto = ToProto(pool);
  auto roundtrip = PoolSpecFromProto(proto);
  ABSL_ASSERT_OK(roundtrip);
  EXPECT_EQ(roundtrip->tag, pool.tag);
  EXPECT_EQ(roundtrip->storage_index, pool.storage_index);
  EXPECT_EQ(roundtrip->base_offset_bytes, pool.base_offset_bytes);
  EXPECT_EQ(roundtrip->block_stride_bytes, pool.block_stride_bytes);
  EXPECT_EQ(roundtrip->num_blocks, pool.num_blocks);
  ASSERT_EQ(roundtrip->regions.size(), pool.regions.size());
  EXPECT_EQ(roundtrip->regions[0].name, pool.regions[0].name);
}

TEST(RegionsCoverRangeTest, Basic) {
  RegionSpec r1{
      .name = "r1",
      .offset_bytes = 0,
      .stride_bytes = 20,
      .unit_bytes = 10,
      .num_units = 2,
      .units_per_stride = 1,
  };
  RegionSpec r2{
      .name = "r2",
      .offset_bytes = 10,
      .stride_bytes = 20,
      .unit_bytes = 10,
      .num_units = 2,
      .units_per_stride = 1,
  };

  // r1 covers: [0, 10), [20, 30)
  // r2 covers: [10, 20), [30, 40)
  // Together they cover [0, 40).
  EXPECT_TRUE(RegionsCoverRange({r1, r2}, 0, 40));
  EXPECT_FALSE(RegionsCoverRange({r1, r2}, 0, 41));
}

TEST(ComputePoolBlockCopyExtentsTest, Basic) {
  PoolSpec pool = ValidPool();
  pool.regions = {RegionSpec{
      .name = "payload",
      .offset_bytes = 0,
      .stride_bytes = 1000,
      .unit_bytes = 1000,
      .num_units = 1,
      .units_per_stride = 1,
  }};
  auto extents_or = ComputePoolBlockCopyExtents(pool, {0, 1, 3, 4, 5});
  ABSL_ASSERT_OK(extents_or);
  const auto& extents = *extents_or;
  ASSERT_EQ(extents.size(), 2);
  EXPECT_EQ(extents[0].offset_bytes, 0);
  EXPECT_EQ(extents[0].size_bytes, 2000);  // block 0 and 1 coalesced
  EXPECT_EQ(extents[1].offset_bytes, 3000);
  EXPECT_EQ(extents[1].size_bytes, 3000);  // block 3, 4, 5 coalesced
}

TEST(ComputePoolBlockCopyExtentsTest, OmitsInterRegionAndTrailingPadding) {
  PoolSpec pool = ValidPool();
  pool.base_offset_bytes = 8;
  pool.block_stride_bytes = 128;
  pool.num_blocks = 2;
  pool.regions = {RegionSpec{
      .name = "payload",
      .offset_bytes = 0,
      .stride_bytes = 32,
      .unit_bytes = 16,
      .num_units = 2,
      .units_per_stride = 1,
  }};

  ABSL_ASSERT_OK(pool.Validate(/*storage_bytes=*/184));
  auto extents_or = ComputePoolBlockCopyExtents(pool, {0, 1});
  ABSL_ASSERT_OK(extents_or);
  ASSERT_EQ(extents_or->size(), 4);
  EXPECT_EQ((*extents_or)[0].offset_bytes, 8);
  EXPECT_EQ((*extents_or)[0].size_bytes, 16);
  EXPECT_EQ((*extents_or)[1].offset_bytes, 40);
  EXPECT_EQ((*extents_or)[1].size_bytes, 16);
  EXPECT_EQ((*extents_or)[2].offset_bytes, 136);
  EXPECT_EQ((*extents_or)[2].size_bytes, 16);
  EXPECT_EQ((*extents_or)[3].offset_bytes, 168);
  EXPECT_EQ((*extents_or)[3].size_bytes, 16);
}

TEST(ComputePoolBlockCopyExtentsTest,
     CoalescesTokenStridesButPreservesAliasedFaRegionGaps) {
  PoolSpec pool{
      .tag = "fa",
      .storage_index = 0,
      .base_offset_bytes = 0,
      .block_stride_bytes = 2162688,
      .num_blocks = 1,
      .regions = {},
  };
  for (int64_t head_group = 0; head_group < 4; ++head_group) {
    pool.regions.push_back(RegionSpec{
        .name = "fa",
        .offset_bytes = head_group * 540672,
        .stride_bytes = 1024,
        .unit_bytes = 512,
        .num_units = 256,
        .units_per_stride = 2,
    });
  }

  ABSL_ASSERT_OK(pool.Validate(/*storage_bytes=*/1884160));
  auto extents_or = ComputePoolBlockCopyExtents(pool, {0});
  ABSL_ASSERT_OK(extents_or);
  ASSERT_EQ(extents_or->size(), 4);
  for (int64_t head_group = 0; head_group < 4; ++head_group) {
    EXPECT_EQ((*extents_or)[head_group].offset_bytes, head_group * 540672);
    EXPECT_EQ((*extents_or)[head_group].size_bytes, 262144);
  }
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
