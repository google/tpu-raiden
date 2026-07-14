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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

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
  EXPECT_OK(region.Validate(1000));
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
  EXPECT_OK(pool.Validate(10000));
}

TEST(PoolSpecTest, ValidateExceedsStorage) {
  PoolSpec pool = ValidPool();
  // 10 blocks * 1000 stride = 10000.  If storage is 9999, it fails.
  EXPECT_THAT(pool.Validate(9999),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exceeds storage bytes")));
}

TEST(PoolSpecTest, ProtoRoundTrip) {
  PoolSpec pool = ValidPool();
  tpu_raiden::rpc::PoolSpecProto proto = ToProto(pool);
  auto roundtrip = PoolSpecFromProto(proto);
  ASSERT_OK(roundtrip);
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
  auto extents_or = ComputePoolBlockCopyExtents(pool, {0, 1, 3, 4, 5});
  ASSERT_OK(extents_or);
  const auto& extents = *extents_or;
  ASSERT_EQ(extents.size(), 2);
  EXPECT_EQ(extents[0].offset_bytes, 0);
  EXPECT_EQ(extents[0].size_bytes, 2000);  // block 0 and 1 coalesced
  EXPECT_EQ(extents[1].offset_bytes, 3000);
  EXPECT_EQ(extents[1].size_bytes, 3000);  // block 3, 4, 5 coalesced
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
