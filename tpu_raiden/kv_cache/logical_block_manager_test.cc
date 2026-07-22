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

#include "tpu_raiden/kv_cache/logical_block_manager.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Optional;

TEST(LogicalBlockManagerTest, InitialState) {
  LogicalBlockManager manager(5);
  EXPECT_EQ(manager.total_blocks(), 5);
  EXPECT_EQ(manager.num_free_blocks(), 5);
  EXPECT_EQ(manager.num_allocated_blocks(), 0);
  EXPECT_EQ(manager.num_locked_blocks(), 0);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(manager.IsAllocated(i));
    EXPECT_FALSE(manager.IsLocked(i));
  }
}

TEST(LogicalBlockManagerTest, BasicAllocation) {
  LogicalBlockManager manager(5);
  auto blocks_or = manager.Allocate(3, /*lock=*/false);
  ASSERT_TRUE(blocks_or.ok());
  EXPECT_THAT(*blocks_or, ElementsAre(0, 1, 2));

  EXPECT_EQ(manager.num_free_blocks(), 2);
  EXPECT_EQ(manager.num_allocated_blocks(), 3);
  EXPECT_EQ(manager.num_locked_blocks(), 0);

  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(manager.IsAllocated(i));
    EXPECT_FALSE(manager.IsLocked(i));
  }
}

TEST(LogicalBlockManagerTest, AllocationWithLocking) {
  LogicalBlockManager manager(5);
  auto blocks_or = manager.Allocate(2, /*lock=*/true);
  ASSERT_TRUE(blocks_or.ok());
  EXPECT_THAT(*blocks_or, ElementsAre(0, 1));

  EXPECT_EQ(manager.num_locked_blocks(), 2);
  EXPECT_TRUE(manager.IsLocked(0));
  EXPECT_TRUE(manager.IsLocked(1));
}

TEST(LogicalBlockManagerTest, LruEvictionOrder) {
  LogicalBlockManager manager(4);

  // Allocate 2 blocks to entity 10 (unlocked).
  auto blocks1 = manager.Allocate(2);
  ASSERT_TRUE(blocks1.ok());
  EXPECT_THAT(*blocks1, ElementsAre(0, 1));

  // Allocate 2 blocks to entity 20 (unlocked).
  auto blocks2 = manager.Allocate(2);
  ASSERT_TRUE(blocks2.ok());
  EXPECT_THAT(*blocks2, ElementsAre(2, 3));

  EXPECT_EQ(manager.num_free_blocks(), 0);

  // Requesting 2 blocks for entity 30 should evict entity 10's blocks
  // because they were allocated earlier (LRU).
  auto blocks3 = manager.Allocate(2);
  ASSERT_TRUE(blocks3.ok());
  EXPECT_THAT(*blocks3, ElementsAre(0, 1));
}

TEST(LogicalBlockManagerTest, AccessUpdatesLruOrder) {
  LogicalBlockManager manager(4);

  auto blocks1 = manager.Allocate(2);
  ASSERT_TRUE(blocks1.ok());
  auto blocks2 = manager.Allocate(2);
  ASSERT_TRUE(blocks2.ok());

  // Access entity 10's blocks, making entity 20's blocks the least recently
  // used.
  EXPECT_TRUE(manager.AccessBlock(0).ok());
  EXPECT_TRUE(manager.AccessBlock(1).ok());

  // Allocate 2 blocks for entity 30. Should evict entity 20's blocks (2 and 3).
  auto blocks3 = manager.Allocate(2);
  ASSERT_TRUE(blocks3.ok());
  EXPECT_THAT(*blocks3, ElementsAre(2, 3));
}

TEST(LogicalBlockManagerTest, LockedBlocksPreventEviction) {
  LogicalBlockManager manager(4);

  // Allocate 2 locked blocks to entity 10.
  ASSERT_TRUE(manager.Allocate(2, /*lock=*/true).ok());
  // Allocate 2 unlocked blocks.
  ASSERT_TRUE(manager.Allocate(2, /*lock=*/false).ok());

  // Requesting 3 blocks should fail since only 2 blocks are evictable.
  auto failed_or = manager.Allocate(3);
  EXPECT_FALSE(failed_or.ok());
  EXPECT_TRUE(absl::IsResourceExhausted(failed_or.status()));
}

TEST(LogicalBlockManagerTest, UnlockAllowsEviction) {
  LogicalBlockManager manager(4);

  ASSERT_TRUE(manager.Allocate(2, /*lock=*/true).ok());
  ASSERT_TRUE(manager.Allocate(2, /*lock=*/false).ok());

  // Unlock entity 10's blocks.
  std::vector<int> to_unlock = {0, 1};
  EXPECT_TRUE(manager.Unlock(to_unlock).ok());
  EXPECT_EQ(manager.num_locked_blocks(), 0);

  // Now requesting 3 blocks succeeds.
  auto blocks_or = manager.Allocate(3);
  EXPECT_TRUE(blocks_or.ok());
  EXPECT_EQ(blocks_or->size(), 3);
}

TEST(LogicalBlockManagerTest, InvalidArguments) {
  LogicalBlockManager manager(5);

  EXPECT_FALSE(manager.Allocate(0).ok());
  EXPECT_FALSE(manager.Allocate(-1).ok());
  EXPECT_FALSE(manager.Allocate(6).ok());

  EXPECT_FALSE(manager.AccessBlock(5).ok());
  EXPECT_FALSE(manager.AccessBlock(-1).ok());
  // Accessing unallocated block
  EXPECT_FALSE(manager.AccessBlock(0).ok());

  std::vector<int> invalid_unlock = {0};
  EXPECT_FALSE(manager.Unlock(invalid_unlock).ok());
}

TEST(LogicalBlockManagerTest, RestoreAllocatedMarksBlocksAllocatedAndLocked) {
  LogicalBlockManager manager(5);
  ASSERT_TRUE(manager.RestoreAllocated({1, 3}).ok());
  EXPECT_TRUE(manager.IsAllocated(1));
  EXPECT_TRUE(manager.IsLocked(1));
  EXPECT_TRUE(manager.IsAllocated(3));
  EXPECT_TRUE(manager.IsLocked(3));
  EXPECT_EQ(manager.num_free_blocks(), 3);

  // Restored blocks are never handed out by subsequent allocations.
  auto blocks_or = manager.Allocate(3, /*lock=*/true);
  ASSERT_TRUE(blocks_or.ok());
  EXPECT_THAT(*blocks_or, ElementsAre(0, 2, 4));
  // Everything is locked now: further allocation must fail.
  EXPECT_FALSE(manager.Allocate(1).ok());
}

TEST(LogicalBlockManagerTest, RestoreAllocatedValidatesAtomically) {
  LogicalBlockManager manager(5);
  ASSERT_TRUE(manager.Allocate(1).ok());  // Block 0 becomes allocated.

  // Out of range.
  EXPECT_EQ(manager.RestoreAllocated({1, 5}).code(),
            absl::StatusCode::kInvalidArgument);
  // Already allocated (even though unlocked).
  EXPECT_EQ(manager.RestoreAllocated({1, 0}).code(),
            absl::StatusCode::kFailedPrecondition);
  // Duplicate ID within the batch.
  EXPECT_EQ(manager.RestoreAllocated({2, 2}).code(),
            absl::StatusCode::kInvalidArgument);

  // Failed calls must not have modified any state.
  EXPECT_FALSE(manager.IsAllocated(1));
  EXPECT_FALSE(manager.IsAllocated(2));
  EXPECT_EQ(manager.num_allocated_blocks(), 1);
}

TEST(LogicalBlockManagerTest, RestoredBlocksReusableAfterUnlock) {
  LogicalBlockManager manager(3);
  ASSERT_TRUE(manager.RestoreAllocated({0, 1, 2}).ok());
  ASSERT_TRUE(manager.Unlock({1}).ok());

  // The unlocked restored block is evictable and gets reused.
  auto blocks_or = manager.Allocate(1);
  ASSERT_TRUE(blocks_or.ok());
  EXPECT_THAT(*blocks_or, ElementsAre(1));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
