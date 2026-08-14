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

#include "tpu_sync/kv_cache/lru_cache.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(LRUCacheTest, BasicPutAndGet) {
  LRUCache<int, std::string> cache(3);

  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);
  EXPECT_EQ(cache.capacity(), 3);

  EXPECT_EQ(cache.Put(1, "one"), std::nullopt);
  EXPECT_EQ(cache.Put(2, "two"), std::nullopt);
  EXPECT_EQ(cache.Put(3, "three"), std::nullopt);

  EXPECT_EQ(cache.size(), 3);
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_TRUE(cache.Contains(3));

  EXPECT_EQ(*cache.Get(1), "one");
  EXPECT_EQ(*cache.Get(2), "two");
  EXPECT_EQ(*cache.Get(3), "three");
}

TEST(LRUCacheTest, EvictionOnOverfull) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  // Get(1) promotes 1 to MRU, making 2 the LRU
  cache.Get(1);

  // Put(3) should evict 2
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
  EXPECT_EQ(evicted->second, "two");

  EXPECT_EQ(cache.size(), 3);
  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_FALSE(cache.Contains(2));  // 2 is candidate
  EXPECT_TRUE(cache.Contains(3));
  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 2);
}

TEST(LRUCacheTest, PinningProtectsAgainstEviction) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  // Pin 2 so it cannot be evicted
  EXPECT_TRUE(cache.Pin(2));
  EXPECT_EQ(cache.GetPinCount(2), 1);

  // Get(2) promotes 2, making 1 the LRU
  // Put(3) should evict 1 because 2 is pinned
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 1);
  EXPECT_EQ(evicted->second, "one");

  EXPECT_EQ(cache.size(), 3);
  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_FALSE(cache.Contains(1));  // 1 is candidate
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_TRUE(cache.Contains(3));

  // Unpin 2. Since it is now unpinned, it is promoted to MRU.
  // 3 (inserted earlier) now becomes the LRU.
  EXPECT_TRUE(cache.Unpin(2));
  EXPECT_EQ(cache.GetPinCount(2), 0);

  auto evicted_unpinned = cache.Put(4, "four");
  ASSERT_TRUE(evicted_unpinned.has_value());
  EXPECT_EQ(evicted_unpinned->first,
            3);  // 3 is now LRU because 2 was promoted on unpin
}

TEST(LRUCacheTest, InsertionRejectedWhenAllPinned) {
  LRUCache<int, std::string> cache(2);

  cache.Put(101, "one");
  cache.Put(102, "two");

  EXPECT_TRUE(cache.Pin(101));
  EXPECT_TRUE(cache.Pin(102));

  auto evicted = cache.Put(103, "three");
  EXPECT_EQ(evicted, std::nullopt);
  EXPECT_FALSE(cache.Contains(103));
  EXPECT_EQ(cache.size(), 2);
}

TEST(LRUCacheTest, PeekDoesNotPromote) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  // Peek(1) should NOT promote 1 to MRU
  EXPECT_EQ(*cache.Peek(1), "one");

  // Put(3) should evict 1 (since 1 is still LRU)
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 1);
}

TEST(LRUCacheTest, EraseAndClear) {
  LRUCache<int, std::string> cache(2);

  cache.Put(1, "one");
  cache.Put(2, "two");

  EXPECT_TRUE(cache.Erase(1));
  EXPECT_FALSE(cache.Contains(1));
  EXPECT_EQ(cache.size(), 1);

  cache.Clear();
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.size(), 0);
}

TEST(LRUCacheTest, PutBack) {
  LRUCache<int, std::string> cache(3);
  cache.Put(1, "one");
  cache.Put(2, "two");
  // MRU to LRU is: 2, 1.

  // PutBack(3) should add 3 to the back (LRU position).
  cache.PutBack(3, "three");
  // Now MRU to LRU should be: 2, 1, 3.
  // Verify by checking Evict(): first evicted item should be 3.
  auto evicted1 = cache.Evict();
  ASSERT_TRUE(evicted1.has_value());
  EXPECT_EQ(evicted1->first, 3);

  auto evicted2 = cache.Evict();
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, 1);

  auto evicted3 = cache.Evict();
  ASSERT_TRUE(evicted3.has_value());
  EXPECT_EQ(evicted3->first, 2);
}

TEST(LRUCacheTest, RescueFromCandidateList) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  // Evict will move 1 to candidate list (since 1 is LRU)
  auto evicted = cache.Evict();
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 1);

  // 1 is in candidate list, size (total including candidates) is still 2
  EXPECT_EQ(cache.size(), 2);
  EXPECT_EQ(cache.active_size(), 1);

  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 1);

  // Get(1) should rescue 1
  auto* val = cache.Get(1);
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, "one");

  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());

  // Since 1 was rescued, it becomes MRU. 2 is now LRU.
  // Putting 3 should evict 2.
  auto evicted2 = cache.Put(3, "three");
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, 2);

  auto candidates2 = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates2.size(), 1);
  EXPECT_EQ(candidates2[0], 2);
}

TEST(LRUCacheTest, PinCandidateAndRescue) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  cache.Evict();  // 1 becomes candidate

  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 1);
  EXPECT_EQ(cache.active_size(), 1);

  // Pinning 1 should rescue and pin it
  EXPECT_TRUE(cache.Pin(1));
  EXPECT_EQ(cache.GetPinCount(1), 1);
  EXPECT_EQ(cache.active_size(), 2);
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());

  // 1 is pinned, 2 is unpinned. Putting 3 should evict 2.
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
}

TEST(LRUCacheTest, EraseCandidate) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  cache.Evict();  // 1 becomes candidate
  auto candidates = cache.GetEvictCandidateKeys();
  ASSERT_EQ(candidates.size(), 1);
  EXPECT_EQ(candidates[0], 1);
  EXPECT_EQ(cache.size(), 2);

  // Erasing candidate 1
  EXPECT_TRUE(cache.Erase(1));
  EXPECT_EQ(cache.size(), 1);
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());
  EXPECT_FALSE(cache.Contains(1));
}

TEST(LRUCacheTest, GetEvictableKeysScanningOrder) {
  LRUCache<int, std::string> cache(3);
  cache.Put(1, "one");
  cache.Put(2, "two");
  cache.Put(3, "three");
  // Active LRU: 3, 2, 1 (MRU to LRU)

  cache.Evict();  // 1 becomes candidate
  // Candidates: 1
  // Active: 3, 2

  cache.Put(4, "four");
  // Candidates: 1
  // Active: 4, 3, 2 (MRU to LRU)

  cache.Evict();  // 2 becomes candidate
  // Candidates: 1, 2 (1 is older)
  // Active: 4, 3 (3 is older active)

  auto keys = cache.GetEvictableKeys(4);
  ASSERT_EQ(keys.size(), 4);
  EXPECT_EQ(keys[0], 1);  // Oldest candidate
  EXPECT_EQ(keys[1], 2);  // Newer candidate
  EXPECT_EQ(keys[2], 3);  // Oldest active
  EXPECT_EQ(keys[3], 4);  // Newer active
}

TEST(LRUCacheTest, RestoreLastCandidate) {
  LRUCache<int, std::string> cache(3);
  cache.Put(1, "one");
  cache.Put(2, "two");
  cache.Put(3, "three");
  // MRU to LRU is: 3, 2, 1.

  // Evict twice.
  auto evicted1 = cache.Evict();
  ASSERT_TRUE(evicted1.has_value());
  EXPECT_EQ(evicted1->first, 1);

  auto evicted2 = cache.Evict();
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, 2);

  // Candidates: 1, 2.

  // Restore once. It should restore 2 (last candidate).
  EXPECT_TRUE(cache.RestoreLastCandidate());
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_FALSE(cache.GetEvictCandidateKeys().empty());
  EXPECT_EQ(cache.GetEvictCandidateKeys().size(), 1);
  EXPECT_EQ(cache.GetEvictCandidateKeys()[0], 1);

  // Restore again. It should restore 1.
  EXPECT_TRUE(cache.RestoreLastCandidate());
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_TRUE(cache.GetEvictCandidateKeys().empty());

  // Restore when candidate list is empty should return false.
  EXPECT_FALSE(cache.RestoreLastCandidate());

  // Verify that restored items are at the back of the LRU list (LRU position).
  // Now MRU to LRU should be: 3, 2, 1.
  auto evict_again1 = cache.Evict();
  ASSERT_TRUE(evict_again1.has_value());
  EXPECT_EQ(evict_again1->first, 1);

  auto evict_again2 = cache.Evict();
  ASSERT_TRUE(evict_again2.has_value());
  EXPECT_EQ(evict_again2->first, 2);

  auto evict_again3 = cache.Evict();
  ASSERT_TRUE(evict_again3.has_value());
  EXPECT_EQ(evict_again3->first, 3);
}

TEST(LRUCacheTest, CandidateVisibility) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  cache.Evict();  // 1 becomes candidate

  // 1 should not be visible to Contains, Peek, PeekMutable
  EXPECT_FALSE(cache.Contains(1));
  EXPECT_EQ(cache.Peek(1), nullptr);
  EXPECT_EQ(cache.PeekMutable(1), nullptr);

  // 1 should still be visible to PeekIncludingCandidates and
  // PeekMutableIncludingCandidates
  EXPECT_NE(cache.PeekIncludingCandidates(1), nullptr);
  EXPECT_EQ(*cache.PeekIncludingCandidates(1), "one");
  EXPECT_NE(cache.PeekMutableIncludingCandidates(1), nullptr);
  EXPECT_EQ(*cache.PeekMutableIncludingCandidates(1), "one");

  // 2 (active) should be visible to all
  EXPECT_TRUE(cache.Contains(2));
  EXPECT_NE(cache.Peek(2), nullptr);
  EXPECT_EQ(*cache.Peek(2), "two");
  EXPECT_NE(cache.PeekMutable(2), nullptr);
  EXPECT_EQ(*cache.PeekMutable(2), "two");
}

TEST(LRUCacheTest, GetAndPinBasic) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  // Non-existent key
  EXPECT_EQ(cache.GetAndPin(99), nullptr);

  // Existing key: returns value pointer and pins item
  auto* val1 = cache.GetAndPin(1);
  ASSERT_NE(val1, nullptr);
  EXPECT_EQ(*val1, "one");
  EXPECT_EQ(cache.GetPinCount(1), 1);

  // Calling GetAndPin again increments pin count further
  auto* val1_again = cache.GetAndPin(1);
  ASSERT_NE(val1_again, nullptr);
  EXPECT_EQ(*val1_again, "one");
  EXPECT_EQ(cache.GetPinCount(1), 2);

  // 1 is pinned twice, 2 is unpinned. Put(3) evicts 2.
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
  // 2 is now a candidate
  EXPECT_FALSE(cache.Contains(2));

  // GetAndPin on candidate 2 rescues and pins it
  auto* val2 = cache.GetAndPin(2);
  ASSERT_NE(val2, nullptr);
  EXPECT_EQ(*val2, "two");
  EXPECT_EQ(cache.GetPinCount(2), 1);
  EXPECT_TRUE(cache.Contains(2));
}

// --- Handles -------------------------------------------------------------
//
// A handle is a stable, pinned reference to one entry. The tests below are
// mostly about the "stable" half: they subject a held handle to every
// operation that moves a node around internally and check it still reaches the
// right value. If any of these fail, the callers that keep handles across an
// async operation are reading freed or wrong memory.

TEST(LRUCacheTest, AcquireHandleReadsAndPins) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");

  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());
  EXPECT_EQ(*handle.value(), "one");
  EXPECT_EQ(cache.GetPinCount(1), 1);

  cache.ReleaseHandle(handle);
  EXPECT_FALSE(handle.valid());
  EXPECT_EQ(cache.GetPinCount(1), 0);
}

TEST(LRUCacheTest, AcquireHandleOnMissingKeyIsEmpty) {
  LRUCache<int, std::string> cache(2);
  auto handle = cache.AcquireHandle(99);
  EXPECT_FALSE(handle.valid());
  // Releasing an empty handle is a no-op, so unwinding a partial batch does
  // not need to track which acquisitions succeeded.
  cache.ReleaseHandle(handle);
  EXPECT_FALSE(handle.valid());
}

TEST(LRUCacheTest, HandleWritesAreVisibleThroughTheCache) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");

  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());
  *handle.value() = "mutated";

  // The whole point: the write lands in the entry itself, not in a copy.
  ASSERT_NE(cache.Peek(1), nullptr);
  EXPECT_EQ(*cache.Peek(1), "mutated");
  cache.ReleaseHandle(handle);
}

TEST(LRUCacheTest, HandleSurvivesPromotionAndOtherInsertions) {
  LRUCache<int, std::string> cache(4);
  cache.Put(1, "one");
  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());

  // Promotions splice the node between lists; insertions grow the map and can
  // rehash it. Neither may disturb a node's address.
  cache.Put(2, "two");
  cache.Put(3, "three");
  (void)cache.Get(2);
  (void)cache.Get(3);
  cache.Put(4, "four");

  EXPECT_EQ(*handle.value(), "one");
  cache.ReleaseHandle(handle);
}

TEST(LRUCacheTest, HandleSurvivesPutOnTheSameKey) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());

  // Put on an existing key assigns the value in place rather than recreating
  // the node, so the handle must observe the new value, not a stale one.
  cache.Put(1, "replaced");
  EXPECT_EQ(*handle.value(), "replaced");
  cache.ReleaseHandle(handle);
}

TEST(LRUCacheTest, HandlePinBlocksEvictionAndSurvivesPressure) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");

  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());

  // 1 is pinned by the handle, so filling the cache must evict 2 instead.
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  EXPECT_EQ(evicted->first, 2);
  EXPECT_EQ(*handle.value(), "one");

  cache.ReleaseHandle(handle);
  // Only after release does 1 become evictable again.
  EXPECT_EQ(cache.GetPinCount(1), 0);
}

TEST(LRUCacheTest, HandlePinIsIndependentOfCallerPins) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");

  // This is the property the async callers rely on: a handle's pin is its own
  // reference, so someone else unpinning cannot make the entry evictable
  // while the handle is still held.
  ASSERT_TRUE(cache.Pin(1));
  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());
  EXPECT_EQ(cache.GetPinCount(1), 2);

  EXPECT_TRUE(cache.Unpin(1));
  EXPECT_EQ(cache.GetPinCount(1), 1);
  EXPECT_EQ(*handle.value(), "one");

  cache.ReleaseHandle(handle);
  EXPECT_EQ(cache.GetPinCount(1), 0);
}

TEST(LRUCacheTest, HandleRescuesAnEvictionCandidate) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");
  cache.Put(2, "two");
  ASSERT_TRUE(cache.Pin(1));
  auto evicted = cache.Put(3, "three");
  ASSERT_TRUE(evicted.has_value());
  ASSERT_EQ(evicted->first, 2);
  ASSERT_FALSE(cache.Contains(2));  // 2 is a candidate, still holding its block

  // Acquiring pins it, which pulls it back out of the candidate list -- the
  // same rescue GetAndPin performs.
  auto handle = cache.AcquireHandle(2);
  ASSERT_TRUE(handle.valid());
  EXPECT_EQ(*handle.value(), "two");
  EXPECT_TRUE(cache.Contains(2));
  cache.ReleaseHandle(handle);
}

TEST(LRUCacheTest, HandleMoveTransfersTheSinglePin) {
  LRUCache<int, std::string> cache(2);
  cache.Put(1, "one");

  auto handle = cache.AcquireHandle(1);
  ASSERT_TRUE(handle.valid());
  EXPECT_EQ(cache.GetPinCount(1), 1);

  // Move must transfer ownership, not duplicate it: one acquire, one release,
  // whatever the handle's storage does in between.
  auto moved = std::move(handle);
  EXPECT_TRUE(moved.valid());
  EXPECT_FALSE(handle.valid());  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(cache.GetPinCount(1), 1);
  EXPECT_EQ(*moved.value(), "one");

  // Releasing the moved-from handle must not drop the live pin.
  cache.ReleaseHandle(handle);  // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(cache.GetPinCount(1), 1);

  cache.ReleaseHandle(moved);
  EXPECT_EQ(cache.GetPinCount(1), 0);
}

TEST(LRUCacheTest, HandlesSurviveBeingCarriedInAVector) {
  LRUCache<int, std::string> cache(8);
  for (int i = 0; i < 4; ++i) cache.Put(i, std::string("v") + std::to_string(i));

  // Callers keep one handle per block in a vector that gets moved between
  // structures; a reallocation there must not disturb the pins.
  std::vector<LRUCache<int, std::string>::Handle> handles;
  for (int i = 0; i < 4; ++i) handles.push_back(cache.AcquireHandle(i));
  auto carried = std::move(handles);

  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(carried[i].valid());
    EXPECT_EQ(*carried[i].value(), std::string("v") + std::to_string(i));
    EXPECT_EQ(cache.GetPinCount(i), 1);
  }
  for (auto& h : carried) cache.ReleaseHandle(h);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(cache.GetPinCount(i), 0);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
