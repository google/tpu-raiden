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

#include "tpu_raiden/kv_cache/lru_cache.h"

#include <optional>
#include <string>
#include <utility>

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

  EXPECT_EQ(cache.size(), 2);
  EXPECT_TRUE(cache.Contains(1));
  EXPECT_FALSE(cache.Contains(2));
  EXPECT_TRUE(cache.Contains(3));
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

  EXPECT_EQ(cache.size(), 2);
  EXPECT_FALSE(cache.Contains(1));
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

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
