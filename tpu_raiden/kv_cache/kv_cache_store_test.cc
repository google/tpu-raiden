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

#include "tpu_raiden/kv_cache/kv_cache_store.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(KVCacheStoreTest, BasicTests) {
  KVCacheStore controller(50);
  EXPECT_EQ(controller.capacity(), 50);

  std::vector<std::string> hashes = {"4001", "4002"};
  std::vector<std::vector<RaidenId>> slices = {
      {RaidenId{"inference_server", "0", "kv_cache", 0}},
      {RaidenId{"inference_server", "1", "kv_cache", 0}}};

  // 1. Insert
  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);
  EXPECT_FALSE(
      controller.Insert(hashes, slices, true).first);  // Already exists

  // 2. Lookup with a partial miss at the end
  std::vector<std::string> hashes_with_miss = {"4001", "4002", "4003"};
  auto lookup_res = controller.Lookup(hashes_with_miss);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "4001");
  EXPECT_EQ((*lookup_res)[0].second.size(), 1);

  // Lookup with an early miss
  std::vector<std::string> hashes_early_miss = {"4001", "4003", "4002"};
  auto lookup_res_early = controller.Lookup(hashes_early_miss);
  ASSERT_TRUE(lookup_res_early.ok());
  EXPECT_EQ(lookup_res_early->size(), 1);
  EXPECT_EQ((*lookup_res_early)[0].first, "4001");

  // 3. Delete
  controller.Delete(hashes, slices);
  EXPECT_TRUE(
      controller.Insert(hashes, slices, true).first);  // Succesful again
}

TEST(KVCacheStoreTest, PinAndRelease) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes = {"101", "102"};
  std::vector<std::vector<RaidenId>> slices = {
      {RaidenId{"inference_server", "0", "kv_cache", 0}},
      {RaidenId{"inference_server", "1", "kv_cache", 0}}};

  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);

  // Pin both
  EXPECT_TRUE(controller.Pin(hashes));
  EXPECT_EQ(controller.GetPinCount("101"), 1);
  EXPECT_EQ(controller.GetPinCount("102"), 1);

  // Inserting a third element should fail to evict because both existing items
  // are pinned
  std::vector<std::string> hash_3 = {"103"};
  std::vector<std::vector<RaidenId>> slice_3 = {
      {RaidenId{"inference_server", "2", "kv_cache", 0}}};
  controller.Insert(hash_3, slice_3, true);

  // Release 101
  controller.Release({"101"});
  EXPECT_EQ(controller.GetPinCount("101"), 0);

  // Now inserting a fourth element (104) should successfully evict 101
  std::vector<std::string> hash_4 = {"104"};
  std::vector<std::vector<RaidenId>> slice_4 = {
      {RaidenId{"inference_server", "3", "kv_cache", 0}}};
  controller.Insert(hash_4, slice_4, true);

  // Lookup 101 should result in an immediate miss (return size 0 before 102)
  EXPECT_EQ(controller.Lookup({"101", "102"})->size(), 0);
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
}

TEST(KVCacheStoreTest, PartialPinRollback) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes = {"201", "202"};
  std::vector<std::vector<RaidenId>> slices = {
      {RaidenId{"inference_server", "0", "kv_cache", 0}},
      {RaidenId{"inference_server", "1", "kv_cache", 0}}};

  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);

  // Attempt to pin a sequence with a missing hash (203)
  EXPECT_FALSE(controller.Pin({"201", "202", "203"}));

  // Confirm that 201 and 202 were completely reverted (pin count 0)
  EXPECT_EQ(controller.GetPinCount("201"), 0);
  EXPECT_EQ(controller.GetPinCount("202"), 0);
}

TEST(KVCacheStoreTest, EvictionTracking) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes_1_2 = {"101", "102"};
  std::vector<std::vector<RaidenId>> slices_1_2 = {
      {RaidenId{"inference_server", "0", "kv_cache", 0}},
      {RaidenId{"inference_server", "1", "kv_cache", 1}}};

  // 1. Insert 101 and 102. No evictions should occur.
  auto res_1_2 = controller.Insert(hashes_1_2, slices_1_2, true);
  EXPECT_TRUE(res_1_2.first);
  EXPECT_TRUE(res_1_2.second.empty());

  // 2. Insert 103. Since capacity is 2, this must evict the LRU block (101).
  std::vector<std::string> hash_3 = {"103"};
  std::vector<std::vector<RaidenId>> slice_3 = {
      {RaidenId{"inference_server", "2", "kv_cache", 2}}};

  auto res_3 = controller.Insert(hash_3, slice_3, true);
  EXPECT_TRUE(res_3.first);
  ASSERT_EQ(res_3.second.size(), 1);
  EXPECT_EQ(res_3.second[0].first, "101");
  ASSERT_EQ(res_3.second[0].second.size(), 1);
  EXPECT_EQ(res_3.second[0].second[0].job_name, "inference_server");
  EXPECT_EQ(res_3.second[0].second[0].data_replica_idx, 0);

  // 3. Verify that lookup for 101 now misses, but 102 and 103 are present.
  EXPECT_EQ(controller.Lookup({"101"})->size(), 0);
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
  EXPECT_EQ(controller.Lookup({"103"})->size(), 1);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
