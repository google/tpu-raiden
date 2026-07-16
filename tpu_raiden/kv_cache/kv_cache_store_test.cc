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

#include <csignal>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/create_channel.h"
#include "grpcpp/grpcpp.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/lru_cache.h"

#ifndef _WIN32
int ignore_sigpipe = []() {
  std::signal(SIGPIPE, SIG_IGN);
  return 0;
}();
#endif

namespace tpu_raiden {
namespace kv_cache {
namespace {

TEST(KVCacheStoreTest, RaidenBlockIDConstructorAndEquality) {
  RaidenId id{"test_job", "0", "test_cache", 0};
  RaidenBlockID block_1(id, 10, 20, BlockStatus::HBM);
  EXPECT_EQ(block_1.raiden_id, id);
  EXPECT_EQ(block_1.host_block_id, 10);
  EXPECT_EQ(block_1.device_block_id, 20);
  EXPECT_EQ(block_1.status, BlockStatus::HBM);

  RaidenBlockID block_2(id, 10, 20, BlockStatus::HBM);
  EXPECT_EQ(block_1, block_2);

  RaidenBlockID block_3(id, 10, 21, BlockStatus::HBM);
  EXPECT_NE(block_1, block_3);

  RaidenBlockID block_4(id, 11, 20, BlockStatus::HBM);
  EXPECT_NE(block_1, block_4);
}

TEST(KVCacheStoreTest, BasicTests) {
  KVCacheStore controller(50);
  EXPECT_EQ(controller.capacity(), 50);

  std::vector<std::string> hashes = {"4001", "4002"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 0}};

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
  EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_replica_id, "0");

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
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 0}};

  EXPECT_TRUE(controller.Insert(hashes, slices, true).first);

  // Pin both
  EXPECT_TRUE(controller.Pin(hashes));
  EXPECT_EQ(controller.GetPinCount("101"), 1);
  EXPECT_EQ(controller.GetPinCount("102"), 1);

  // Inserting a third element should fail to evict because both existing items
  // are pinned
  std::vector<std::string> hash_3 = {"103"};
  std::vector<RaidenBlockID> slice_3 = {
      RaidenId{"inference_server", "2", "kv_cache", 0}};
  controller.Insert(hash_3, slice_3, true);

  // Release 101
  controller.Release({"101"});
  EXPECT_EQ(controller.GetPinCount("101"), 0);

  // Now inserting a fourth element (104) should successfully evict 101
  std::vector<std::string> hash_4 = {"104"};
  std::vector<RaidenBlockID> slice_4 = {
      RaidenId{"inference_server", "3", "kv_cache", 0}};
  controller.Insert(hash_4, slice_4, true);

  // Lookup 101 should result in an immediate miss (return size 0 before 102)
  EXPECT_EQ(controller.Lookup({"101", "102"})->size(), 0);
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
}

TEST(KVCacheStoreTest, PartialPinRollback) {
  KVCacheStore controller(2);

  std::vector<std::string> hashes = {"201", "202"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 0}};

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
  std::vector<RaidenBlockID> slices_1_2 = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1}};

  // 1. Insert 101 and 102. No evictions should occur.
  auto res_1_2 = controller.Insert(hashes_1_2, slices_1_2, true);
  EXPECT_TRUE(res_1_2.first);
  EXPECT_TRUE(res_1_2.second.empty());

  // 2. Insert 103. Since capacity is 2, this must evict the LRU block (101).
  std::vector<std::string> hash_3 = {"103"};
  std::vector<RaidenBlockID> slice_3 = {
      RaidenId{"inference_server", "2", "kv_cache", 2}};

  auto res_3 = controller.Insert(hash_3, slice_3, true);
  EXPECT_TRUE(res_3.first);
  ASSERT_EQ(res_3.second.size(), 1);
  EXPECT_EQ(res_3.second[0].first, "101");
  EXPECT_EQ(res_3.second[0].second.raiden_id.job_name, "inference_server");
  EXPECT_EQ(res_3.second[0].second.raiden_id.data_replica_idx, 0);

  // 3. Verify that lookup for 101 now misses, but 102 and 103 are present.
  EXPECT_EQ(controller.Lookup({"101"})->size(), 0);
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
  EXPECT_EQ(controller.Lookup({"103"})->size(), 1);
}

TEST(KVCacheStoreTest, GlobalLookupFallback) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Register some blocks in the registry
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::string hash1 = "global_hash_1";
  RaidenId host1{"job1", "0", "kv_cache", 0};
  int32_t block1 = 42;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash_shared = "shared_hash";
  RaidenId host_shared_remote{"job_shared", "0", "kv_cache", 0};
  int32_t block_shared_remote = 99;

  ASSERT_TRUE(
      registry_client
          .Register({{hash1, host1, block1},
                     {hash2, host2, block2},
                     {hash_shared, host_shared_remote, block_shared_remote}})
          .ok());

  // 3. Create KVCacheStore with the registry address
  KVCacheStore store(50, server_address);

  // Insert blocks locally
  std::vector<std::string> local_hashes = {"local_only_hash", "shared_hash"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"local_job", "0", "kv_cache", 1}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Case 1: Full local hit, no global hit
  {
    auto lookup_res = store.Lookup({"local_only_hash"}, /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 0);
  }

  // Case 2: Both local and global has the same hit, but we return local hit
  // results
  {
    auto lookup_res = store.Lookup({"shared_hash"}, /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "shared_hash");
    // Should return local info, not remote info from registry
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 1);
  }

  // Case 3: No local hit, only global hits
  {
    auto lookup_res = store.Lookup({"global_hash_1", "global_hash_2"},
                                   /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);

    EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "job1");
    EXPECT_EQ((*lookup_res)[0].second.host_block_id, 42);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);

    EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
    EXPECT_EQ((*lookup_res)[1].second.raiden_id.job_name, "job2");
    EXPECT_EQ((*lookup_res)[1].second.host_block_id, 43);
    EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::REMOTE);
  }

  // 4. Lookup with enable_global = false
  // It should stop at the first miss (which is the global hash if we query it)
  // If we query {"local_only_hash", "global_hash_1"}, it should return
  // local_only_hash and stop.
  {
    auto lookup_res = store.Lookup({"local_only_hash", "global_hash_1"},
                                   /*enable_global=*/false);
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
  }

  // 5. Lookup with enable_global = true
  // It should return both local and global
  {
    auto lookup_res =
        store.Lookup({"local_only_hash", "global_hash_1", "global_hash_2"},
                     /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 3);

    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");

    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
    EXPECT_EQ((*lookup_res)[1].second.raiden_id.job_name, "job1");
    EXPECT_EQ((*lookup_res)[1].second.host_block_id, 42);
    EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::REMOTE);

    EXPECT_EQ((*lookup_res)[2].first, "global_hash_2");
    EXPECT_EQ((*lookup_res)[2].second.raiden_id.job_name, "job2");
    EXPECT_EQ((*lookup_res)[2].second.host_block_id, 43);
    EXPECT_EQ((*lookup_res)[2].second.status, BlockStatus::REMOTE);
  }

  // 6. Lookup with enable_global = true, but registry has a miss
  // It should stop at the first miss in registry
  {
    auto lookup_res = store.Lookup(
        {"local_only_hash", "global_hash_1", "missing_hash", "global_hash_2"},
        /*enable_global=*/true);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);  // local_only_hash, global_hash_1
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
  }

  server->Shutdown();
}

TEST(KVCacheStoreTest, GlobalLookupRegistryDown) {
  // Create KVCacheStore with an unreachable registry address
  KVCacheStore store(50, "invalid.address:12345");

  // Insert one block locally
  std::vector<std::string> local_hashes = {"local_hash"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Lookup with enable_global = true.
  // It should NOT fail even though the registry is down. It should return the
  // local hit.
  auto lookup_res = store.Lookup({"local_hash", "missing_hash"},
                                 /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash");
  EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
}

TEST(KVCacheStoreTest, LookupCapLimit) {
  KVCacheStore store(2);

  std::vector<std::string> hashes = {"101", "102"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1}};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"101", "102", "103"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
}

TEST(KVCacheStoreTest, LookupCapLimitWithGlobal) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Register some blocks in the registry
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::string hash1 = "global_hash_1";
  RaidenId host1{"job1", "0", "kv_cache", 0};
  int32_t block1 = 42;

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash3 = "global_hash_3";
  RaidenId host3{"job3", "0", "kv_cache", 0};
  int32_t block3 = 44;

  ASSERT_TRUE(registry_client
                  .Register({{hash1, host1, block1},
                             {hash2, host2, block2},
                             {hash3, host3, block3}})
                  .ok());

  // 3. Create KVCacheStore with capacity 2
  KVCacheStore store(2, server_address);

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"global_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");

  server->Shutdown();
}

TEST(KVCacheStoreTest, LookupCapLimitMixed) {
  // 1. Start a local registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Register some blocks in the registry
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  std::string hash2 = "global_hash_2";
  RaidenId host2{"job2", "0", "kv_cache", 0};
  int32_t block2 = 43;

  std::string hash3 = "global_hash_3";
  RaidenId host3{"job3", "0", "kv_cache", 0};
  int32_t block3 = 44;

  ASSERT_TRUE(
      registry_client.Register({{hash2, host2, block2}, {hash3, host3, block3}})
          .ok());

  // 3. Create KVCacheStore with capacity 2
  KVCacheStore store(2, server_address);

  // Insert 1 block locally
  std::vector<std::string> local_hashes = {"local_hash_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Lookup 3 hashes, but capacity is 2. It should only return 2 (1 local, 1
  // global).
  std::vector<std::string> lookup_hashes = {"local_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");

  server->Shutdown();
}

TEST(KVCacheStoreTest, LookupAvailableSpaceLimit) {
  KVCacheStore store(3);

  std::vector<std::string> hashes = {"101", "102", "103"};
  std::vector<RaidenBlockID> slices = {
      RaidenId{"inference_server", "0", "kv_cache", 0},
      RaidenId{"inference_server", "1", "kv_cache", 1},
      RaidenId{"inference_server", "2", "kv_cache", 2}};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Pin 101. Pinned count = 1. Available space = 3 - 1 = 2.
  EXPECT_TRUE(store.Pin({"101"}));

  // Lookup 4 hashes. Since available space is 2, it should only return the
  // first 2.
  std::vector<std::string> lookup_hashes = {"101", "102", "103", "104"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
}

TEST(KVCacheStoreTest, InsertAndPin) {
  KVCacheStore store(2);

  // Insert local block
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Execute InsertAndPin
  std::vector<RaidenBlockID> slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"remote_job", "0", "kv_cache", 42}};
  auto res = store.InsertAndPin({"local_1", "remote_1"}, slices, true);
  EXPECT_TRUE(res.first);
  EXPECT_TRUE(res.second.empty());
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);

  // Since capacity is 2 and both local_1 and remote_1 are pinned, available
  // space is 0. Attempting to InsertAndPin remote_2 should fail due to lack
  // of space.
  auto res_fail = store.InsertAndPin({"remote_2"}, {}, true);
  EXPECT_FALSE(res_fail.first);
}

TEST(KVCacheStoreTest, LruCachePutBack) {
  LRUCache<std::string, int> cache(3);
  cache.Put("A", 1);
  cache.Put("B", 2);
  // MRU to LRU is: B, A.

  // PutBack("C", 3) should add C to the back (LRU position).
  cache.PutBack("C", 3);
  // Now MRU to LRU should be: B, A, C.
  // Let's verify by checking Evict(): first evicted item should be C!
  auto evicted1 = cache.Evict();
  ASSERT_TRUE(evicted1.has_value());
  EXPECT_EQ(evicted1->first, "C");

  auto evicted2 = cache.Evict();
  ASSERT_TRUE(evicted2.has_value());
  EXPECT_EQ(evicted2->first, "A");

  auto evicted3 = cache.Evict();
  ASSERT_TRUE(evicted3.has_value());
  EXPECT_EQ(evicted3->first, "B");
}

TEST(KVCacheStoreTest, ReleaseAndDelete) {
  KVCacheStore store(2);

  // Insert two local blocks (not remote)
  std::vector<std::string> local_hashes = {"local_1", "local_2"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::HOST),
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Now InsertAndPin two remote blocks, which will evict local_1 and local_2.
  std::vector<std::string> remote_hashes = {"remote_1", "remote_2"};
  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::REMOTE),
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::REMOTE)};
  auto res = store.InsertAndPin(remote_hashes, remote_slices, true);
  ASSERT_TRUE(res.first);
  ASSERT_EQ(res.second.size(), 2);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_2"), 1);
  EXPECT_EQ(store.Lookup({"local_1"})->size(), 0);
  EXPECT_EQ(store.Lookup({"local_2"})->size(), 0);

  // Now call ReleaseAndDelete to revert InsertAndPin!
  auto release_res = store.ReleaseAndDelete(remote_hashes, res.second);
  EXPECT_EQ(release_res.first, 2);
  EXPECT_TRUE(release_res.second.empty());

  // remote_1 and remote_2 should be unpinned and deleted (since REMOTE)
  EXPECT_EQ(store.GetPinCount("remote_1"), 0);
  EXPECT_EQ(store.GetPinCount("remote_2"), 0);
  EXPECT_EQ(store.Lookup({"remote_1"})->size(), 0);
  EXPECT_EQ(store.Lookup({"remote_2"})->size(), 0);

  // local_1 and local_2 should be restored to the cache!
  auto lookup_res = store.Lookup({"local_1", "local_2"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);

  // Test non-remote block in ReleaseAndDelete: should unpin without deleting
  store.InsertAndPin({"local_1"}, {local_slices[0]}, true);
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  auto res_non_remote = store.ReleaseAndDelete({"local_1"});
  EXPECT_EQ(res_non_remote.first, 0);
  EXPECT_TRUE(res_non_remote.second.empty());
  EXPECT_EQ(store.GetPinCount("local_1"), 0);
  EXPECT_EQ(store.Lookup({"local_1"})->size(), 1);

  // Test remote block pinned twice: after one ReleaseAndDelete, pin count is 1
  // so it should NOT be deleted!
  store.InsertAndPin({"remote_1"}, {remote_slices[0]}, true);
  store.Pin({"remote_1"});  // pin count is now 2
  EXPECT_EQ(store.GetPinCount("remote_1"), 2);
  auto res_pinned = store.ReleaseAndDelete({"remote_1"});
  EXPECT_EQ(res_pinned.first, 0);  // 0 deleted because pin count was 2 -> 1
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);
  EXPECT_EQ(store.Lookup({"remote_1"})->size(), 1);
  store.Release({"remote_1"});
  store.Delete({"remote_1"}, {remote_slices[0]});

  // Test partial restore: 1 deleted remote block with 2 evicted entries
  BlockSliceList mock_evicted = {{"evict_1", local_slices[0]},
                                 {"evict_2", local_slices[1]}};
  store.InsertAndPin({"remote_2"}, {remote_slices[1]}, true);
  auto res_partial = store.ReleaseAndDelete({"remote_2"}, mock_evicted);
  EXPECT_EQ(res_partial.first, 1);
  ASSERT_EQ(res_partial.second.size(), 1);
  EXPECT_EQ(res_partial.second[0].first, "evict_1");  // evict_2 was restored!
}

using ::testing::ElementsAre;

class KVCacheStoreEmbeddedControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    unit_.set_job_name("test_job");
    unit_.set_job_replica_id("0");
    unit_.set_data_name("test_data");

    orchestrator_service_ =
        std::make_unique<::tpu_raiden::RaidenOrchestrator>();
    grpc::ServerBuilder builder;
    int bound_port = 0;
    builder.AddListeningPort("0.0.0.0:0", grpc::InsecureServerCredentials(),
                             &bound_port);
    builder.RegisterService(orchestrator_service_.get());
    orchestrator_server_ = builder.BuildAndStart();
    orchestrator_address_ = absl::StrCat("localhost:", bound_port);
  }

  void TearDown() override {
    if (orchestrator_server_) {
      orchestrator_server_->Shutdown();
      orchestrator_server_->Wait();
    }
  }

  void RegisterAndInitWorker(
      ::tpu_raiden::controller::RaidenController& controller,
      const std::string& worker_id, const std::string& worker_address) {
    std::string server_address =
        absl::StrCat("localhost:", controller.raiden_controller_port());
    ::tpu_raiden::core::controller::RaidenControllerClient client(
        server_address);
    auto status =
        client.RegisterWorker(worker_id, worker_address, worker_address);
    ASSERT_TRUE(status.ok()) << status.message();
  }

  rpc::RaidenIdProto unit_;
  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer> test_server_;
  std::unique_ptr<::tpu_raiden::RaidenOrchestrator> orchestrator_service_;
  std::unique_ptr<grpc::Server> orchestrator_server_;
  std::string orchestrator_address_;
};

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, 0, orchestrator_address_);
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid);

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockID(rid, -1, 1, BlockStatus::HBM)};

  // Insert them as HBM blocks
  ASSERT_TRUE(store.Insert(hashes, slices, false).first);

  // Pin them
  ASSERT_TRUE(store.Pin(hashes));

  // Save them
  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    if (!save_done.empty()) {
      EXPECT_THAT(save_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // Verify transfer manager was called
  EXPECT_EQ(mock_mgr.d2h_calls, 1);
  EXPECT_EQ(mock_mgr.h2d_calls, 0);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  // host_block_ids are allocated starting from 0
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(0, 1));

  // Verify status in store is updated to HOST_AND_HBM
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 0);
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 1);
  EXPECT_EQ((*lookup_res)[1].second.device_block_id, 1);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, 0, orchestrator_address_);
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid);

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  // Insert as HOST only blocks
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, -1, BlockStatus::HOST),
      RaidenBlockID(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Pin them
  ASSERT_TRUE(store.Pin(hashes));

  // Load them to device block 2 and 3
  absl::Status status = store.Load(hashes, {2, 3});
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    if (!load_failed.empty()) {
      FAIL() << "Async Load failed during polling";
    }
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // Verify transfer manager was called
  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(2, 3));

  // Verify status in store is updated to HOST_AND_HBM
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 2);
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 1);
  EXPECT_EQ((*lookup_res)[1].second.device_block_id, 3);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveMultiWorkerSuccess) {
  auto test_server_0 = ::tpu_raiden::controller::CreateTestWorkerServer();
  auto test_server_1 = ::tpu_raiden::controller::CreateTestWorkerServer();

  ::tpu_raiden::controller::MockTransferManager mock_mgr_0;
  ::tpu_raiden::controller::MockTransferManager mock_mgr_1;

  test_server_0->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_0));
  test_server_1->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_1));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, 0, orchestrator_address_);

  RegisterAndInitWorker(*controller, "worker_0", test_server_0->server_address);
  RegisterAndInitWorker(*controller, "worker_1", test_server_1->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid);

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockID(rid, -1, 1, BlockStatus::HBM)};

  ASSERT_TRUE(store.Insert(hashes, slices, false).first);

  // Pin them
  ASSERT_TRUE(store.Pin(hashes));

  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    if (!save_done.empty()) {
      EXPECT_THAT(save_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  EXPECT_EQ(mock_mgr_0.d2h_calls, 1);
  EXPECT_EQ(mock_mgr_0.h2d_calls, 0);
  EXPECT_THAT(mock_mgr_0.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_0.last_dst_offsets, ElementsAre(0, 1));

  EXPECT_EQ(mock_mgr_1.d2h_calls, 1);
  EXPECT_EQ(mock_mgr_1.h2d_calls, 0);
  EXPECT_THAT(mock_mgr_1.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_1.last_dst_offsets, ElementsAre(0, 1));

  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 0);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadMultiWorkerSuccess) {
  auto test_server_0 = ::tpu_raiden::controller::CreateTestWorkerServer();
  auto test_server_1 = ::tpu_raiden::controller::CreateTestWorkerServer();

  ::tpu_raiden::controller::MockTransferManager mock_mgr_0;
  ::tpu_raiden::controller::MockTransferManager mock_mgr_1;

  test_server_0->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_0));
  test_server_1->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr_1));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, 0, orchestrator_address_);

  RegisterAndInitWorker(*controller, "worker_0", test_server_0->server_address);
  RegisterAndInitWorker(*controller, "worker_1", test_server_1->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid);

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, -1, BlockStatus::HOST),
      RaidenBlockID(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Pin them
  ASSERT_TRUE(store.Pin(hashes));

  absl::Status status = store.Load(hashes, {2, 3});
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  while (!done) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    if (!load_failed.empty()) {
      FAIL() << "Async Load failed during polling";
    }
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  EXPECT_EQ(mock_mgr_0.d2h_calls, 0);
  EXPECT_EQ(mock_mgr_0.h2d_calls, 1);
  EXPECT_THAT(mock_mgr_0.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_0.last_dst_offsets, ElementsAre(2, 3));

  EXPECT_EQ(mock_mgr_1.d2h_calls, 0);
  EXPECT_EQ(mock_mgr_1.h2d_calls, 1);
  EXPECT_THAT(mock_mgr_1.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr_1.last_dst_offsets, ElementsAre(2, 3));

  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 2);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveWriteThrough) {
  // 1. Start a local mock registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // 2. Setup mock transfer manager & controller
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, 0, orchestrator_address_);
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // 3. Initialize KVCacheStore with the registry server address & controller
  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), server_address, rid);

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockID(rid, -1, 1, BlockStatus::HBM)};

  // 4. Insert them as HBM blocks locally and pin them
  ASSERT_TRUE(store.Insert(hashes, slices, false).first);
  ASSERT_TRUE(store.Pin(hashes));

  // 5. Call Save on the store
  absl::Status status = store.Save(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // 6. Poll for completion
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    if (!save_done.empty()) {
      EXPECT_THAT(save_done,
                  ::testing::UnorderedElementsAre("hash_1", "hash_2"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // 7. Verify registry has been updated (need to poll registry since
  // registration is async)
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  bool registered = false;
  std::vector<global_registry::KVBlockMetadata> metadata_results;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup(hashes);
    if (lookup_res.ok() && lookup_res->size() == 2) {
      metadata_results = *std::move(lookup_res);
      registered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }

  ASSERT_TRUE(registered)
      << "Block hashes were not registered in global registry";

  // Verify the metadata results
  EXPECT_EQ(metadata_results[0].raiden_id().job_name(), rid.job_name);
  EXPECT_EQ(metadata_results[0].raiden_id().job_replica_id(),
            rid.job_replica_id);
  EXPECT_EQ(metadata_results[0].raiden_id().data_name(), rid.data_name);
  EXPECT_EQ(metadata_results[0].raiden_id().data_replica_idx(),
            rid.data_replica_idx);
  EXPECT_EQ(metadata_results[0].block_id(),
            0);  // first host block allocated is 0

  EXPECT_EQ(metadata_results[1].raiden_id().job_name(), rid.job_name);
  EXPECT_EQ(metadata_results[1].raiden_id().job_replica_id(),
            rid.job_replica_id);
  EXPECT_EQ(metadata_results[1].raiden_id().data_name(), rid.data_name);
  EXPECT_EQ(metadata_results[1].raiden_id().data_replica_idx(),
            rid.data_replica_idx);
  EXPECT_EQ(metadata_results[1].block_id(),
            1);  // second host block allocated is 1

  server->Shutdown();
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
