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

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/controller/controller_client.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

#ifndef _WIN32
int ignore_sigpipe = []() {
  std::signal(SIGPIPE, SIG_IGN);
  return 0;
}();
#endif

namespace tpu_raiden {
namespace kv_cache {

class KVCacheStoreTest {
 public:
  static size_t Evict(KVCacheStore& store,
                      const std::vector<std::string>& block_hashes) {
    return store.Evict(block_hashes);
  }

  static ::tpu_raiden::controller::RaidenController* GetController(
      const KVCacheStore& store) {
    return store.raiden_controller_.get();
  }

  static std::vector<std::string> GetEvictCandidateKeys(
      const KVCacheStore& store) {
    absl::MutexLock lock(store.mutex_);
    return store.lru_cache_.GetEvictCandidateKeys();
  }
};

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

  // 101 is candidate.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("101"));

  // Lookup {"101", "102"} will miss 101 (since candidates are invisible to
  // Peek). It will break early and return empty.
  auto lookup_res = controller.Lookup({"101", "102"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 0);

  // 102 is still in cache.
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

  // 3. Verify that 101 is in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("101"));

  // 4. Verify that lookup for 101 misses (since lookup uses Peek and ignores
  // candidates).
  auto lookup_res = controller.Lookup({"101"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 0);
  // 101 should still be in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("101"));

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

// --- ReadRemote step 6a: source-side ValidateAndPinHostBlocks ---

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksSuccessReDerivesIdsAndPins) {
  KVCacheStore store(4);
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"h0", "h1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, /*host_block_id=*/5, /*device_block_id=*/-1,
                    BlockStatus::HOST),
      RaidenBlockID(rid, /*host_block_id=*/7, /*device_block_id=*/-1,
                    BlockStatus::HOST_AND_HBM)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/true).first);

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  ASSERT_TRUE(ids_or.ok()) << ids_or.status().message();
  // Source ids are re-derived from the LRU (not from the request).
  EXPECT_THAT(*ids_or, ::testing::ElementsAre(5, 7));
  EXPECT_EQ(store.GetPinCount("h0"), 1);
  EXPECT_EQ(store.GetPinCount("h1"), 1);

  store.UnpinHostBlocks(hashes);
  EXPECT_EQ(store.GetPinCount("h0"), 0);
  EXPECT_EQ(store.GetPinCount("h1"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksMissingReturnsNotFound) {
  KVCacheStore store(4);
  auto ids_or =
      store.ValidateAndPinHostBlocks(std::vector<std::string>{"missing"});
  EXPECT_TRUE(absl::IsNotFound(ids_or.status())) << ids_or.status();
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksWrongStatusFailedPrecondition) {
  KVCacheStore store(4);
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"remote_h"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, /*host_block_id=*/-1, /*device_block_id=*/-1,
                    BlockStatus::REMOTE)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/false).first);

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  EXPECT_TRUE(absl::IsFailedPrecondition(ids_or.status())) << ids_or.status();
  EXPECT_EQ(store.GetPinCount("remote_h"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksAtomicRollbackOnPartialMiss) {
  KVCacheStore store(4);
  RaidenId rid{"src_job", "0", "src_cache", 0};
  // "ok" is HOST, "bad" is REMOTE -> the whole batch must abort and "ok" must
  // NOT remain pinned (all-or-nothing).
  std::vector<std::string> hashes = {"ok", "bad"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, /*host_block_id=*/3, /*device_block_id=*/-1,
                    BlockStatus::HOST),
      RaidenBlockID(rid, /*host_block_id=*/-1, /*device_block_id=*/-1,
                    BlockStatus::REMOTE)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/false).first);

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  EXPECT_FALSE(ids_or.ok());
  EXPECT_EQ(store.GetPinCount("ok"), 0);
  EXPECT_EQ(store.GetPinCount("bad"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksEmptyInputIsOk) {
  KVCacheStore store(4);
  auto ids_or = store.ValidateAndPinHostBlocks(std::vector<std::string>{});
  ASSERT_TRUE(ids_or.ok());
  EXPECT_TRUE(ids_or->empty());
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksIncrementsAndReleasesExistingPin) {
  KVCacheStore store(4);
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"h0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, /*host_block_id=*/9, /*device_block_id=*/-1,
                    BlockStatus::HOST)};
  // insert_and_lock pins once.
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, /*on_host=*/true));
  EXPECT_EQ(store.GetPinCount("h0"), 1);

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  ASSERT_TRUE(ids_or.ok());
  EXPECT_THAT(*ids_or, ::testing::ElementsAre(9));
  EXPECT_EQ(store.GetPinCount("h0"), 2);  // verify added a second pin.

  store.UnpinHostBlocks(hashes);
  EXPECT_EQ(store.GetPinCount("h0"), 1);  // back to the caller's pin.
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

TEST(KVCacheStoreTest, InsertAndLock) {
  KVCacheStore store(2);

  // Insert local block
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Execute InsertAndLock
  std::vector<RaidenBlockID> slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"remote_job", "0", "kv_cache", 42}};
  auto res = store.InsertAndLock({"local_1", "remote_1"}, slices, true);
  EXPECT_TRUE(res);
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);

  // Since capacity is 2 and both local_1 and remote_1 are pinned, available
  // space is 0. Attempting to InsertAndLock remote_2 should fail due to lack
  // of space.
  auto res_fail = store.InsertAndLock({"remote_2"}, {}, true);
  EXPECT_FALSE(res_fail);
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

  // Now InsertAndLock two remote blocks, which will evict local_1 and local_2.
  std::vector<std::string> remote_hashes = {"remote_1", "remote_2"};
  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::REMOTE),
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::REMOTE)};
  auto res = store.InsertAndLock(remote_hashes, remote_slices, true);
  ASSERT_TRUE(res);
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);
  EXPECT_EQ(store.GetPinCount("remote_2"), 1);
  EXPECT_EQ(store.Lookup({"local_1"})->size(), 0);
  EXPECT_EQ(store.Lookup({"local_2"})->size(), 0);

  // Now call ReleaseAndDelete to revert InsertAndLock!
  auto release_res = store.ReleaseAndDelete(remote_hashes);
  EXPECT_EQ(release_res, 2);

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
  store.InsertAndLock({"local_1"}, {local_slices[0]}, true);
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  auto res_non_remote = store.ReleaseAndDelete({"local_1"});
  EXPECT_EQ(res_non_remote, 0);
  EXPECT_EQ(store.GetPinCount("local_1"), 0);
  EXPECT_EQ(store.Lookup({"local_1"})->size(), 1);

  // Test remote block pinned twice: after one ReleaseAndDelete, pin count is 1
  // so it should NOT be deleted!
  store.InsertAndLock({"remote_1"}, {remote_slices[0]}, true);
  store.Pin({"remote_1"});  // pin count is now 2
  EXPECT_EQ(store.GetPinCount("remote_1"), 2);
  auto res_pinned = store.ReleaseAndDelete({"remote_1"});
  EXPECT_EQ(res_pinned, 0);  // 0 deleted because pin count was 2 -> 1
  EXPECT_EQ(store.GetPinCount("remote_1"), 1);
  EXPECT_EQ(store.Lookup({"remote_1"})->size(), 1);
  store.Release({"remote_1"});
  store.Delete({"remote_1"}, {remote_slices[0]});

  // Test partial restore: candidate list has 2 entries, but we only delete 1
  // block in ReleaseAndDelete. It should restore only the last candidate. Cache
  // capacity is 2. Currently local_1 and local_2 are in cache. Insert local_3
  // -> evicts local_1 (candidates: local_1).
  std::vector<std::string> local_hash_3 = {"local_3"};
  std::vector<RaidenBlockID> local_slice_3 = {RaidenBlockID(
      RaidenId{"local_job", "0", "kv_cache", 2}, -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hash_3, local_slice_3, true).first);

  // Insert local_4 -> evicts local_2 (candidates: local_1, local_2).
  std::vector<std::string> local_hash_4 = {"local_4"};
  std::vector<RaidenBlockID> local_slice_4 = {RaidenBlockID(
      RaidenId{"local_job", "0", "kv_cache", 3}, -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hash_4, local_slice_4, true).first);

  // Now InsertAndLock remote_2 -> evicts local_3 (candidates: local_1, local_2,
  // local_3). Cache has local_4, remote_2.
  ASSERT_TRUE(store.InsertAndLock({"remote_2"}, {remote_slices[1]}, true));

  // ReleaseAndDelete remote_2 -> deletes remote_2 and restores local_3.
  auto res_partial = store.ReleaseAndDelete({"remote_2"});
  EXPECT_EQ(res_partial, 1);
  // Candidates list should now contain local_2, local_1.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("local_2", "local_1"));
}

TEST(KVCacheStoreTest, RollbackRescue) {
  KVCacheStore store(3);

  // 1. Insert 3 local blocks to fill the cache (HOST status)
  std::vector<std::string> local_hashes = {"local_1", "local_2", "local_3"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::HOST),
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::HOST),
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 2}, -1,
                    BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // 2. Insert local_4 -> evicts local_1
  std::vector<std::string> local_hash_4 = {"local_4"};
  std::vector<RaidenBlockID> local_slice_4 = {RaidenBlockID(
      RaidenId{"local_job", "0", "kv_cache", 3}, -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hash_4, local_slice_4, true).first);

  // Candidates list should contain "local_1"
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("local_1"));

  // 3. Delete local_4 to free up a slot (so cache has local_2, local_3)
  store.Delete(local_hash_4, local_slice_4);

  // 4. InsertAndLock: local_2 (existing), remote_1 (new), remote_2 (new)
  // This will use the free slot for remote_1 (0 evictions), and evict local_3
  // for remote_2 (1 eviction)
  std::vector<std::string> insert_hashes = {"local_2", "remote_1", "remote_2"};
  std::vector<RaidenBlockID> insert_slices = {
      RaidenBlockID(RaidenId{"local_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::HOST),
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 0}, -1,
                    BlockStatus::REMOTE),
      RaidenBlockID(RaidenId{"remote_job", "0", "kv_cache", 1}, -1,
                    BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(insert_hashes, insert_slices, true));

  // Verify eviction count was 1 (so candidates should be local_1, local_3)
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("local_1", "local_3"));

  // 5. ReleaseAndDelete: should only restore 1 block (local_3) and not restore
  // local_1
  auto release_res = store.ReleaseAndDelete(insert_hashes);
  EXPECT_EQ(
      release_res,
      2);  // remote_1 and remote_2 are deleted since they are REMOTE status

  // local_3 should be restored to the cache, local_1 should still be in
  // candidates
  auto lookup_res = store.Lookup({"local_3"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);

  auto lookup_local_1 = store.Lookup({"local_1"});
  ASSERT_TRUE(lookup_local_1.ok());
  EXPECT_EQ(lookup_local_1->size(), 0);

  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("local_1"));
}

TEST(KVCacheStoreTest, EvictRaceCondition) {
  KVCacheStore store(3);

  // Insert local_1 (HOST status)
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockID> local_slices = {RaidenBlockID(
      RaidenId{"local_job", "0", "kv_cache", 0}, -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Pin local_1
  ASSERT_TRUE(store.Pin({"local_1"}));
  EXPECT_EQ(store.GetPinCount("local_1"), 1);

  // Attempt Evict on local_1 (which is pinned)
  size_t evicted = KVCacheStoreTest::Evict(store, {"local_1"});
  EXPECT_EQ(evicted, 0);

  // Verify local_1 is still in the cache and pinned
  EXPECT_EQ(store.GetPinCount("local_1"), 1);
  auto lookup_res = store.Lookup({"local_1"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
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
    auto resolve_or = controller.ResolvePeerController(unit_);
    ASSERT_TRUE(resolve_or.ok());
    std::string server_address = *resolve_or;
    ::tpu_raiden::core::controller::RaidenControllerClient client(
        server_address);
    auto status = client.RegisterWorker(worker_id, worker_address,
                                        {{worker_address, {}}});
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
          unit_, 10, 1, 512, orchestrator_address_, "");
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
          unit_, 10, 1, 512, orchestrator_address_, "");
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
          unit_, 10, 1, 512, orchestrator_address_, "");

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
          unit_, 10, 1, 512, orchestrator_address_, "");

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
          unit_, 10, 1, 512, orchestrator_address_, "");
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

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictByHashesHostAndHbmToErased) {
  // 1. Start a local registry server
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
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // Allocate 2 block IDs from controller so we have host_block_ids
  auto alloc_or = controller->AllocateBlockIds(2);
  ASSERT_TRUE(alloc_or.ok());
  std::vector<int> host_block_ids = *alloc_or;
  ASSERT_EQ(host_block_ids.size(), 2);

  // 3. Initialize KVCacheStore with the registry server address & controller
  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), server_address, rid);

  // Register in global registry first to simulate write-through
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  ASSERT_TRUE(registry_client
                  .Register({{"hash_1", rid, host_block_ids[0]},
                             {"hash_2", rid, host_block_ids[1]}})
                  .ok());

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, host_block_ids[0], 0, BlockStatus::HOST_AND_HBM),
      RaidenBlockID(rid, host_block_ids[1], 1, BlockStatus::HOST_AND_HBM)};

  // 4. Insert them as HOST_AND_HBM blocks locally
  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  // Sanity check: verify they are lookable before evict
  {
    auto lookup_res = store.Lookup({"hash_1", "hash_2"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);
  }

  // 5. Check locked blocks on controller
  auto* controller_ptr = KVCacheStoreTest::GetController(store);
  ASSERT_NE(controller_ptr, nullptr);
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 2);

  // 6. Evict "hash_1"
  size_t evicted = KVCacheStoreTest::Evict(store, {"hash_1"});
  EXPECT_EQ(evicted, 1);

  // 7. Verify "hash_1" is erased and "hash_2" is unchanged
  {
    auto lookup_res = store.Lookup({"hash_1"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 0);
  }
  {
    auto lookup_res = store.Lookup({"hash_2"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "hash_2");
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
    EXPECT_EQ((*lookup_res)[0].second.host_block_id, host_block_ids[1]);
    EXPECT_EQ((*lookup_res)[0].second.device_block_id, 1);
  }

  // 8. Verify controller block manager has 1 locked block now
  // (host_block_ids[0] unlocked)
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 1);

  // 9. Verify global registry has unregistered "hash_1" (need to poll)
  bool unregistered = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup({"hash_1"});
    if (lookup_res.ok() && lookup_res->empty()) {
      unregistered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(unregistered);

  server->Shutdown();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictByHashesHostToErased) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  auto alloc_or = controller->AllocateBlockIds(2);
  ASSERT_TRUE(alloc_or.ok());
  std::vector<int> host_block_ids = *alloc_or;

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), server_address, rid);

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);
  ASSERT_TRUE(registry_client
                  .Register({{"hash_1", rid, host_block_ids[0]},
                             {"hash_2", rid, host_block_ids[1]}})
                  .ok());

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, host_block_ids[0], -1, BlockStatus::HOST),
      RaidenBlockID(rid, host_block_ids[1], -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  auto* controller_ptr = KVCacheStoreTest::GetController(store);
  ASSERT_NE(controller_ptr, nullptr);
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 2);

  // Evict "hash_1"
  size_t evicted = KVCacheStoreTest::Evict(store, {"hash_1"});
  EXPECT_EQ(evicted, 1);

  // Verify "hash_1" is completely erased, but "hash_2" is still there
  // Since Lookup stops at first miss, Lookup({"hash_1", "hash_2"}) should
  // return 0 items. Lookup({"hash_2"}) should return 1 item.
  EXPECT_EQ(store.Lookup({"hash_1", "hash_2"})->size(), 0);
  auto lookup_res = store.Lookup({"hash_2"});
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].first, "hash_2");

  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 1);

  bool unregistered = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup({"hash_1"});
    if (lookup_res.ok() && lookup_res->empty()) {
      unregistered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(unregistered);

  server->Shutdown();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictOnSave) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 2, 1, 512, orchestrator_address_, "");
  auto* controller_ptr = controller.get();
  RegisterAndInitWorker(*controller_ptr, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(3, std::move(controller), server_address, rid);

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  auto alloc_or = controller_ptr->AllocateBlockIds(2);
  ASSERT_TRUE(alloc_or.ok());
  std::vector<int> host_block_ids = *alloc_or;
  ASSERT_EQ(host_block_ids.size(), 2);

  ASSERT_TRUE(registry_client
                  .Register({{"block_A", rid, host_block_ids[0]},
                             {"block_B", rid, host_block_ids[1]}})
                  .ok());

  std::vector<std::string> hashes = {"block_A", "block_B"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, host_block_ids[0], -1, BlockStatus::HOST),
      RaidenBlockID(rid, host_block_ids[1], -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  std::vector<std::string> hashes_C = {"block_C"};
  std::vector<RaidenBlockID> slices_C = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM)};
  ASSERT_TRUE(store.Insert(hashes_C, slices_C, false).first);
  ASSERT_TRUE(store.Pin(hashes_C));

  EXPECT_EQ(controller_ptr->block_manager()->num_free_blocks(), 0);

  absl::Status status = store.Save(hashes_C);
  ASSERT_TRUE(status.ok()) << status.message();

  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending] = store.PollSaveStatus();
    if (!save_failed.empty()) {
      FAIL() << "Async Save failed during polling";
    }
    if (!save_done.empty()) {
      EXPECT_THAT(save_done, ::testing::ElementsAre("block_C"));
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  EXPECT_EQ(store.Lookup({"block_A"})->size(), 0);

  auto lookup_res = store.Lookup({"block_C"});
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, host_block_ids[0]);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 0);

  bool unregistered_A = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup({"block_A"});
    if (lookup_res.ok() && lookup_res->empty()) {
      unregistered_A = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(unregistered_A);

  server->Shutdown();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ProactiveEvictionWithCandidates) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  // Capacity is 2
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 2, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(2, std::move(controller), "", rid);

  std::vector<std::string> hashes = {"hash_A", "hash_B"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockID(rid, -1, 1, BlockStatus::HBM)};

  // 1. Insert A and B as HBM blocks
  ASSERT_TRUE(store.Insert(hashes, slices, false).first);

  // 2. Save A and B (allocates host blocks for both)
  ASSERT_TRUE(store.Pin(hashes));
  ASSERT_TRUE(store.Save(hashes).ok());

  // Poll for Save completion
  bool save_done = false;
  while (!save_done) {
    auto [done, failed, pending] = store.PollSaveStatus();
    ASSERT_TRUE(failed.empty());
    if (!done.empty()) {
      save_done = true;
    } else {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }
  store.Release(hashes);

  // Verify both are HOST_AND_HBM
  {
    auto lookup_res = store.Lookup({"hash_A", "hash_B"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
    EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST_AND_HBM);
  }

  // Active LRU: A, B (A is MRU, B is LRU).

  // 3. Insert C (HBM block). This exceeds store capacity (2) and evicts B.
  std::vector<std::string> hash_C = {"hash_C"};
  std::vector<RaidenBlockID> slice_C = {
      RaidenBlockID(rid, -1, 2, BlockStatus::HBM)};
  ASSERT_TRUE(store.Insert(hash_C, slice_C, false).first);

  // B should now be in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_B"));

  // 4. Access B (using Lookup). This should miss because B is candidate and
  // lookup uses Peek.
  {
    auto lookup_res = store.Lookup({"hash_B"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 0);
  }

  // B is still in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_B"));

  // Active LRU: C, A (C is MRU, A is LRU).
  // 5. Insert D (HBM block). This exceeds capacity and evicts A (since A is
  // LRU).
  std::vector<std::string> hash_D = {"hash_D"};
  std::vector<RaidenBlockID> slice_D = {
      RaidenBlockID(rid, -1, 3, BlockStatus::HBM)};
  ASSERT_TRUE(store.Insert(hash_D, slice_D, false).first);

  // Candidates list should now contain B, then A.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_B", "hash_A"));

  // 6. Save D. Requires 1 host block.
  // Controller free host blocks: 0 (used by A and B).
  // It should pick candidate B for eviction and deallocate its host block.
  // A (candidate HOST_AND_HBM) should not be affected.
  ASSERT_TRUE(store.Pin(hash_D));
  ASSERT_TRUE(store.Save(hash_D).ok());

  // Poll for Save completion
  save_done = false;
  while (!save_done) {
    auto [done, failed, pending] = store.PollSaveStatus();
    ASSERT_TRUE(failed.empty());
    if (!done.empty()) {
      save_done = true;
    } else {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }
  store.Release(hash_D);

  // 7. Verify states:
  // - B should be erased (since it was HOST_AND_HBM and got evicted)
  // - A should remain in candidates (HOST_AND_HBM)
  // - D should be HOST_AND_HBM
  {
    auto lookup_res = store.Lookup({"hash_B"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 0);
  }
  {
    // A should miss because it is still in candidate list.
    auto lookup_res = store.Lookup({"hash_A"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 0);
  }
  {
    auto lookup_res = store.Lookup({"hash_D"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST_AND_HBM);
    EXPECT_NE((*lookup_res)[0].second.host_block_id, -1);
  }
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_A"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteSuccess) {
  // 1. Start a local mock registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder registry_builder;
  int registry_port = 0;
  registry_builder.AddListeningPort(
      "localhost:0", grpc::InsecureServerCredentials(), &registry_port);
  registry_builder.RegisterService(service.get());
  auto registry_server = registry_builder.BuildAndStart();
  std::string registry_address = "localhost:" + std::to_string(registry_port);

  // 2. Start src controller server
  auto src_controller_server = core::controller::CreateTestControllerServer();

  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id;
  src_raiden_id.job_name = "src_job";
  src_raiden_id.job_replica_id = "0";
  src_raiden_id.data_name = "src_data";
  src_raiden_id.data_replica_idx = 0;

  // Register src controller with orchestrator
  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  auto register_status = orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address);
  ASSERT_TRUE(register_status.ok()) << register_status.message();

  // Setup src worker registration on src controller
  auto register_src_worker = [&](const std::string& worker_id,
                                 const std::string& worker_address,
                                 const std::string& transfer_endpoint) {
    auto status = src_controller_server->client->RegisterWorker(
        worker_id, worker_address, {{transfer_endpoint, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  register_src_worker("worker_0", "src_worker_0_addr", "src_worker_0_transfer");

  // Setup src controller's transfer callback to simulate successful H2H
  std::atomic<bool> callback_triggered = false;
  src_controller_server->service->SetTransferBuffersCallback(
      [&](absl::Span<const Buffer> src_buffers,
          absl::Span<const Buffer> dst_buffers) {
        callback_triggered = true;
        std::vector<int64_t> src_offsets;
        for (const auto& buf : src_buffers) {
          EXPECT_EQ(buf.memory_type(), rpc::MemoryType::MEMORY_TYPE_DRAM);
          src_offsets.push_back(buf.index());
        }
        EXPECT_THAT(src_offsets, ::testing::ElementsAre(42));

        std::vector<int64_t> dst_offsets;
        std::vector<std::string> peer_addrs;
        for (const auto& buf : dst_buffers) {
          EXPECT_EQ(buf.memory_type(), rpc::MemoryType::MEMORY_TYPE_DRAM);
          dst_offsets.push_back(buf.index());
          // ReadRemote carries the destination peer workers as per-worker
          // endpoint groups (each tagged with node_id); the source narrows them
          // to the node_id-matched worker.
          for (const auto& group : buf.remote_worker_endpoints()) {
            for (const auto& p : group.endpoints) {
              peer_addrs.push_back(p.endpoint);
            }
          }
        }
        EXPECT_THAT(dst_offsets, ::testing::ElementsAre(
                                     0));  // allocated local host_block_id
        EXPECT_THAT(peer_addrs,
                    ::testing::ElementsAre(test_server_->server_address));
        return tsl::Future<>(absl::OkStatus());
      });

  // Setup dest controller and KVCacheStore
  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(10, std::move(dest_controller), registry_address, rid);

  // Insert and pin remote block in local store
  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // Trigger ReadRemote
  absl::Status status = store.ReadRemote(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(failed_hashes.empty());
    if (!done_hashes.empty()) {
      EXPECT_THAT(done_hashes, ::testing::ElementsAre("hash_0"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);
  EXPECT_TRUE(callback_triggered);

  // Verify status in LRU is HOST, host_block_id is 0
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 0);

  // Verify registration in global registry (need to poll registry since
  // registration is async)
  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  global_registry::GlobalRegistryClient registry_client(channel);

  bool registered = false;
  std::vector<global_registry::KVBlockMetadata> metadata_results;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto lookup_res = registry_client.Lookup(hashes);
    if (lookup_res.ok() && lookup_res->size() == 1) {
      metadata_results = *std::move(lookup_res);
      registered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(registered)
      << "Block hashes were not registered in global registry";

  EXPECT_EQ(metadata_results[0].raiden_id().job_name(), rid.job_name);
  EXPECT_EQ(metadata_results[0].block_id(), 0);

  registry_server->Shutdown();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteFailure) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder registry_builder;
  int registry_port = 0;
  registry_builder.AddListeningPort(
      "localhost:0", grpc::InsecureServerCredentials(), &registry_port);
  registry_builder.RegisterService(service.get());
  auto registry_server = registry_builder.BuildAndStart();
  std::string registry_address = "localhost:" + std::to_string(registry_port);

  auto src_controller_server = core::controller::CreateTestControllerServer();

  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id;
  src_raiden_id.job_name = "src_job";
  src_raiden_id.job_replica_id = "0";
  src_raiden_id.data_name = "src_data";
  src_raiden_id.data_replica_idx = 0;

  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  auto register_status = orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address);
  ASSERT_TRUE(register_status.ok()) << register_status.message();

  auto register_src_worker = [&](const std::string& worker_id,
                                 const std::string& worker_address,
                                 const std::string& transfer_endpoint) {
    auto status = src_controller_server->client->RegisterWorker(
        worker_id, worker_address, {{transfer_endpoint, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  register_src_worker("worker_0", "src_worker_0_addr", "src_worker_0_transfer");

  // Setup src controller to fail
  src_controller_server->service->SetTransferBuffersCallback(
      [&](absl::Span<const Buffer> src_buffers,
          absl::Span<const Buffer> dst_buffers) {
        return tsl::Future<>(absl::InternalError("H2H Transfer Failed"));
      });

  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(2, std::move(dest_controller), registry_address, rid);

  // Fill cache with two local blocks
  std::vector<std::string> local_hashes = {"local_1", "local_2"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenBlockID(rid, -1, BlockStatus::HOST),
      RaidenBlockID(rid, -1, BlockStatus::HOST)};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Unpin local_1 so it is evictable
  store.Release({"local_1"});

  // Insert and pin remote block (evicts local_1)
  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // Trigger ReadRemote
  absl::Status status = store.ReadRemote(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for failure
  bool failed = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(done_hashes.empty());
    if (!failed_hashes.empty()) {
      EXPECT_THAT(failed_hashes, ::testing::ElementsAre("hash_0"));
      failed = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);

  // Verify hash_0 is still REMOTE
  {
    auto lookup_res = store.Lookup(hashes);
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);
  }

  // Caller calls ReleaseAndDelete to clean up failed remote read
  size_t deleted = store.ReleaseAndDelete(hashes);
  EXPECT_EQ(deleted, 1);

  // Verify hash_0 is deleted, local_1 is restored
  {
    auto lookup_res = store.Lookup(hashes);
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_EQ(lookup_res->size(), 0);
  }
  {
    auto lookup_res = store.Lookup({"local_1"});
    ASSERT_TRUE(lookup_res.ok());
    EXPECT_EQ(lookup_res->size(), 1);
  }

  registry_server->Shutdown();
}

// ReadRemote step 6a end-to-end at the store level: the source controller's
// verify hook rejects the requested hash -> the destination read fails and its
// pre-allocated host block is reverted (via PollRemoteReadsInternal).
TEST_F(KVCacheStoreEmbeddedControllerTest,
       ReadRemoteSourceVerifyMissingRevertsDestination) {
  auto src_controller_server = core::controller::CreateTestControllerServer();
  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);
  kv_cache::RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  ASSERT_TRUE(orchestrator_client
                  .RegisterController(src_unit,
                                      src_controller_server->server_address)
                  .ok());

  std::vector<std::string> validated;
  bool transfer_ran = false;
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        validated.assign(h.begin(), h.end());
        return absl::NotFoundError("BLOCK_HASH_NOT_FOUND: h");
      },
      [&](absl::Span<const std::string> /*h*/) {});
  src_controller_server->service->SetTransferBuffersCallback(
      [&](absl::Span<const Buffer> /*s*/, absl::Span<const Buffer> /*d*/) {
        transfer_ran = true;
        return tsl::Future<>(absl::OkStatus());
      });

  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);
  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(2, std::move(dest_controller), "", rid);

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  ASSERT_TRUE(store.ReadRemote(hashes).ok());

  bool failed = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(done_hashes.empty());
    if (!failed_hashes.empty()) {
      EXPECT_THAT(failed_hashes, ::testing::ElementsAre("hash_0"));
      failed = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(failed);
  // The block_hash flowed to the source and the transfer was never dispatched.
  EXPECT_THAT(validated, ::testing::ElementsAre("hash_0"));
  EXPECT_FALSE(transfer_ran);
  // The block is still REMOTE on the destination (not promoted to HOST).
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::REMOTE);
}

// The source verify hook accepts the hash -> the transfer runs and the
// destination block is promoted to HOST. Confirms the block_hashes reach the
// source verify path on the success flow.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       ReadRemoteSourceVerifySuccessTransfers) {
  auto src_controller_server = core::controller::CreateTestControllerServer();
  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);
  kv_cache::RaidenId src_raiden_id{"src_job", "0", "src_data", 0};
  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  ASSERT_TRUE(orchestrator_client
                  .RegisterController(src_unit,
                                      src_controller_server->server_address)
                  .ok());

  std::vector<std::string> validated, unpinned;
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        validated.assign(h.begin(), h.end());
        return std::vector<int32_t>{42};  // authoritative source id
      },
      [&](absl::Span<const std::string> h) {
        unpinned.assign(h.begin(), h.end());
      });
  src_controller_server->service->SetTransferBuffersCallback(
      [&](absl::Span<const Buffer> /*s*/, absl::Span<const Buffer> /*d*/) {
        return tsl::Future<>(absl::OkStatus());
      });

  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);
  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(2, std::move(dest_controller), "", rid);

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  ASSERT_TRUE(store.ReadRemote(hashes).ok());

  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(failed_hashes.empty());
    if (!done_hashes.empty()) {
      EXPECT_THAT(done_hashes, ::testing::ElementsAre("hash_0"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);
  EXPECT_THAT(validated, ::testing::ElementsAre("hash_0"));
  EXPECT_THAT(unpinned, ::testing::ElementsAre("hash_0"));
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteDuplicateFails) {
  auto src_controller_server = core::controller::CreateTestControllerServer();

  rpc::RaidenIdProto src_unit;
  src_unit.set_job_name("src_job");
  src_unit.set_job_replica_id("0");
  src_unit.set_data_name("src_data");
  src_unit.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id;
  src_raiden_id.job_name = "src_job";
  src_raiden_id.job_replica_id = "0";
  src_raiden_id.data_name = "src_data";
  src_raiden_id.data_replica_idx = 0;

  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  auto register_status = orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address);
  ASSERT_TRUE(register_status.ok()) << register_status.message();

  auto register_src_worker = [&](const std::string& worker_id,
                                 const std::string& worker_address,
                                 const std::string& transfer_endpoint) {
    auto status = src_controller_server->client->RegisterWorker(
        worker_id, worker_address, {{transfer_endpoint, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  };
  register_src_worker("worker_0", "src_worker_0_addr", "src_worker_0_transfer");

  // Keep transfer pending by not fulfilling the promise
  auto promise_and_future = tsl::MakePromise();
  auto& promise = promise_and_future.first;
  src_controller_server->service->SetTransferBuffersCallback(
      [f = promise_and_future.second](absl::Span<const Buffer> src_buffers,
                                      absl::Span<const Buffer> dst_buffers) {
        return f;
      });

  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(10, std::move(dest_controller), "", rid);

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // First call succeeds
  absl::Status status1 = store.ReadRemote(hashes);
  ASSERT_TRUE(status1.ok()) << status1.message();

  // Second call fails with FailedPreconditionError
  absl::Status status2 = store.ReadRemote(hashes);
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status2.message(), ::testing::HasSubstr("already reading"));

  // Fulfill promise to clean up
  promise.Set(absl::OkStatus());
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteAllocationFailureAborts) {
  // 1. Start a local mock registry server
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder registry_builder;
  int registry_port = 0;
  registry_builder.AddListeningPort(
      "localhost:0", grpc::InsecureServerCredentials(), &registry_port);
  registry_builder.RegisterService(service.get());
  auto registry_server = registry_builder.BuildAndStart();
  std::string registry_address = "localhost:" + std::to_string(registry_port);

  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 1, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(2, std::move(dest_controller), registry_address, rid);

  // Insert and pin local_1 (HBM status, device_block_id = 0)
  std::vector<std::string> local_hashes = {"local_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM)};
  ASSERT_TRUE(store.InsertAndLock(local_hashes, local_slices, true));

  // Trigger Save on local_1 to allocate its host block
  absl::Status save_status = store.Save(local_hashes);
  ASSERT_TRUE(save_status.ok()) << save_status.message();

  // Poll for Save completion
  bool save_done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] = store.PollSaveStatus();
    ASSERT_TRUE(failed_hashes.empty());
    if (!done_hashes.empty()) {
      save_done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(save_done);

  // Now, 1 host block is allocated, free = 0. And local_1 remains pinned.
  // Insert and pin remote block hash_0
  kv_cache::RaidenId src_raiden_id{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // ReadRemote should fail because allocation of host block fails (0 free, 0
  // evictable)
  absl::Status status = store.ReadRemote(hashes);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted);

  // Verify hash_0 is NOT in reading_hashes_ (so calling it again doesn't report
  // duplicate)
  absl::Status status2 = store.ReadRemote(hashes);
  EXPECT_EQ(status2.code(), absl::StatusCode::kResourceExhausted);

  registry_server->Shutdown();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, ReadRemoteMultipleSources) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder registry_builder;
  int registry_port = 0;
  registry_builder.AddListeningPort(
      "localhost:0", grpc::InsecureServerCredentials(), &registry_port);
  registry_builder.RegisterService(service.get());
  auto registry_server = registry_builder.BuildAndStart();
  std::string registry_address = "localhost:" + std::to_string(registry_port);

  // 1. Start two source controller servers
  auto src_controller_server_1 = core::controller::CreateTestControllerServer();
  auto src_controller_server_2 = core::controller::CreateTestControllerServer();

  rpc::RaidenIdProto src_unit_1;
  src_unit_1.set_job_name("src_job_1");
  src_unit_1.set_job_replica_id("0");
  src_unit_1.set_data_name("src_data_1");
  src_unit_1.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id_1;
  src_raiden_id_1.job_name = "src_job_1";
  src_raiden_id_1.job_replica_id = "0";
  src_raiden_id_1.data_name = "src_data_1";
  src_raiden_id_1.data_replica_idx = 0;

  rpc::RaidenIdProto src_unit_2;
  src_unit_2.set_job_name("src_job_2");
  src_unit_2.set_job_replica_id("0");
  src_unit_2.set_data_name("src_data_2");
  src_unit_2.set_data_replica_idx(0);

  kv_cache::RaidenId src_raiden_id_2;
  src_raiden_id_2.job_name = "src_job_2";
  src_raiden_id_2.job_replica_id = "0";
  src_raiden_id_2.data_name = "src_data_2";
  src_raiden_id_2.data_replica_idx = 0;

  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  ASSERT_TRUE(orchestrator_client
                  .RegisterController(src_unit_1,
                                      src_controller_server_1->server_address)
                  .ok());
  ASSERT_TRUE(orchestrator_client
                  .RegisterController(src_unit_2,
                                      src_controller_server_2->server_address)
                  .ok());

  // Register worker on each source controller
  ASSERT_TRUE(src_controller_server_1->client
                  ->RegisterWorker("worker_0", "src_worker_1_addr",
                                   {{"src_worker_1_transfer", {}}})
                  .ok());
  ASSERT_TRUE(src_controller_server_2->client
                  ->RegisterWorker("worker_0", "src_worker_2_addr",
                                   {{"src_worker_2_transfer", {}}})
                  .ok());

  // Setup callbacks with promises to control completion
  std::atomic<bool> callback_1_triggered = false;
  auto promise_and_future_1 = tsl::MakePromise();
  auto& promise1 = promise_and_future_1.first;
  src_controller_server_1->service->SetTransferBuffersCallback(
      [f = promise_and_future_1.second, &callback_1_triggered](
          absl::Span<const Buffer> src_buffers,
          absl::Span<const Buffer> dst_buffers) {
        callback_1_triggered = true;
        std::vector<int64_t> src_offsets;
        for (const auto& buf : src_buffers) {
          src_offsets.push_back(buf.index());
        }
        EXPECT_THAT(src_offsets, ::testing::ElementsAre(10));
        return f;
      });

  std::atomic<bool> callback_2_triggered = false;
  auto promise_and_future_2 = tsl::MakePromise();
  auto& promise2 = promise_and_future_2.first;
  src_controller_server_2->service->SetTransferBuffersCallback(
      [f = promise_and_future_2.second, &callback_2_triggered](
          absl::Span<const Buffer> src_buffers,
          absl::Span<const Buffer> dst_buffers) {
        callback_2_triggered = true;
        std::vector<int64_t> src_offsets;
        for (const auto& buf : src_buffers) {
          src_offsets.push_back(buf.index());
        }
        EXPECT_THAT(src_offsets, ::testing::ElementsAre(20));
        return f;
      });

  auto dest_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dest_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dest_job", "0", "dest_cache", 0};
  KVCacheStore store(10, std::move(dest_controller), registry_address, rid);

  // Insert and pin remote block hash_0 and hash_1
  std::vector<std::string> hashes = {"hash_0", "hash_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id_1, 10, BlockStatus::REMOTE),
      RaidenBlockID(src_raiden_id_2, 20, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // Trigger ReadRemote for both
  absl::Status status = store.ReadRemote(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Wait for both callbacks to be triggered asynchronously
  bool callbacks_triggered = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    if (callback_1_triggered && callback_2_triggered) {
      callbacks_triggered = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(callbacks_triggered) << "Callbacks were not triggered in time";

  // Fulfill first promise
  promise1.Set(absl::OkStatus());

  // Poll: neither should be done since promise2 is pending
  absl::SleepFor(absl::Milliseconds(50));
  auto [done_hashes1, failed_hashes1, pending_hashes1] =
      store.PollRemoteReadStatus();
  EXPECT_TRUE(done_hashes1.empty());
  EXPECT_TRUE(failed_hashes1.empty());
  EXPECT_THAT(pending_hashes1,
              ::testing::UnorderedElementsAre("hash_0", "hash_1"));

  // Fulfill second promise
  promise2.Set(absl::OkStatus());

  // Poll: now both should be done
  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(failed_hashes.empty());
    if (!done_hashes.empty()) {
      EXPECT_THAT(done_hashes,
                  ::testing::UnorderedElementsAre("hash_0", "hash_1"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // Verify both statuses are HOST
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup_res)[1].second.status, BlockStatus::HOST);

  registry_server->Shutdown();
}

// In-process registry plus client, for RecoverFromRegistry tests.
struct RecoveryRegistrySetup {
  RecoveryRegistrySetup() {
    service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(service.get());
    server = builder.BuildAndStart();
    address = absl::StrCat("localhost:", port);
    client = std::make_unique<global_registry::GlobalRegistryClient>(
        grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
  }
  ~RecoveryRegistrySetup() { server->Shutdown(); }

  std::unique_ptr<global_registry::GlobalRegistryServiceImpl> service;
  std::unique_ptr<grpc::Server> server;
  std::string address;
  std::unique_ptr<global_registry::GlobalRegistryClient> client;
};

// Worker-less controller: sufficient for recovery, which only touches the
// logical block manager.
std::unique_ptr<::tpu_raiden::controller::RaidenController>
MakeRecoveryController(const RaidenId& rid, int num_blocks) {
  rpc::RaidenIdProto unit;
  unit.set_job_name(rid.job_name);
  unit.set_job_replica_id(rid.job_replica_id);
  unit.set_data_name(rid.data_name);
  unit.set_data_replica_idx(rid.data_replica_idx);
  return std::make_unique<::tpu_raiden::controller::RaidenController>(
      unit, num_blocks, /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*raiden_orchestrator_address=*/"", /*raiden_controller_address=*/"");
}

TEST(KVCacheStoreTest, RecoverFromRegistryRebuildsDirectory) {
  RecoveryRegistrySetup registry;
  RaidenId rid{"recover_job", "0", "kv_cache", 0};

  // Entries left behind by the previous incarnation of this owner, plus one
  // foreign entry that must not be recovered.
  ASSERT_TRUE(registry.client
                  ->Register({{"rh1", rid, 5, absl::Seconds(60)},
                              {"rh2", rid, 7, absl::Seconds(600)},
                              {"rh3", rid, 9, absl::Seconds(6000)}})
                  .ok());
  RaidenId other{"other_job", "0", "kv_cache", 0};
  ASSERT_TRUE(
      registry.client->Register({{"oh", other, 3, absl::Seconds(600)}}).ok());

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  KVCacheStore store(10, std::move(controller), registry.address, rid);

  auto recovered_or = store.RecoverFromRegistry();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 3);

  auto lookup = store.Lookup({"rh1", "rh2", "rh3"});
  ASSERT_TRUE(lookup.ok());
  ASSERT_EQ(lookup->size(), 3);
  EXPECT_EQ((*lookup)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup)[0].second.host_block_id, 5);
  EXPECT_EQ((*lookup)[1].second.host_block_id, 7);
  EXPECT_EQ((*lookup)[2].second.host_block_id, 9);
  EXPECT_EQ(store.Lookup({"oh"})->size(), 0);

  // Recovered blocks are allocated and locked; new allocations avoid them.
  auto* block_manager = controller_ptr->block_manager();
  for (int id : {5, 7, 9}) {
    EXPECT_TRUE(block_manager->IsAllocated(id));
    EXPECT_TRUE(block_manager->IsLocked(id));
  }
  auto alloc_or = controller_ptr->AllocateBlockIds(7);
  ASSERT_TRUE(alloc_or.ok());
  for (int id : *alloc_or) {
    EXPECT_NE(id, 5);
    EXPECT_NE(id, 7);
    EXPECT_NE(id, 9);
  }
}

TEST(KVCacheStoreTest, RecoverFromRegistryPrefersLargestRemainingTtl) {
  RecoveryRegistrySetup registry;
  RaidenId rid{"recover_job_ttl", "0", "kv_cache", 0};
  ASSERT_TRUE(registry.client
                  ->Register({{"rh1", rid, 1, absl::Seconds(30)},
                              {"rh2", rid, 2, absl::Seconds(300)},
                              {"rh3", rid, 3, absl::Seconds(3000)}})
                  .ok());

  // Directory capacity 2 < 3 pulled entries: only the two entries with the
  // largest remaining TTL are recovered.
  KVCacheStore store(2, MakeRecoveryController(rid, 10), registry.address, rid);
  auto recovered_or = store.RecoverFromRegistry();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 2);
  EXPECT_EQ(store.Lookup({"rh1"})->size(), 0);
  EXPECT_EQ(store.Lookup({"rh2"})->size(), 1);
  EXPECT_EQ(store.Lookup({"rh3"})->size(), 1);
}

TEST(KVCacheStoreTest, RecoverFromRegistrySkipsExistingHashes) {
  RecoveryRegistrySetup registry;
  RaidenId rid{"recover_job_skip", "0", "kv_cache", 0};
  ASSERT_TRUE(registry.client
                  ->Register({{"rh1", rid, 5, absl::Seconds(600)},
                              {"rh2", rid, 7, absl::Seconds(600)}})
                  .ok());

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  KVCacheStore store(10, std::move(controller), registry.address, rid);

  // rh1 is already tracked locally (e.g. freshly computed): recovery must not
  // overwrite it, and must not restore its stale registry block ID.
  ASSERT_TRUE(store
                  .Insert({"rh1"},
                          {RaidenBlockID(rid, -1, 0, BlockStatus::HBM)},
                          /*on_host=*/false)
                  .first);

  auto recovered_or = store.RecoverFromRegistry();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 1);

  auto lookup = store.Lookup({"rh1"});
  ASSERT_EQ(lookup->size(), 1);
  EXPECT_EQ((*lookup)[0].second.status, BlockStatus::HBM);
  EXPECT_EQ((*lookup)[0].second.host_block_id, -1);
  EXPECT_FALSE(controller_ptr->block_manager()->IsAllocated(5));
}

// Only reachable through misuse: recovery must run on a fresh store, so a
// conflicting allocation means someone allocated before (or instead of)
// recovering. Verifies the failure is clean — error out, directory untouched.
TEST(KVCacheStoreTest, RecoverFromRegistryFailsOnAllocatorConflict) {
  RecoveryRegistrySetup registry;
  RaidenId rid{"recover_job_conflict", "0", "kv_cache", 0};

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  // Block 0 is already taken locally before recovery runs.
  ASSERT_TRUE(controller_ptr->AllocateBlockIds(1).ok());
  ASSERT_TRUE(
      registry.client->Register({{"rh1", rid, 0, absl::Seconds(600)}}).ok());

  KVCacheStore store(10, std::move(controller), registry.address, rid);
  auto recovered_or = store.RecoverFromRegistry();
  EXPECT_EQ(recovered_or.status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(store.Lookup({"rh1"})->size(), 0);
}

TEST(KVCacheStoreTest, RecoverFromRegistryPreconditions) {
  RaidenId rid{"recover_job_pre", "0", "kv_cache", 0};
  // No registry connection.
  KVCacheStore store_no_registry(10, MakeRecoveryController(rid, 10), "", rid);
  EXPECT_EQ(store_no_registry.RecoverFromRegistry().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Registry reachable but empty: recovery succeeds with zero blocks.
  RecoveryRegistrySetup registry;
  KVCacheStore store_empty(10, MakeRecoveryController(rid, 10),
                           registry.address, rid);
  auto recovered_or = store_empty.RecoverFromRegistry();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 0);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
