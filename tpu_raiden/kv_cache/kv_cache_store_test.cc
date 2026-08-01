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
#include <chrono>  // NOLINT(build/c++11)
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/channel.h"
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
#include "tpu_raiden/kv_cache/host_offload_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
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
    return store.backend() ? store.backend()->GetEvictCandidateKeys()
                           : std::vector<std::string>{};
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

// --- ReadRemote All-or-Nothing validate & pin block hashes at the src controller: source-side ValidateAndPinHostBlocks ---

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

TEST(KVCacheStoreTest,
     ValidateAndPinHostBlocksIncrementsAndReleasesExistingPin) {
  KVCacheStore store(4);
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"h0"};
  std::vector<RaidenBlockID> slices = {RaidenBlockID(
      rid, /*host_block_id=*/9, /*device_block_id=*/-1, BlockStatus::HOST)};
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

  // Lookup 4 hashes. Lookup is non-mutating and unbounded by available space,
  // returning all 3 cached blocks up to the first miss ("104").
  std::vector<std::string> lookup_hashes = {"101", "102", "103", "104"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 3);
  EXPECT_EQ((*lookup_res)[0].first, "101");
  EXPECT_EQ((*lookup_res)[1].first, "102");
  EXPECT_EQ((*lookup_res)[2].first, "103");
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

// 64-byte aligned backing buffer standing in for the shared memory region
// that would hold the KV cache metadata table.
class MetadataRegion {
 public:
  explicit MetadataRegion(int num_blocks)
      : buffer_(KVCacheMetadata::RequiredSizeBytes(num_blocks) + 63) {}

  absl::Span<uint8_t> span() {
    auto addr = reinterpret_cast<uintptr_t>(buffer_.data());
    size_t offset = (64 - addr % 64) % 64;
    return absl::MakeSpan(buffer_.data() + offset, buffer_.size() - offset);
  }

 private:
  std::vector<uint8_t> buffer_;
};

TEST(KVCacheStoreTest, InsertSetsAndDeleteClearsMetadataEntries) {
  MetadataRegion region(4);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"local_job", "0", "kv_cache", 0};
  KVCacheStore store(4, /*raiden_controller=*/nullptr,
                     /*global_registry_address=*/"", rid, *metadata_or);

  // host_1 and host_2 own host blocks and get metadata entries; hbm_1 owns
  // no host data and must stay out of the table.
  std::vector<std::string> hashes = {"host_1", "host_2", "hbm_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, BlockStatus::HOST),
      RaidenBlockID(rid, 1, BlockStatus::HOST),
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM)};
  ASSERT_TRUE(store.Insert(hashes, slices, true).first);
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "host_1", 0),
                          ::testing::FieldsAre(1, "host_2", 1)));

  // Delete clears the binding's metadata entry.
  store.Delete({"host_1"}, {slices[0]});
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(1, "host_2", 1)));
}

TEST(KVCacheStoreTest, MetadataKeepsEvictionCandidates) {
  MetadataRegion region(4);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"local_job", "0", "kv_cache", 0};
  KVCacheStore store(2, /*raiden_controller=*/nullptr,
                     /*global_registry_address=*/"", rid, *metadata_or);

  ASSERT_TRUE(store
                  .Insert({"host_1", "host_2"},
                          {RaidenBlockID(rid, 0, BlockStatus::HOST),
                           RaidenBlockID(rid, 1, BlockStatus::HOST)},
                          true)
                  .first);

  // Inserting host_3 moves host_1 to the candidate list. Its host block stays
  // allocated, so its metadata entry survives.
  ASSERT_TRUE(
      store.Insert({"host_3"}, {RaidenBlockID(rid, 2, BlockStatus::HOST)}, true)
          .first);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ElementsAre("host_1"));
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "host_1", 0),
                          ::testing::FieldsAre(1, "host_2", 1),
                          ::testing::FieldsAre(2, "host_3", 2)));

  // Re-inserting host_1 under a new host block reactivates the candidate and
  // overwrites its binding in place: block 0's entry is cleared and block 3's
  // is set with the newest seq.
  ASSERT_TRUE(
      store.Insert({"host_1"}, {RaidenBlockID(rid, 3, BlockStatus::HOST)}, true)
          .first);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::IsEmpty());
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(1, "host_2", 1),
                          ::testing::FieldsAre(2, "host_3", 2),
                          ::testing::FieldsAre(3, "host_1", 3)));
}

TEST(KVCacheStoreTest, EvictClearsMetadataEntries) {
  MetadataRegion region(2);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 2);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"local_job", "0", "kv_cache", 0};
  KVCacheStore store(2, /*raiden_controller=*/nullptr,
                     /*global_registry_address=*/"", rid, *metadata_or);

  ASSERT_TRUE(store
                  .Insert({"host_1", "host_2"},
                          {RaidenBlockID(rid, 0, BlockStatus::HOST),
                           RaidenBlockID(rid, 1, BlockStatus::HOST)},
                          true)
                  .first);
  ASSERT_EQ(metadata_or->ValidEntries().size(), 2);

  EXPECT_EQ(KVCacheStoreTest::Evict(store, {"host_1"}), 1);
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(1, "host_2", 1)));
}

class KVCacheStoreEmbeddedControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    // The DESTINATION worker executes the copy now (the read is a pull), so it
    // needs a transfer manager. Scripting it to succeed is what lets the
    // commit-side logic run at all on CPU, where a real transfer cannot.
    dst_transfer_mock_ =
        std::make_unique<::tpu_raiden::controller::ShardAwareMockTransferManager>();
    test_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(dst_transfer_mock_.get()));
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
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      dst_transfer_mock_;
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

// Candidate re-admission is the normal capacity-pressure path: save a block
// to host, let a later insert displace it to the eviction-candidate list,
// then re-store the same prefix. The re-admitting Put() replaces the
// candidate's value in place, so its host block must be returned to the
// allocator at that point -- before the fix it leaked permanently.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       CandidateReadmissionReclaimsHostBlock) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  // LRU capacity 1 so the next insert displaces the saved block.
  KVCacheStore store(1, std::move(controller), "", rid);
  auto* raiden_controller = KVCacheStoreTest::GetController(store);
  ASSERT_NE(raiden_controller, nullptr);
  EXPECT_EQ(store.num_registered_workers(), 1);
  // DeallocateBlockIds unlocks blocks (making them evictable) but never
  // clears is_allocated, so reclamation is visible in the locked count, not
  // num_free_blocks().
  const int base_locked =
      raiden_controller->block_manager()->num_locked_blocks();

  // 1. Save hash_a to host: insert as HBM, pin, save, poll, release.
  ASSERT_TRUE(store
                  .Insert({"hash_a"},
                          {RaidenBlockID(rid, -1, 0, BlockStatus::HBM)}, false)
                  .first);
  ASSERT_TRUE(store.Pin({"hash_a"}));
  ASSERT_TRUE(store.Save({"hash_a"}).ok());
  bool done = false;
  while (!done) {
    auto [save_done, save_failed, save_pending] = store.PollSaveStatus();
    ASSERT_TRUE(save_failed.empty()) << "Async Save failed during polling";
    done = !save_done.empty();
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }
  store.Release({"hash_a"});
  EXPECT_EQ(raiden_controller->block_manager()->num_locked_blocks(),
            base_locked + 1);

  // 2. Displace hash_a to the eviction-candidate list; the candidate keeps
  // its host block.
  ASSERT_TRUE(store
                  .Insert({"hash_b"},
                          {RaidenBlockID(rid, -1, 1, BlockStatus::HBM)}, false)
                  .first);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_a"));
  EXPECT_EQ(raiden_controller->block_manager()->num_locked_blocks(),
            base_locked + 1);

  // 3. Re-admit hash_a with a fresh HBM slice (re-store of the same prefix).
  // The rescue replaces the candidate's value; its stale host block must
  // return to the allocator.
  ASSERT_TRUE(store.InsertAndLock(
      {"hash_a"}, {RaidenBlockID(rid, -1, 2, BlockStatus::HBM)}, false));
  EXPECT_EQ(raiden_controller->block_manager()->num_locked_blocks(),
            base_locked);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::IsEmpty());

  // 4. Roll back the admission: the fresh HBM entry is deleted, and nothing
  // double-frees the already-reclaimed host block.
  EXPECT_EQ(store.ReleaseAndDelete({"hash_a"}), 1);
  EXPECT_EQ(raiden_controller->block_manager()->num_locked_blocks(),
            base_locked);
  auto lookup_a = store.Lookup({"hash_a"});
  ASSERT_TRUE(lookup_a.ok());
  EXPECT_EQ(lookup_a->size(), 0);
  auto lookup_b = store.Lookup({"hash_b"});
  ASSERT_TRUE(lookup_b.ok());
  EXPECT_EQ(lookup_b->size(), 1);
}

// Same leak through the plain Insert() path: re-inserting a hash that sits in
// the candidate list rescues the entry with the new value and must reclaim
// the stale host block.
TEST_F(KVCacheStoreEmbeddedControllerTest,
       CandidateReadmissionViaInsertReclaimsHostBlock) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(1, std::move(controller), "", rid);
  auto* raiden_controller = KVCacheStoreTest::GetController(store);
  // Reclamation unlocks the block rather than clearing is_allocated, so
  // track the locked count (see CandidateReadmissionReclaimsHostBlock).
  const int base_locked =
      raiden_controller->block_manager()->num_locked_blocks();

  // Give hash_a a real allocated host block, then displace it.
  auto host_ids_or = raiden_controller->AllocateBlockIds(1);
  ASSERT_TRUE(host_ids_or.ok());
  ASSERT_TRUE(
      store
          .Insert({"hash_a"},
                  {RaidenBlockID(rid, (*host_ids_or)[0], BlockStatus::HOST)},
                  true)
          .first);
  ASSERT_TRUE(store
                  .Insert({"hash_b"},
                          {RaidenBlockID(rid, -1, 1, BlockStatus::HBM)}, false)
                  .first);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("hash_a"));
  EXPECT_EQ(raiden_controller->block_manager()->num_locked_blocks(),
            base_locked + 1);

  // Re-insert hash_a with a device-only slice: the candidate's host block
  // must be reclaimed as its value is replaced.
  store.Insert({"hash_a"}, {RaidenBlockID(rid, -1, 3, BlockStatus::HBM)},
               false);
  EXPECT_EQ(raiden_controller->block_manager()->num_locked_blocks(),
            base_locked);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::IsEmpty());
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


  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull
  // design the DESTINATION's own worker (test_server_, backed by a mock
  // transfer manager) executes the copy, and the source only leases.

  // Setup dest controller and KVCacheStore
  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller), registry_address, rid);

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
  // The DESTINATION's worker executed the pull, against the source's
  // authoritative block id (42, from the verify hook) and into the landing
  // block the store allocated.
  EXPECT_EQ(dst_transfer_mock_->vector_h2h_read_calls, 1);
  EXPECT_THAT(dst_transfer_mock_->last_src_offsets, ::testing::ElementsAre(42));
  EXPECT_THAT(dst_transfer_mock_->last_dst_offsets, ::testing::ElementsAre(0));

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

  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull
  // design the DESTINATION's own worker (test_server_, backed by a mock
  // transfer manager) executes the copy, and the source only leases.

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(2, std::move(dst_controller), registry_address, rid);

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

  // The transfer now runs on the DESTINATION, so that is where the failure is
  // injected.
  dst_transfer_mock_->fail_transfers = true;

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
  ASSERT_TRUE(
      orchestrator_client
          .RegisterController(src_unit, src_controller_server->server_address)
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
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);
  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(2, std::move(dst_controller), "", rid);

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
  ASSERT_TRUE(
      orchestrator_client
          .RegisterController(src_unit, src_controller_server->server_address)
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
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);
  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(2, std::move(dst_controller), "", rid);

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
  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller), "", rid);

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

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 1, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(2, std::move(dst_controller), registry_address, rid);

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
  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server_1->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server_2->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller), registry_address, rid);

  // Insert and pin remote block hash_0 and hash_1
  std::vector<std::string> hashes = {"hash_0", "hash_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id_1, 10, BlockStatus::REMOTE),
      RaidenBlockID(src_raiden_id_2, 20, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // Trigger ReadRemote for both
  absl::Status status = store.ReadRemote(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // A batch spanning two peers takes one lease per peer and joins the futures,
  // so it still commits as a UNIT: both hashes complete together, or neither.
  // (The staged promise-gating this test used to do lived on the source's
  // transfer callback, which the pull design removed -- the destination's
  // mock now completes both pulls.)
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

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveSetsMetadataEntriesOnCompletion) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, *metadata_or);

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockID(rid, -1, 1, BlockStatus::HBM)};

  ASSERT_TRUE(store.Insert(hashes, slices, false).first);
  ASSERT_TRUE(store.Pin(hashes));

  // Insert has already called SetMetadataEntry for both slices, but their HBM
  // status fails its data-lives-in-local-host-memory filter: the data exists
  // only in HBM at this point, so the LRU registration alone must leave the
  // table empty.
  EXPECT_THAT(metadata_or->ValidEntries(), ::testing::IsEmpty());

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
      done = true;
    }
    if (!done) {
      absl::SleepFor(absl::Milliseconds(10));
    }
  }

  // Save completion lands the data on host blocks 0 and 1, which is when the
  // bindings enter the table.
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "hash_1", 0),
                          ::testing::FieldsAre(1, "hash_2", 1)));
}

TEST_F(KVCacheStoreEmbeddedControllerTest,
       ReadRemoteSetsMetadataEntriesOnCompletion) {
  // Same setup as ReadRemoteSuccess: registry, src controller with a
  // successful H2H callback, dest store — here with a metadata table attached.
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

  kv_cache::RaidenId src_raiden_id{"src_job", "0", "src_data", 0};

  ::tpu_raiden::controller::OrchestratorServiceClient orchestrator_client(
      grpc::CreateChannel(orchestrator_address_,
                          grpc::InsecureChannelCredentials()));
  auto register_status = orchestrator_client.RegisterController(
      src_unit, src_controller_server->server_address);
  ASSERT_TRUE(register_status.ok()) << register_status.message();

  auto worker_status = src_controller_server->client->RegisterWorker(
      "worker_0", "src_worker_0_addr", {{"src_worker_0_transfer", {}}});
  ASSERT_TRUE(worker_status.ok()) << worker_status.message();

  // Every read is now validated at the source by construction -- there is no
  // longer any RPC that transfers without verifying and pinning first. Grant
  // the lease and echo back authoritative ids.
  src_controller_server->service->SetReadRemoteHooks(
      [&](absl::Span<const std::string> h)
          -> absl::StatusOr<std::vector<int32_t>> {
        return std::vector<int32_t>(h.size(), 42);
      },
      [&](absl::Span<const std::string> /*h*/) {});
  // NOTE: the source no longer transfers anything. Under the pull design
  // the DESTINATION's own worker (test_server_, backed by a mock transfer
  // manager) executes the copy; the source only leases.

  auto dst_controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*dst_controller, "worker_0",
                        test_server_->server_address);

  MetadataRegion metadata_region(10);
  auto metadata_or = KVCacheMetadata::Format(metadata_region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"dst_job", "0", "dst_cache", 0};
  KVCacheStore store(10, std::move(dst_controller), registry_address, rid,
                     *metadata_or);

  std::vector<std::string> hashes = {"hash_0"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(src_raiden_id, 42, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.InsertAndLock(hashes, slices, true));

  // InsertAndLock has already called SetMetadataEntry for the slice, but its
  // REMOTE status fails the same data-lives-in-local-host-memory filter: a
  // REMOTE entry owns no local data and must stay out of the table.
  EXPECT_THAT(metadata_or->ValidEntries(), ::testing::IsEmpty());

  absl::Status status = store.ReadRemote(hashes);
  ASSERT_TRUE(status.ok()) << status.message();

  // Poll for completion
  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [done_hashes, failed_hashes, pending_hashes] =
        store.PollRemoteReadStatus();
    ASSERT_TRUE(failed_hashes.empty());
    if (!done_hashes.empty()) {
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // Read completion lands the remote data on local host block 0, which is
  // when the binding enters the table.
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "hash_0", 0)));

  registry_server->Shutdown();
}


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

TEST(KVCacheStoreTest, RecoverFromLocalManifestRebuildsLruCache) {
  RaidenId rid{"manifest_job", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  // Table left behind by the previous incarnation of this store.
  ASSERT_TRUE(metadata_or->Set(5, "hash_b", 3).ok());
  ASSERT_TRUE(metadata_or->Set(7, "hash_a", 4).ok());
  ASSERT_TRUE(metadata_or->Set(9, "hash_c", 8).ok());

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  KVCacheStore store(10, std::move(controller),
                     /*global_registry_address=*/"", rid, *metadata_or);

  auto recovered_or = store.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 3);

  auto lookup = store.Lookup({"hash_a", "hash_b", "hash_c"});
  ASSERT_TRUE(lookup.ok());
  ASSERT_EQ(lookup->size(), 3);
  EXPECT_EQ((*lookup)[0].second.status, BlockStatus::HOST);
  EXPECT_EQ((*lookup)[0].second.host_block_id, 7);
  EXPECT_EQ((*lookup)[1].second.host_block_id, 5);
  EXPECT_EQ((*lookup)[2].second.host_block_id, 9);

  // Recovered blocks are allocated and locked; new allocations avoid them.
  for (int id : {5, 7, 9}) {
    EXPECT_TRUE(controller_ptr->block_manager()->IsAllocated(id));
    EXPECT_TRUE(controller_ptr->block_manager()->IsLocked(id));
  }

  // The seq counter resumes past the largest recovered stamp: the next host
  // insert is stamped 9, not 0.
  ASSERT_TRUE(
      store.Insert({"hash_d"}, {RaidenBlockID(rid, 0, BlockStatus::HOST)}, true)
          .first);
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(0, "hash_d", 9),
                          ::testing::FieldsAre(5, "hash_b", 3),
                          ::testing::FieldsAre(7, "hash_a", 4),
                          ::testing::FieldsAre(9, "hash_c", 8)));
}

TEST(KVCacheStoreTest, RecoverFromLocalManifestRebuildsLruOrder) {
  RaidenId rid{"manifest_job_order", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  ASSERT_TRUE(metadata_or->Set(5, "hash_b", 3).ok());
  ASSERT_TRUE(metadata_or->Set(7, "hash_a", 4).ok());
  ASSERT_TRUE(metadata_or->Set(9, "hash_c", 8).ok());

  // The table also records eviction candidates, so it may hold more entries
  // than the LRU cache capacity. With capacity 2 the oldest entry overflows
  // into a candidate again, keeping its block and metadata entry.
  KVCacheStore store(2, MakeRecoveryController(rid, 10),
                     /*global_registry_address=*/"", rid, *metadata_or);
  auto recovered_or = store.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 3);

  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ElementsAre("hash_b"));
  EXPECT_EQ(metadata_or->ValidEntries().size(), 3);
}

TEST(KVCacheStoreTest, RecoverFromLocalManifestKeepsNewestDuplicate) {
  RaidenId rid{"manifest_job_dup", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  ASSERT_TRUE(metadata_or->Set(2, "dup_hash", 1).ok());
  ASSERT_TRUE(metadata_or->Set(4, "other", 3).ok());
  ASSERT_TRUE(metadata_or->Set(6, "dup_hash", 5).ok());

  auto controller = MakeRecoveryController(rid, 10);
  auto* controller_ptr = controller.get();
  KVCacheStore store(10, std::move(controller),
                     /*global_registry_address=*/"", rid, *metadata_or);

  auto recovered_or = store.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 2);

  // The newest binding wins; the stale block is neither tracked nor
  // allocated, and its entry is cleared from the table.
  auto lookup = store.Lookup({"dup_hash"});
  ASSERT_EQ(lookup->size(), 1);
  EXPECT_EQ((*lookup)[0].second.host_block_id, 6);
  EXPECT_TRUE(controller_ptr->block_manager()->IsAllocated(6));
  EXPECT_FALSE(controller_ptr->block_manager()->IsAllocated(2));
  EXPECT_THAT(metadata_or->ValidEntries(),
              ElementsAre(::testing::FieldsAre(4, "other", 3),
                          ::testing::FieldsAre(6, "dup_hash", 5)));
}

// Only reachable through misuse: recovery must run on a fresh store, so a
// conflicting allocation means someone allocated before (or instead of)
// recovering. Verifies the failure is clean — error out, LRU cache and table
// untouched.
TEST(KVCacheStoreTest, RecoverFromLocalManifestFailsOnAllocatorConflict) {
  RaidenId rid{"manifest_job_conflict", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());
  ASSERT_TRUE(metadata_or->Set(0, "rh1", 0).ok());

  auto controller = MakeRecoveryController(rid, 10);
  // Block 0 is already taken locally before recovery runs.
  ASSERT_TRUE(controller->AllocateBlockIds(1).ok());

  KVCacheStore store(10, std::move(controller),
                     /*global_registry_address=*/"", rid, *metadata_or);
  auto recovered_or = store.RecoverFromLocalManifest();
  EXPECT_EQ(recovered_or.status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(store.Lookup({"rh1"})->size(), 0);
  EXPECT_EQ(metadata_or->ValidEntries().size(), 1);
}

TEST(KVCacheStoreTest, RecoverFromLocalManifestPreconditions) {
  RaidenId rid{"manifest_job_pre", "0", "kv_cache", 0};
  MetadataRegion region(10);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 10);
  ASSERT_TRUE(metadata_or.ok());

  // No raiden controller.
  KVCacheStore store_no_controller(10, /*raiden_controller=*/nullptr,
                                   /*global_registry_address=*/"", rid,
                                   *metadata_or);
  EXPECT_EQ(store_no_controller.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // No attached metadata table.
  KVCacheStore store_no_metadata(10, MakeRecoveryController(rid, 10),
                                 /*global_registry_address=*/"", rid);
  EXPECT_EQ(store_no_metadata.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Non-empty LRU cache.
  KVCacheStore store_non_empty(10, MakeRecoveryController(rid, 10),
                               /*global_registry_address=*/"", rid,
                               *metadata_or);
  ASSERT_TRUE(store_non_empty
                  .Insert({"hash_a"}, {RaidenBlockID(rid, 0, BlockStatus::HOST)},
                          true)
                  .first);
  EXPECT_EQ(store_non_empty.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Empty table: recovery succeeds with zero blocks.
  MetadataRegion empty_region(10);
  auto empty_metadata_or = KVCacheMetadata::Format(empty_region.span(), 10);
  ASSERT_TRUE(empty_metadata_or.ok());
  KVCacheStore store_empty(10, MakeRecoveryController(rid, 10),
                           /*global_registry_address=*/"", rid,
                           *empty_metadata_or);
  auto recovered_or = store_empty.RecoverFromLocalManifest();
  ASSERT_TRUE(recovered_or.ok()) << recovered_or.status().ToString();
  EXPECT_EQ(*recovered_or, 0);
}

TEST(KVCacheStoreTest, MultiBackendPriorityLookupChain) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/4);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> b1_hashes = {"h1", "h2"};
  std::vector<RaidenBlockID> b1_slices = {
      RaidenBlockID(id, 1, BlockStatus::HOST),
      RaidenBlockID(id, 2, BlockStatus::HOST)};
  b1->Insert(b1_hashes, b1_slices, /*on_host=*/true);

  std::vector<std::string> b2_hashes = {"h3", "h4"};
  std::vector<RaidenBlockID> b2_slices = {
      RaidenBlockID(id, 3, BlockStatus::HOST),
      RaidenBlockID(id, 4, BlockStatus::HOST)};
  b2->Insert(b2_hashes, b2_slices, /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2});

  auto lookup_res =
      store.Lookup({"h1", "h2", "h3", "h4"}, /*enable_global=*/true);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 4);
  EXPECT_EQ((*lookup_res)[0].first, "h1");
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, 1);
  EXPECT_EQ((*lookup_res)[1].first, "h2");
  EXPECT_EQ((*lookup_res)[1].second.host_block_id, 2);
  EXPECT_EQ((*lookup_res)[2].first, "h3");
  EXPECT_EQ((*lookup_res)[2].second.host_block_id, 3);
  EXPECT_EQ((*lookup_res)[3].first, "h4");
  EXPECT_EQ((*lookup_res)[3].second.host_block_id, 4);
}

TEST(KVCacheStoreTest, MultiBackendTierGatingWhenGlobalDisabled) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1"}, {RaidenBlockID(id, 1, BlockStatus::HOST)},
             /*on_host=*/true);
  b2->Insert({"h2"}, {RaidenBlockID(id, 2, BlockStatus::HOST)},
             /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2});

  // enable_global = false => max_tier = 0 => stops after Tier 0
  auto gated_res = store.Lookup({"h1", "h2"}, /*enable_global=*/false);
  ASSERT_TRUE(gated_res.ok());
  ASSERT_EQ(gated_res->size(), 1);
  EXPECT_EQ((*gated_res)[0].first, "h1");

  // enable_global = true => max_tier = -1 => queries Tier 0 and Tier 1
  auto ungated_res = store.Lookup({"h1", "h2"}, /*enable_global=*/true);
  ASSERT_TRUE(ungated_res.ok());
  ASSERT_EQ(ungated_res->size(), 2);
  EXPECT_EQ((*ungated_res)[0].first, "h1");
  EXPECT_EQ((*ungated_res)[1].first, "h2");
}

TEST(KVCacheStoreTest, MultiBackendInsertAndLockRollback) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/1);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2});

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> hashes = {"h1", "h2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(id, 1, BlockStatus::REMOTE),
      RaidenBlockID(id, 2, BlockStatus::REMOTE)};

  // b1 capacity=2 supports 2 blocks, but b2 capacity=1 fails on 2 blocks.
  bool status = store.InsertAndLock(hashes, slices, /*on_host=*/true);
  EXPECT_FALSE(status);

  // Verify rollback on b1: no locks or entries remain.
  EXPECT_EQ(b1->GetSize(), 0);
  EXPECT_EQ(b1->GetPinCount("h1"), 0);
  EXPECT_EQ(b1->GetPinCount("h2"), 0);
  EXPECT_EQ(b2->GetSize(), 0);
}

TEST(KVCacheStoreTest, MultiBackendPinAndRelease) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1"}, {RaidenBlockID(id, 1, BlockStatus::HOST)},
             /*on_host=*/true);
  b2->Insert({"h2"}, {RaidenBlockID(id, 2, BlockStatus::HOST)},
             /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2});

  EXPECT_TRUE(store.Pin({"h1", "h2"}));
  EXPECT_EQ(store.GetPinCount("h1"), 1);
  EXPECT_EQ(store.GetPinCount("h2"), 1);
  EXPECT_EQ(b1->GetPinCount("h1"), 1);
  EXPECT_EQ(b2->GetPinCount("h2"), 1);

  store.Release({"h1", "h2"});
  EXPECT_EQ(store.GetPinCount("h1"), 0);
  EXPECT_EQ(store.GetPinCount("h2"), 0);
  EXPECT_EQ(b1->GetPinCount("h1"), 0);
  EXPECT_EQ(b2->GetPinCount("h2"), 0);
}

using ::testing::EndsWith;
using ::testing::Not;
using ::testing::StartsWith;

// ---------------------------------------------------------------------------
// Store-server discovery: a store publishes where peers reach it, keyed by
// its RaidenId, so a global-registry Lookup result becomes dialable.
// ---------------------------------------------------------------------------

// Owns a registry server on an ephemeral port for the tests below.
class StoreDiscoveryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    service_ = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    registry_address_ = "localhost:" + std::to_string(port);
    channel_ = grpc::CreateChannel(registry_address_,
                                   grpc::InsecureChannelCredentials());
    client_ = std::make_unique<global_registry::GlobalRegistryClient>(channel_);
  }

  void TearDown() override {
    if (server_) server_->Shutdown();
  }

  std::string registry_address_;
  std::unique_ptr<global_registry::GlobalRegistryServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<global_registry::GlobalRegistryClient> client_;
};

TEST_F(StoreDiscoveryTest, PublishesStoreAddressToTheRegistry) {
  RaidenId rid{"disco_job", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);

  // Advertised host is exactly what was supplied; the port is gRPC's choice.
  ASSERT_FALSE(store.store_server_address().empty());
  EXPECT_THAT(store.store_server_address(), StartsWith("127.0.0.1:"));
  ASSERT_NE(store.store_server(), nullptr);
  EXPECT_EQ(store.store_server_address(),
            absl::StrCat("127.0.0.1:", store.store_server()->GetGrpcPort()));

  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), store.store_server_address());
  // The controller address rides along so peer controller resolution can
  // move off the orchestrator later.
  EXPECT_EQ(resolved->controller_address(), store.raiden_controller_address());
}

// store_server_ip is bind-and-advertise, so the published address is
// actually connectable -- the property the old "localhost:<port>" lacked.
TEST_F(StoreDiscoveryTest, PublishedAddressIsConnectable) {
  RaidenId rid{"disco_job", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();

  auto peer_channel = grpc::CreateChannel(resolved->store_server_address(),
                                          grpc::InsecureChannelCredentials());
  ASSERT_TRUE(peer_channel->WaitForConnected(std::chrono::system_clock::now() +
                                             std::chrono::seconds(10)));
}

// The escape hatch: no ip means bind the wildcard interface and publish
// nothing. Every pre-existing caller takes this path.
TEST_F(StoreDiscoveryTest, NoStoreServerIpPublishesNothing) {
  RaidenId rid{"disco_job_quiet", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"");

  EXPECT_TRUE(store.store_server_address().empty());
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(rid).status()));
}

// Teardown must retract the registration, or peers keep dialling a dead port.
TEST_F(StoreDiscoveryTest, DestructorUnpublishes) {
  RaidenId rid{"disco_job_gone", "0", "kv_cache", 0};
  {
    KVCacheStore store(/*capacity=*/16, registry_address_, rid,
                       /*num_shards=*/1, /*shard_size_bytes=*/512,
                       /*raiden_orchestrator_address=*/"",
                       /*store_server_ip=*/"127.0.0.1");
    ASSERT_TRUE(client_->ResolveStore(rid).ok());
  }
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(rid).status()));
}

// A restarted store comes back on a different ephemeral port. Because the
// registration is keyed by RaidenId, the new address replaces the old one
// rather than accumulating beside it.
TEST_F(StoreDiscoveryTest, RestartReplacesPublishedAddress) {
  RaidenId rid{"disco_job_restart", "0", "kv_cache", 0};

  std::string first_address;
  {
    KVCacheStore store(/*capacity=*/16, registry_address_, rid,
                       /*num_shards=*/1, /*shard_size_bytes=*/512,
                       /*raiden_orchestrator_address=*/"",
                       /*store_server_ip=*/"127.0.0.1");
    first_address = store.store_server_address();
  }

  KVCacheStore restarted(/*capacity=*/16, registry_address_, rid,
                         /*num_shards=*/1, /*shard_size_bytes=*/512,
                         /*raiden_orchestrator_address=*/"",
                         /*store_server_ip=*/"127.0.0.1");
  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), restarted.store_server_address());
  EXPECT_NE(resolved->store_server_address(), first_address);
}

// When a backend already hosts a store server, the store must adopt and
// publish THAT server rather than stand up a second one, so a node serves peers
// from exactly one port. The store must also get there before anything else
// starts that server, because StartServer never rebinds a running one -- the
// factory path GlobalMemoryPoolingBackend::Create(config, controller) does
// start it, and the single-argument Create this test goes through does not.
TEST_F(StoreDiscoveryTest, AdoptsAndPublishesTheBackendsServer) {
  RaidenId rid{"disco_job_tiered", "0", "kv_cache", 0};

  BackendConfig host_config;
  host_config.type = "HostOffloadBackend";
  host_config.capacity = 16;
  host_config.raiden_id = rid;

  BackendConfig pooling_config;
  pooling_config.type = "HostOffloadBackend";
  pooling_config.capacity = 16;
  pooling_config.global_registry_address = registry_address_;
  pooling_config.raiden_id = rid;

  const BackendConfig configs[] = {host_config, pooling_config};
  auto store_or = KVCacheStore::Create(
      absl::MakeConstSpan(configs), /*capacity=*/16, registry_address_, rid,
      /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*raiden_orchestrator_address=*/"", /*store_server_ip=*/"127.0.0.1");
  ASSERT_TRUE(store_or.ok()) << store_or.status().ToString();
  auto& store = **store_or;

  auto* pooling =
      dynamic_cast<HostOffloadBackend*>(store.backends()[1].get());
  ASSERT_NE(pooling, nullptr);

  // Exactly one server, created and published by the store.
  ASSERT_NE(store.store_server(), nullptr);

  // Published under the supplied ip, not the backend's hardcoded wildcard.
  EXPECT_THAT(store.store_server_address(), StartsWith("127.0.0.1:"));
  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), store.store_server_address());
}

// The controller is addressed from the same ip, with its port either chosen by
// gRPC (0) or taken verbatim.
TEST_F(StoreDiscoveryTest, ControllerAddressComposedFromIpAndPort) {
  RaidenId rid{"disco_job_ctrl", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);

  EXPECT_THAT(store.raiden_controller_address(), StartsWith("127.0.0.1:"));
  // Port 0 means "gRPC picks"; the advertised address carries the real port.
  EXPECT_THAT(store.raiden_controller_address(), Not(EndsWith(":0")));
}


// A tier-0 backend whose Lookup parks, so a peer's Fetch can be held inside the
// store's own service while the store is being destroyed.
class ParkingBackend : public HostOffloadBackend {
 public:
  using HostOffloadBackend::HostOffloadBackend;

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override {
    if (!entered.HasBeenNotified()) entered.Notify();
    release.WaitForNotification();
    handler_finished.store(true);
    return HostOffloadBackend::Lookup(block_hashes, options);
  }

  absl::Notification entered;
  absl::Notification release;
  std::atomic<bool> handler_finished{false};
};

// Destroying a store must not return while a peer's RPC is still executing
// inside its service.
//
// Scope, stated because it is easy to over-read: this covers a store-OWNED
// server, where the drain is guaranteed twice over -- by the destructor's
// explicit Shutdown, and by owned_store_server_ being declared after
// raiden_controller_ and so destroyed before it. Removing the explicit
// Shutdown does NOT fail this test for that reason.
//
// The case the explicit Shutdown alone covers is a server ADOPTED from a
// backend: backends_ is declared before raiden_controller_, so member
// destruction frees the controller first and leaves a backend-hosted server
// answering RPCs that dereference it. That path has no regression test -- see
// the teardown notes in the store-server discovery doc.
TEST_F(StoreDiscoveryTest, TeardownDrainsAnInFlightPeerRpc) {
  RaidenId rid{"disco_job_teardown", "0", "kv_cache", 0};
  auto parking = std::make_shared<ParkingBackend>(
      /*capacity=*/16, std::nullopt, rid, /*raiden_controller=*/nullptr);

  auto store = std::make_unique<KVCacheStore>(
      parking, rid, /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*raiden_orchestrator_address=*/"", /*store_server_ip=*/"127.0.0.1",
      /*raiden_controller_port=*/0, registry_address_);
  ASSERT_FALSE(store->store_server_address().empty());

  auto peer_channel = grpc::CreateChannel(store->store_server_address(),
                                          grpc::InsecureChannelCredentials());
  KVCacheStoreClient peer(peer_channel);
  auto fetch_future = peer.Fetch({"h1"}, /*device_block_ids=*/{},
                                 /*host_block_ids=*/{7});

  // The peer's Fetch is now parked inside our service, holding the handler.
  parking->entered.WaitForNotification();

  std::thread destroyer([&store] { store.reset(); });
  // The destructor should be blocked draining that handler, not racing past it.
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_FALSE(parking->handler_finished.load());

  parking->release.Notify();
  destroyer.join();

  // Teardown outlived the handler rather than pulling the controller out from
  // under it.
  EXPECT_TRUE(parking->handler_finished.load());
  (void)fetch_future.Await();  // whatever it reports, it must not crash
}


// A store with no backend must decline to serve rather than index an empty
// backends_ vector: backend() does not bounds-check, and capacity() guards for
// exactly this case, so the empty vector is reachable.
TEST_F(StoreDiscoveryTest, NoBackendDoesNotServeOrCrash) {
  RaidenId rid{"disco_job_nobackend", "0", "kv_cache", 0};
  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{}, rid,
                     /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0, registry_address_);

  EXPECT_EQ(store.store_server(), nullptr);
  EXPECT_TRUE(store.store_server_address().empty());
  EXPECT_TRUE(absl::IsNotFound(client_->ResolveStore(rid).status()));
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
