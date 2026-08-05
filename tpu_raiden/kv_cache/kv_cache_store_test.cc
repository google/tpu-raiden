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
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
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
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/controller/orchestrator_service_client.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/controller/raiden_orchestrator.h"
#include "tpu_raiden/core/controller/test_util.h"
#include "tpu_raiden/core/kv_manager_holder.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry.grpc.pb.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_server.h"
#include "tpu_raiden/kv_cache/global_registry/test_util.h"
#include "tpu_raiden/kv_cache/host_offload_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/kv_cache_store_service.grpc.pb.h"

#ifndef _WIN32
int ignore_sigpipe = []() {
  std::signal(SIGPIPE, SIG_IGN);
  return 0;
}();

// Every KVCacheStore owns a real
// RaidenController, so this binary mints many controllers in one process.
// A private ControllerServer per controller keeps them from sharing (and
// re-pointing) the process-wide singleton.
int use_private_controller_servers = []() {
  setenv("RAIDEN_DISABLE_SINGLETON_WORKER", "1", /*overwrite=*/1);
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
  KVCacheStore controller(50, "", {}, /*num_shards=*/1,
                          /*shard_size_bytes=*/512, "",
                          /*store_server_ip=*/"127.0.0.1");
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
      controller.Insert(hashes, slices, true).first);  // Successful again
}

TEST(KVCacheStoreTest, PinAndRelease) {
  KVCacheStore controller(2, "", {}, /*num_shards=*/1,
                          /*shard_size_bytes=*/512, "",
                          /*store_server_ip=*/"127.0.0.1");

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

  // Lookup {"101", "102"} will miss on 101 (since lookup uses
  // candidate-invisible Peek).
  auto lookup_res = controller.Lookup({"101", "102"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_TRUE(lookup_res->empty());
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("101"));

  // 102 is still in cache.
  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
}

TEST(KVCacheStoreTest, PartialPinRollback) {
  KVCacheStore controller(2, "", {}, /*num_shards=*/1,
                          /*shard_size_bytes=*/512, "",
                          /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore controller(2, "", {}, /*num_shards=*/1,
                          /*shard_size_bytes=*/512, "",
                          /*store_server_ip=*/"127.0.0.1");

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

  // 4. Verify that lookup for 101 misses (candidate invisible with Peek).
  auto lookup_res = controller.Lookup({"101"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_TRUE(lookup_res->empty());
  // 101 should still be in candidates.
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(controller),
              ::testing::ElementsAre("101"));

  EXPECT_EQ(controller.Lookup({"102"})->size(), 1);
  EXPECT_EQ(controller.Lookup({"103"})->size(), 1);
}

TEST(KVCacheStoreTest, GlobalLookupFallback) {
  // 1. Start a local registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // 2. Register some blocks in the registry
  auto& registry_client = *reg_server->client;

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
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(50, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

  // Insert blocks locally
  std::vector<std::string> local_hashes = {"local_only_hash", "shared_hash"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0},
      RaidenId{"local_job", "0", "kv_cache", 1}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Case 1: Full local hit, no global hit
  {
    auto lookup_res = store.Lookup({"local_only_hash"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 0);
  }

  // Case 2: Both local and global has the same hit, but we return local hit
  // results
  {
    auto lookup_res = store.Lookup({"shared_hash"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 1);
    EXPECT_EQ((*lookup_res)[0].first, "shared_hash");
    // Should return local info, not remote info from registry
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");
    EXPECT_EQ((*lookup_res)[0].second.raiden_id.data_replica_idx, 1);
  }

  // Case 3: No local hit, only global hits
  {
    auto lookup_res = store.Lookup({"global_hash_1", "global_hash_2"});
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
        store.Lookup({"local_only_hash", "global_hash_1", "global_hash_2"});
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
        {"local_only_hash", "global_hash_1", "missing_hash", "global_hash_2"});
    ASSERT_TRUE(lookup_res.ok());
    ASSERT_EQ(lookup_res->size(), 2);  // local_only_hash, global_hash_1
    EXPECT_EQ((*lookup_res)[0].first, "local_only_hash");
    EXPECT_EQ((*lookup_res)[1].first, "global_hash_1");
  }
}

// Delegates every RPC to a real GlobalRegistryServiceImpl except Lookup,
// which always fails. Models a registry that IS reachable -- RegisterStore at
// construction succeeds -- but whose Lookup path is down. The old version of
// this test pointed construction itself at an unreachable address; that no
// longer works because RegisterStore failure now fails construction:
// a registry that cannot be reached AT ALL can no
// longer be distinguished from a misconfigured store, so the only
// representable "down" is Lookup failing after the store is already up.
class LookupFailingRegistryService final
    : public global_registry::GlobalRegistryService::Service {
 public:
  grpc::Status Register(grpc::ServerContext* context,
                        const global_registry::RegisterRequest* request,
                        global_registry::RegisterResponse* response) override {
    return delegate_.Register(context, request, response);
  }
  grpc::Status Lookup(grpc::ServerContext* context,
                      const global_registry::LookupRequest* request,
                      global_registry::LookupResponse* response) override {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, "registry lookup down");
  }
  grpc::Status Unregister(
      grpc::ServerContext* context,
      const global_registry::UnregisterRequest* request,
      global_registry::UnregisterResponse* response) override {
    return delegate_.Unregister(context, request, response);
  }
  grpc::Status PullOwned(
      grpc::ServerContext* context,
      const global_registry::PullOwnedRequest* request,
      grpc::ServerWriter<global_registry::PullOwnedResponse>* writer) override {
    return delegate_.PullOwned(context, request, writer);
  }
  grpc::Status RegisterStore(
      grpc::ServerContext* context,
      const global_registry::RegisterStoreRequest* request,
      global_registry::RegisterStoreResponse* response) override {
    return delegate_.RegisterStore(context, request, response);
  }
  grpc::Status ResolveStore(
      grpc::ServerContext* context,
      const global_registry::ResolveStoreRequest* request,
      global_registry::ResolveStoreResponse* response) override {
    return delegate_.ResolveStore(context, request, response);
  }
  grpc::Status UnregisterStore(
      grpc::ServerContext* context,
      const global_registry::UnregisterStoreRequest* request,
      global_registry::UnregisterStoreResponse* response) override {
    return delegate_.UnregisterStore(context, request, response);
  }

 private:
  global_registry::GlobalRegistryServiceImpl delegate_;
};

TEST(KVCacheStoreTest, GlobalLookupRegistryDown) {
  LookupFailingRegistryService service;
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // Construction succeeds: RegisterStore is reachable and healthy.
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(50, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

  // Insert one block locally
  std::vector<std::string> local_hashes = {"local_hash"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Lookup with enable_global = true.
  // It should NOT fail even though Lookup RPCs to the registry fail. It
  // should return the local hit.
  auto lookup_res = store.Lookup({"local_hash", "missing_hash"});
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash");
  EXPECT_EQ((*lookup_res)[0].second.raiden_id.job_name, "local_job");

  server->Shutdown();
}

// --- ReadRemote All-or-Nothing validate & pin block hashes at the src
// controller: source-side ValidateAndPinHostBlocks ---

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksSuccessReDerivesIdsAndPins) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");
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
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");
  auto ids_or =
      store.ValidateAndPinHostBlocks(std::vector<std::string>{"missing"});
  EXPECT_TRUE(absl::IsNotFound(ids_or.status())) << ids_or.status();
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksWrongStatusFailedPrecondition) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");
  RaidenId rid{"src_job", "0", "src_cache", 0};
  std::vector<std::string> hashes = {"remote_h"};
  std::vector<RaidenBlockID> slices = {RaidenBlockID(
      rid, /*host_block_id=*/-1, /*device_block_id=*/-1, BlockStatus::REMOTE)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/false).first);

  auto ids_or = store.ValidateAndPinHostBlocks(hashes);
  EXPECT_TRUE(absl::IsFailedPrecondition(ids_or.status())) << ids_or.status();
  EXPECT_EQ(store.GetPinCount("remote_h"), 0);
}

TEST(KVCacheStoreTest, ValidateAndPinHostBlocksAtomicRollbackOnPartialMiss) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");
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
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");
  auto ids_or = store.ValidateAndPinHostBlocks(std::vector<std::string>{});
  ASSERT_TRUE(ids_or.ok());
  EXPECT_TRUE(ids_or->empty());
}

TEST(KVCacheStoreTest,
     ValidateAndPinHostBlocksIncrementsAndReleasesExistingPin) {
  KVCacheStore store(4, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");
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
  KVCacheStore store(2, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

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
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // 2. Register some blocks in the registry
  auto& registry_client = *reg_server->client;

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
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(2, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

  // Lookup 3 hashes, but capacity is 2. It should only return 2.
  std::vector<std::string> lookup_hashes = {"global_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "global_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
}

TEST(KVCacheStoreTest, LookupCapLimitMixed) {
  // 1. Start a local registry server
  auto reg_server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = reg_server->server_address;

  // 2. Register some blocks in the registry
  auto& registry_client = *reg_server->client;

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
  RaidenId store_id{"store_job", "0", "kv_cache", 0};
  KVCacheStore store(2, server_address, store_id, /*num_shards=*/1,
                     /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

  // Insert 1 block locally
  std::vector<std::string> local_hashes = {"local_hash_1"};
  std::vector<RaidenBlockID> local_slices = {
      RaidenId{"local_job", "0", "kv_cache", 0}};
  ASSERT_TRUE(store.Insert(local_hashes, local_slices, true).first);

  // Lookup 3 hashes, but capacity is 2. It should only return 2 (1 local, 1
  // global).
  std::vector<std::string> lookup_hashes = {"local_hash_1", "global_hash_2",
                                            "global_hash_3"};
  auto lookup_res = store.Lookup(lookup_hashes);
  ASSERT_TRUE(lookup_res.ok());
  EXPECT_EQ(lookup_res->size(), 2);
  EXPECT_EQ((*lookup_res)[0].first, "local_hash_1");
  EXPECT_EQ((*lookup_res)[1].first, "global_hash_2");
}

TEST(KVCacheStoreTest, LookupAvailableSpaceLimit) {
  KVCacheStore store(3, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

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
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::UnorderedElementsAre("local_1", "local_2"));

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
  KVCacheStore store(3, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

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

  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("local_1"));

  // Lookup for local_1 misses (candidate invisible with Peek)
  auto lookup_local_1 = store.Lookup({"local_1"});
  ASSERT_TRUE(lookup_local_1.ok());
  EXPECT_TRUE(lookup_local_1->empty());
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre("local_1"));
}

TEST(KVCacheStoreTest, EvictRaceCondition) {
  KVCacheStore store(3, "", {}, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

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

// Worker-less controller: sufficient for the metadata and recovery tests,
// which only touch the logical block manager.
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

TEST(KVCacheStoreTest, InsertSetsAndDeleteClearsMetadataEntries) {
  MetadataRegion region(4);
  auto metadata_or = KVCacheMetadata::Format(region.span(), 4);
  ASSERT_TRUE(metadata_or.ok());

  RaidenId rid{"local_job", "0", "kv_cache", 0};
  KVCacheStore store(4, MakeRecoveryController(rid, 4),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, MakeRecoveryController(rid, 4),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, MakeRecoveryController(rid, 2),
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

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
    dst_transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
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

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveReusesFreedBlocksAfterEvict) {
  // Evicted host blocks return to the free pool even when the directory has
  // nothing left to evict: a full evict empties the directory and deallocates
  // every block, and the save below must be served from those.
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  // A two-block pool, so the first save exhausts it.
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 2, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);
  auto* controller_ptr = controller.get();

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(2, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  auto save_and_wait =
      [&store](const std::vector<std::string>& hashes) -> absl::Status {
    absl::Status status = store.Save(hashes);
    if (!status.ok()) return status;
    while (true) {
      auto [done, failed, pending] = store.PollSaveStatus();
      if (!failed.empty()) return absl::InternalError("async save failed");
      if (!done.empty()) return absl::OkStatus();
      absl::SleepFor(absl::Milliseconds(10));
    }
  };

  std::vector<std::string> first = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> first_slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM),
      RaidenBlockID(rid, -1, 1, BlockStatus::HBM)};
  ASSERT_TRUE(store.Insert(first, first_slices, false).first);
  ASSERT_TRUE(store.Pin(first));
  ASSERT_TRUE(save_and_wait(first).ok());
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 2);

  // Evict everything: the directory goes empty and both host blocks are
  // deallocated back to the free pool.
  store.Release(first);
  ASSERT_EQ(KVCacheStoreTest::Evict(store, first), 2);
  EXPECT_EQ(controller_ptr->block_manager()->num_free_blocks(), 2);
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 0);

  // The directory is empty and hash_3 gets pinned, so nothing is evictable:
  // the host block for this save has to come from the freed pool.
  std::vector<std::string> second = {"hash_3"};
  std::vector<RaidenBlockID> second_slices = {
      RaidenBlockID(rid, -1, 0, BlockStatus::HBM)};
  ASSERT_TRUE(store.Insert(second, second_slices, false).first);
  ASSERT_TRUE(store.Pin(second));
  absl::Status status = save_and_wait(second);
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(controller_ptr->block_manager()->num_locked_blocks(), 1);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, SaveSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesSuccess) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, -1, BlockStatus::HOST),
      RaidenBlockID(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);
  ASSERT_TRUE(store.Pin(hashes));

  absl::Status status = store.Load(hashes, slices, {2, 3});
  ASSERT_TRUE(status.ok()) << status.message();

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

  EXPECT_EQ(mock_mgr.d2h_calls, 0);
  EXPECT_EQ(mock_mgr.h2d_calls, 1);
  EXPECT_THAT(mock_mgr.last_src_offsets, ElementsAre(0, 1));
  EXPECT_THAT(mock_mgr.last_dst_offsets, ElementsAre(2, 3));

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

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesSizeMismatch) {
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, -1, BlockStatus::HOST)};

  absl::Status status = store.Load(hashes, slices, {2, 3});
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(std::string(status.message()), ::testing::HasSubstr("mismatch"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesUnpinnedSucceeds) {
  ::tpu_raiden::controller::MockTransferManager mock_mgr;
  test_server_->service->SetTransferManager(
      ::tpu_raiden::KVManagerHolder(&mock_mgr));

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);

  absl::Status status = store.Load(hashes, slices, {2});
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesAlreadyLoadingFails) {
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(rid, 0, -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert(hashes, slices, true).first);
  ASSERT_TRUE(store.Pin(hashes));

  absl::Status status1 = store.Load(hashes, slices, {2});
  ASSERT_TRUE(status1.ok());

  absl::Status status2 = store.Load(hashes, slices, {3});
  EXPECT_TRUE(absl::IsFailedPrecondition(status2));
  EXPECT_THAT(std::string(status2.message()),
              ::testing::HasSubstr("Block is already loading"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesMixedStatusesFails) {
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId rid{"test_job", "0", "test_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"hash_1", "hash_2"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(remote_rid, 0, -1, BlockStatus::REMOTE),
      RaidenBlockID(rid, 1, -1, BlockStatus::HOST)};

  ASSERT_TRUE(store.Insert({"hash_1"}, {slices[0]}, /*on_host=*/false).first);
  ASSERT_TRUE(store.Insert({"hash_2"}, {slices[1]}, /*on_host=*/true).first);
  ASSERT_TRUE(store.Pin(hashes));

  absl::Status status = store.Load(hashes, slices, {2, 3});
  EXPECT_TRUE(absl::IsInvalidArgument(status));
  EXPECT_THAT(std::string(status.message()),
              ::testing::HasSubstr("Mixed block statuses"));
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadWithSlicesRemoteSuccess) {
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  RaidenId local_rid{"local_job", "0", "local_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};

  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = registry_address;
  remote_config.raiden_id = remote_rid;

  auto remote_backend_or =
      HostOffloadBackend::Create(remote_config, controller.get());
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);

  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(remote_rid, 42, BlockStatus::HOST),
  };
  remote_backend->Insert({"load_remote_hash_1"}, remote_slices,
                         /*on_host=*/true);

  auto remote_server = KVCacheStoreServer::Create();
  ASSERT_OK(remote_server->StartServer(remote_backend.get(), controller.get(),
                                       "127.0.0.1"));

  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);
  ASSERT_OK(registry_client->RegisterStore(
      remote_rid, remote_server->GetServerAddress(), orchestrator_address_));

  KVCacheStore store(10, std::move(controller), registry_address, local_rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"load_remote_hash_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(remote_rid, 42, BlockStatus::REMOTE)};

  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/false).first);
  ASSERT_TRUE(store.Pin(hashes));

  absl::Status status = store.Load(hashes, slices, {5});
  ASSERT_TRUE(status.ok()) << status.message();

  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_failed.empty());
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("load_remote_hash_1"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, -1);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 5);
  EXPECT_EQ((*lookup_res)[0].second.raiden_id, local_rid);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadRemoteSuccess) {
  // 1. Setup GlobalRegistry server
  auto registry_server = global_registry::CreateTestGlobalRegistryServer();
  std::string registry_address = registry_server->server_address;

  RaidenId local_rid{"local_job", "0", "local_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};

  // 2. Setup local RaidenController & KVCacheStore
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  // 3. Setup remote node's backend & server
  BackendConfig remote_config;
  remote_config.type = "HostOffloadBackend";
  remote_config.capacity = 100;
  remote_config.global_registry_address = registry_address;
  remote_config.raiden_id = remote_rid;

  auto remote_backend_or =
      HostOffloadBackend::Create(remote_config, controller.get());
  ASSERT_OK(remote_backend_or.status());
  auto remote_backend =
      std::dynamic_pointer_cast<HostOffloadBackend>(*remote_backend_or);
  ASSERT_NE(remote_backend, nullptr);

  std::vector<RaidenBlockID> remote_slices = {
      RaidenBlockID(remote_rid, 42, BlockStatus::HOST),
  };
  remote_backend->Insert({"load_remote_hash_1"}, remote_slices,
                         /*on_host=*/true);

  auto remote_server = KVCacheStoreServer::Create();
  ASSERT_OK(remote_server->StartServer(remote_backend.get(), controller.get(),
                                       "127.0.0.1"));

  auto channel =
      grpc::CreateChannel(registry_address, grpc::InsecureChannelCredentials());
  auto registry_client =
      std::make_shared<global_registry::GlobalRegistryClient>(channel);
  ASSERT_OK(registry_client->RegisterStore(
      remote_rid, remote_server->GetServerAddress(), orchestrator_address_));

  // 4. Create store and insert remote block entry
  KVCacheStore store(10, std::move(controller), registry_address, local_rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"load_remote_hash_1"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(remote_rid, 42, BlockStatus::REMOTE)};

  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/false).first);
  ASSERT_TRUE(store.Pin(hashes));

  // 5. Load remote block into local device block 5
  absl::Status status = store.Load(hashes, {5});
  ASSERT_TRUE(status.ok()) << status.message();

  // 6. Poll for completion
  bool done = false;
  for (int attempt = 0; attempt < 100; ++attempt) {
    auto [load_done, load_failed, load_pending] = store.PollLoadStatus();
    ASSERT_TRUE(load_failed.empty());
    if (!load_done.empty()) {
      EXPECT_THAT(load_done,
                  ::testing::UnorderedElementsAre("load_remote_hash_1"));
      done = true;
      break;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(done);

  // 7. Verify status in store is updated to HBM and device_block_id is 5
  auto lookup_res = store.Lookup(hashes);
  ASSERT_TRUE(lookup_res.ok());
  ASSERT_EQ(lookup_res->size(), 1);
  EXPECT_EQ((*lookup_res)[0].second.status, BlockStatus::HBM);
  EXPECT_EQ((*lookup_res)[0].second.host_block_id, -1);
  EXPECT_EQ((*lookup_res)[0].second.device_block_id, 5);
  EXPECT_EQ((*lookup_res)[0].second.raiden_id, local_rid);
}

TEST_F(KVCacheStoreEmbeddedControllerTest, LoadUnpinnedRemoteBlockFails) {
  auto controller =
      std::make_unique<::tpu_raiden::controller::RaidenController>(
          unit_, 10, 1, 512, orchestrator_address_, "");
  RegisterAndInitWorker(*controller, "worker_0", test_server_->server_address);

  RaidenId local_rid{"local_job", "0", "local_cache", 0};
  RaidenId remote_rid{"remote_job", "0", "remote_cache", 0};
  KVCacheStore store(10, std::move(controller), "", local_rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

  std::vector<std::string> hashes = {"unpinned_remote_hash"};
  std::vector<RaidenBlockID> slices = {
      RaidenBlockID(remote_rid, /*host_block_id=*/-1, /*device_block_id=*/-1,
                    BlockStatus::REMOTE)};
  ASSERT_TRUE(store.Insert(hashes, slices, /*on_host=*/false).first);

  absl::Status status = store.Load(hashes, {0});
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("is not pinned"));
}

TEST(KVCacheStoreTest, InsertAndLockDetailedClassifiesBatch) {
  RaidenId rid{"test_job", "0", "test_cache", 0};
  KVCacheStore store(2, "", rid, /*num_shards=*/1, /*shard_size_bytes=*/512, "",
                     /*store_server_ip=*/"127.0.0.1");

  // hash_a pre-exists (unpinned); hash_b is new. The batch must classify
  // them apart, with no displacement at capacity 2.
  ASSERT_TRUE(
      store.Insert({"hash_a"}, {RaidenBlockID(rid, 0, BlockStatus::HOST)}, true)
          .first);
  InsertAndLockResult result =
      store.InsertAndLockDetailed({"hash_a", "hash_b"},
                                  {RaidenBlockID(rid, 0, BlockStatus::HOST),
                                   RaidenBlockID(rid, -1, 1, BlockStatus::HBM)},
                                  false);
  EXPECT_TRUE(result.success);
  EXPECT_THAT(result.existing, ::testing::ElementsAre("hash_a"));
  EXPECT_THAT(result.inserted, ::testing::ElementsAre("hash_b"));
  EXPECT_THAT(result.displaced, ::testing::IsEmpty());
  EXPECT_EQ(store.GetPinCount("hash_a"), 1);
  EXPECT_EQ(store.GetPinCount("hash_b"), 1);

  // Capacity full and everything pinned: the admission fails, is fully
  // rolled back, and reports empty lists.
  InsertAndLockResult rejected = store.InsertAndLockDetailed(
      {"hash_c"}, {RaidenBlockID(rid, -1, 2, BlockStatus::HBM)}, false);
  EXPECT_FALSE(rejected.success);
  EXPECT_THAT(rejected.existing, ::testing::IsEmpty());
  EXPECT_THAT(rejected.inserted, ::testing::IsEmpty());
  EXPECT_THAT(rejected.displaced, ::testing::IsEmpty());
  EXPECT_EQ(store.GetPinCount("hash_a"), 1);
  auto lookup_c = store.Lookup({"hash_c"});
  ASSERT_TRUE(lookup_c.ok());
  EXPECT_EQ(lookup_c->size(), 0);

  // With the pins released, admitting hash_c displaces the LRU entry into
  // the candidate list and reports it.
  store.Release({"hash_a", "hash_b"});
  InsertAndLockResult displacing = store.InsertAndLockDetailed(
      {"hash_c"}, {RaidenBlockID(rid, -1, 2, BlockStatus::HBM)}, false);
  EXPECT_TRUE(displacing.success);
  EXPECT_THAT(displacing.inserted, ::testing::ElementsAre("hash_c"));
  ASSERT_EQ(displacing.displaced.size(), 1);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::ElementsAre(displacing.displaced[0].first));

  // Reverting the admission restores the displaced candidate.
  EXPECT_EQ(store.ReleaseAndDelete({"hash_c"}), 1);
  EXPECT_THAT(KVCacheStoreTest::GetEvictCandidateKeys(store),
              ::testing::IsEmpty());
  auto lookup_restored = store.Lookup({displacing.displaced[0].first});
  ASSERT_TRUE(lookup_restored.ok());
  EXPECT_EQ(lookup_restored->size(), 1);
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
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(10, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

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
  KVCacheStore store(10, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
}

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictByHashesHostAndHbmToErased) {
  // 1. Start a local registry server
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

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
  KVCacheStore store(10, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
}

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictByHashesHostToErased) {
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

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
  KVCacheStore store(10, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
}

TEST_F(KVCacheStoreEmbeddedControllerTest, EvictOnSave) {
  auto server = global_registry::CreateTestGlobalRegistryServer();
  std::string server_address = server->server_address;

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
  KVCacheStore store(3, std::move(controller), server_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, std::move(controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
    auto lookup_res = store.Lookup({"hash_B", "hash_A"});
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
  KVCacheStore store(10, std::move(dst_controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, std::move(dst_controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, std::move(dst_controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, std::move(dst_controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(10, std::move(dst_controller), "", rid, std::nullopt,
                     /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(2, std::move(dst_controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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
  KVCacheStore store(10, std::move(dst_controller), registry_address, rid,
                     std::nullopt, /*store_server_ip=*/"127.0.0.1");

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

TEST_F(KVCacheStoreEmbeddedControllerTest,
       SaveSetsMetadataEntriesOnCompletion) {
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
  KVCacheStore store(10, std::move(controller), "", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

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
                     *metadata_or, /*store_server_ip=*/"127.0.0.1");

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
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

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
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");
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
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");

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
                     /*global_registry_address=*/"", rid, *metadata_or,
                     /*store_server_ip=*/"127.0.0.1");
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

  // A controller-less store is unrepresentable under the construction rules,
  // so the old no-controller sub-case is gone.

  // No attached metadata table.
  KVCacheStore store_no_metadata(10, MakeRecoveryController(rid, 10),
                                 /*global_registry_address=*/"", rid,
                                 std::nullopt,
                                 /*store_server_ip=*/"127.0.0.1");
  EXPECT_EQ(store_no_metadata.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Non-empty LRU cache.
  KVCacheStore store_non_empty(10, MakeRecoveryController(rid, 10),
                               /*global_registry_address=*/"", rid,
                               *metadata_or,
                               /*store_server_ip=*/"127.0.0.1");
  ASSERT_TRUE(
      store_non_empty
          .Insert({"hash_a"}, {RaidenBlockID(rid, 0, BlockStatus::HOST)}, true)
          .first);
  EXPECT_EQ(store_non_empty.RecoverFromLocalManifest().status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Empty table: recovery succeeds with zero blocks.
  MetadataRegion empty_region(10);
  auto empty_metadata_or = KVCacheMetadata::Format(empty_region.span(), 10);
  ASSERT_TRUE(empty_metadata_or.ok());
  KVCacheStore store_empty(10, MakeRecoveryController(rid, 10),
                           /*global_registry_address=*/"", rid,
                           *empty_metadata_or,
                           /*store_server_ip=*/"127.0.0.1");
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

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

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

TEST(KVCacheStoreTest, MultiBackendLocalLookupWhenGlobalDisabled) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1"}, {RaidenBlockID(id, 1, BlockStatus::HOST)},
             /*on_host=*/true);
  b2->Insert({"h2"}, {RaidenBlockID(id, 2, BlockStatus::HOST)},
             /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

  // enable_global = false => searches all local backends (b1 and b2)
  auto gated_res = store.Lookup({"h1", "h2"}, /*enable_global=*/false);
  ASSERT_TRUE(gated_res.ok());
  ASSERT_EQ(gated_res->size(), 2);
  EXPECT_EQ((*gated_res)[0].first, "h1");
  EXPECT_EQ((*gated_res)[1].first, "h2");

  // enable_global = true => queries all backends as well
  auto ungated_res = store.Lookup({"h1", "h2"}, /*enable_global=*/true);
  ASSERT_TRUE(ungated_res.ok());
  ASSERT_EQ(ungated_res->size(), 2);
  EXPECT_EQ((*ungated_res)[0].first, "h1");
  EXPECT_EQ((*ungated_res)[1].first, "h2");
}

TEST(KVCacheStoreTest, MultiBackendInsertAndLockRollback) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/1);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

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

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

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

  auto* pooling = dynamic_cast<HostOffloadBackend*>(store.backends()[1].get());
  ASSERT_NE(pooling, nullptr);

  // Exactly one server, created and published by the store.
  ASSERT_NE(store.store_server(), nullptr);

  // Published under the supplied ip, not the backend's hardcoded wildcard.
  EXPECT_THAT(store.store_server_address(), StartsWith("127.0.0.1:"));
  auto resolved = client_->ResolveStore(rid);
  ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
  EXPECT_EQ(resolved->store_server_address(), store.store_server_address());
}

// A backend that already hosts a KVCacheStoreServer (because something started
// one on it explicitly before store construction) is adopted by the store
// rather than standing up a second server.
TEST_F(StoreDiscoveryTest, AdoptsTheBackendsServerRatherThanOwningASecond) {
  RaidenId rid{"disco_job_adopt", "0", "kv_cache", 0};

  // Nothing starts a backend's server implicitly: the only way a backend can
  // already host one by the time the store wires up is a caller starting it
  // explicitly beforehand, as here.
  // The bootstrap controller only needs to be non-null for StartServer to
  // succeed; the store below wires its own controller in afterward.
  auto bootstrap_controller = MakeRecoveryController(rid, /*num_blocks=*/16);
  auto backend = std::make_shared<HostOffloadBackend>(
      /*capacity=*/16, std::nullopt, rid, bootstrap_controller.get());
  ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
  ASSERT_NE(backend->store_server(), nullptr);

  KVCacheStore store(
      std::vector<std::shared_ptr<KVCacheStoreBackend>>{backend}, rid,
      /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*raiden_orchestrator_address=*/"", /*store_server_ip=*/"127.0.0.1",
      /*raiden_controller_port=*/0, registry_address_);

  ASSERT_NE(store.store_server(), nullptr);
  ASSERT_FALSE(store.backends().empty());
  EXPECT_EQ(store.store_server(), store.backends()[0]->store_server());
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

// Constructing a store with no tier-0 backend (or a null tier-0 pointer) is
// now a construction-rule violation. Catching it at construction guarantees
// that a store configured with a registry is always registered by preventing
// the un-registered state from being constructed.
TEST_F(StoreDiscoveryTest, NoBackendIsAConstructionRuleViolation) {
  RaidenId rid{"disco_job_nobackend", "0", "kv_cache", 0};
  EXPECT_DEATH(
      {
        KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{},
                           rid,
                           /*num_shards=*/1, /*shard_size_bytes=*/512,
                           /*raiden_orchestrator_address=*/"",
                           /*store_server_ip=*/"127.0.0.1",
                           /*raiden_controller_port=*/0, registry_address_);
      },
      "requires at least one backend");

  EXPECT_DEATH(
      {
        KVCacheStore store(
            std::vector<std::shared_ptr<KVCacheStoreBackend>>{nullptr}, rid,
            /*num_shards=*/1, /*shard_size_bytes=*/512,
            /*raiden_orchestrator_address=*/"",
            /*store_server_ip=*/"127.0.0.1",
            /*raiden_controller_port=*/0, registry_address_);
      },
      "tier-0 backend must not be null");

  EXPECT_FALSE(KVCacheStore::Create(std::vector<BackendConfig>{},
                                    /*capacity=*/16, registry_address_, rid,
                                    /*num_shards=*/1, /*shard_size_bytes=*/512,
                                    /*raiden_orchestrator_address=*/"",
                                    /*store_server_ip=*/"127.0.0.1",
                                    /*raiden_controller_port=*/0)
                   .ok());
}

// Giving those two constructors a real registry client is a behaviour change,
// not just a repair: their backends now take part in the global tier. Pinned
// here so it stays a decision on the record rather than something a later
// reader "fixes" back.
TEST_F(StoreDiscoveryTest, CapacityConstructedStoreJoinsTheGlobalTier) {
  RaidenId rid{"disco_job_globaltier", "0", "kv_cache", 0};
  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);

  auto* backend = dynamic_cast<HostOffloadBackend*>(store.backend().get());
  ASSERT_NE(backend, nullptr);

  // Insert registers globally (InsertAndLock deliberately does not).
  backend->Insert({"tiered_hash"}, {RaidenBlockID(rid, 3, BlockStatus::HOST)},
                  /*on_host=*/true);
  auto looked_up = client_->Lookup({"tiered_hash"});
  ASSERT_TRUE(looked_up.ok()) << looked_up.status().ToString();
  ASSERT_EQ(looked_up->size(), 1);
  EXPECT_EQ((*looked_up)[0].block_id(), 3);

  // A local miss consults tier 1 and comes back with the owning peer.
  RaidenId peer{"some_peer", "0", "kv_cache", 0};
  ASSERT_TRUE(client_->Register({{"peer_hash", peer, 9}}).ok());
  auto result = backend->Lookup({"peer_hash"});
  ASSERT_TRUE(result.ok()) << result.status().ToString();
  ASSERT_EQ(result->size(), 1);
  EXPECT_EQ((*result)[0].second.raiden_id, peer);
  EXPECT_EQ((*result)[0].second.status, BlockStatus::REMOTE);
}

// A backend's server can be started AFTER the store was constructed -- which
// is too late for the store's adoption pass to have seen it -- and with no
// registry the store never adopts one at all. That server holds the store's
// RaidenController in a pointer it cannot re-seat, so leaving it running past
// the store's destructor gives a live service dereferencing a freed
// controller. Waiting for ~HostOffloadBackend is not enough: backends_ holds
// shared_ptrs, so a caller that keeps its own reference outlives the store,
// which is exactly what this test does.
//
// GetGrpcPort() going to 0 is Shutdown()'s observable effect, so this fails
// deterministically without a sanitizer.
TEST_F(StoreDiscoveryTest, DestructorShutsDownABackendStartedServer) {
  RaidenId rid{"disco_job_h1", "0", "kv_cache", 0};
  std::shared_ptr<KVCacheStoreBackend> backend_ref;
  {
    KVCacheStore store(/*capacity=*/16, /*global_registry_address=*/"", rid,
                       /*num_shards=*/1, /*shard_size_bytes=*/512,
                       /*raiden_orchestrator_address=*/"",
                       /*store_server_ip=*/"127.0.0.1",
                       /*raiden_controller_port=*/0);
    // No registry, so the store owns and adopts nothing.
    ASSERT_EQ(store.store_server(), nullptr);

    backend_ref = store.backend();
    auto* backend = dynamic_cast<HostOffloadBackend*>(backend_ref.get());
    ASSERT_NE(backend, nullptr);
    ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
    ASSERT_NE(backend->store_server(), nullptr);
    ASSERT_GT(backend->store_server()->GetGrpcPort(), 0);
  }

  auto* backend = dynamic_cast<HostOffloadBackend*>(backend_ref.get());
  ASSERT_NE(backend->store_server(), nullptr);
  EXPECT_EQ(backend->store_server()->GetGrpcPort(), 0)
      << "the store's destructor left a backend-hosted server running, still "
         "holding the controller it just destroyed";
}

// Same sweep, but with a registry, so the store DID adopt the backend's
// server. Covers the de-duplication: the adopted server is shut once, not
// twice.
TEST_F(StoreDiscoveryTest, DestructorSweepSkipsTheAdoptedServer) {
  RaidenId rid{"disco_job_h1_adopt", "0", "kv_cache", 0};
  auto bootstrap_controller = MakeRecoveryController(rid, /*num_blocks=*/16);
  auto backend = std::make_shared<HostOffloadBackend>(
      /*capacity=*/16, std::nullopt, rid, bootstrap_controller.get());
  ASSERT_TRUE(backend->StartServer("127.0.0.1").ok());
  ASSERT_GT(backend->store_server()->GetGrpcPort(), 0);

  {
    KVCacheStore store(
        std::vector<std::shared_ptr<KVCacheStoreBackend>>{backend}, rid,
        /*num_shards=*/1, /*shard_size_bytes=*/512,
        /*raiden_orchestrator_address=*/"",
        /*store_server_ip=*/"127.0.0.1",
        /*raiden_controller_port=*/0, registry_address_);
    ASSERT_EQ(store.store_server(), backend->store_server());
  }

  EXPECT_EQ(backend->store_server()->GetGrpcPort(), 0);
}

// store_clients_ caches one client per peer and nothing ever erased it, so a
// peer that restarted on a new port stayed undialable for the life of THIS
// process -- even after the registry had healed. The invalidation call was
// lost when GlobalMemoryPoolingBackend was folded into HostOffloadBackend.
//
// The two failures are deliberately different so the assertion has teeth:
// dialling a closed port is UNAVAILABLE, while reaching a real peer that does
// not hold the hash is NOT_FOUND. Without invalidation the second Load
// redials the dead address and stays UNAVAILABLE.
TEST_F(StoreDiscoveryTest, FailedLoadDropsTheCachedPeerClient) {
  RaidenId rid{"disco_job_invalidate", "0", "kv_cache", 0};
  RaidenId peer_rid{"disco_job_invalidate_peer", "0", "kv_cache", 0};

  KVCacheStore store(/*capacity=*/16, registry_address_, rid, /*num_shards=*/1,
                     /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1",
                     /*raiden_controller_port=*/0);
  auto* backend = dynamic_cast<HostOffloadBackend*>(store.backend().get());
  ASSERT_NE(backend, nullptr);

  // Port 1 is reserved and never listening, so this connect always fails.
  ASSERT_TRUE(client_
                  ->RegisterStore(peer_rid, "127.0.0.1:1",
                                  /*controller_address=*/"")
                  .ok());
  auto first = backend->Load(peer_rid, {"h"}).Await();
  ASSERT_FALSE(first.ok());
  ASSERT_TRUE(absl::IsUnavailable(first)) << first.ToString();

  // The peer comes back on a real port and republishes, replacing its entry.
  KVCacheStore peer(/*capacity=*/16, registry_address_, peer_rid,
                    /*num_shards=*/1, /*shard_size_bytes=*/512,
                    /*raiden_orchestrator_address=*/"",
                    /*store_server_ip=*/"127.0.0.1",
                    /*raiden_controller_port=*/0);

  ASSERT_FALSE(peer.store_server_address().empty());

  auto second = backend->Load(peer_rid, {"h"}).Await();
  EXPECT_TRUE(absl::IsNotFound(second))
      << "expected the restarted peer to answer; got " << second.ToString()
      << " -- the cached client still points at the address it had";
}

// ===========================================================================
// Remote write -- source side.
// ===========================================================================

// A destination that says whatever the test tells it to.
//
// The source's job is to turn each terminal verdict into the right thing for
// its caller, and that is worth testing on its own. Driving a real destination
// into a given verdict means arranging the destination's internal state --
// which tests the destination, and reaches the source's handling only by
// implication. STORED_UNREGISTERED is the clearest case: producing it for real
// needs a registry that fails at one exact moment mid-transfer.
class FakeDestinationService : public proto::KVCacheStoreService::Service {
 public:
  static constexpr uint64_t kOperationId = 4242;

  ::grpc::Status WriteRemote(::grpc::ServerContext* /*context*/,
                             const proto::WriteRemoteRequest* request,
                             proto::WriteRemoteResponse* response) override {
    absl::MutexLock lock(&mutex_);
    ++write_calls_;
    requested_deadline_ms_ = request->deadline_ms();
    response->set_operation_id(kOperationId);
    response->set_exist_state(proto::WRITE_EXIST_STATE_UNSPECIFIED);
    response->set_granted_deadline_ms(request->deadline_ms());
    return ::grpc::Status::OK;
  }

  ::grpc::Status PollWriteRemote(
      ::grpc::ServerContext* /*context*/,
      const proto::PollWriteRemoteRequest* /*request*/,
      proto::PollWriteRemoteResponse* response) override {
    absl::MutexLock lock(&mutex_);
    ++poll_calls_;
    *response = poll_response_;
    return ::grpc::Status::OK;
  }

  void SetPollResponse(proto::PollWriteRemoteResponse response) {
    absl::MutexLock lock(&mutex_);
    poll_response_ = std::move(response);
  }

  int write_calls() const {
    absl::MutexLock lock(&mutex_);
    return write_calls_;
  }
  int64_t requested_deadline_ms() const {
    absl::MutexLock lock(&mutex_);
    return requested_deadline_ms_;
  }

 private:
  mutable absl::Mutex mutex_;
  proto::PollWriteRemoteResponse poll_response_ ABSL_GUARDED_BY(mutex_);
  int write_calls_ ABSL_GUARDED_BY(mutex_) = 0;
  int poll_calls_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t requested_deadline_ms_ ABSL_GUARDED_BY(mutex_) = 0;
};

class RemoteWriteSourceTest : public StoreDiscoveryTest {
 protected:
  static constexpr int kCapacity = 8;

  std::unique_ptr<KVCacheStore> MakeStore(const RaidenId& id,
                                          bool with_registry = true) {
    return std::make_unique<KVCacheStore>(
        /*capacity=*/kCapacity,
        with_registry ? registry_address_ : std::string(), id,
        /*num_shards=*/1, /*shard_size_bytes=*/1024,
        /*raiden_orchestrator_address=*/"",
        /*store_server_ip=*/"127.0.0.1",
        /*raiden_controller_port=*/0);
  }

  // A source with no registered workers has no data plane to be pulled from,
  // and the destination refuses such an offer -- correctly, but it means every
  // case below needs one. The worker is real (registration allocates buffers
  // on it) but nothing here drives a transfer to completion through it.
  void RegisterWorker(KVCacheStore& store) {
    worker_server_ = ::tpu_raiden::controller::CreateTestWorkerServer();
    transfer_mock_ = std::make_unique<
        ::tpu_raiden::controller::ShardAwareMockTransferManager>();
    worker_server_->service->SetTransferManager(
        ::tpu_raiden::KVManagerHolder(transfer_mock_.get()));

    ::tpu_raiden::core::controller::RaidenControllerClient client(
        store.raiden_controller_address());
    auto status =
        client.RegisterWorker("worker_0", worker_server_->server_address,
                              {{worker_server_->server_address, {}}});
    ASSERT_TRUE(status.ok()) << status.message();
  }

  // Puts `hashes` in `store` as host-resident, which is the precondition for
  // offering them.
  static void Populate(KVCacheStore& store, const RaidenId& id,
                       const std::vector<std::string>& hashes) {
    std::vector<RaidenBlockID> slices;
    for (size_t i = 0; i < hashes.size(); ++i) {
      slices.push_back(
          RaidenBlockID(id, static_cast<int>(i), BlockStatus::HOST));
    }
    ASSERT_TRUE(store.InsertAndLock(hashes, slices, /*on_host=*/true));
  }

  // Drives the store's poller until the write leaves the pending set.
  // Returns {done, failed, existing, unregistered}.
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>, std::vector<std::string>>
  AwaitWriteSettled(KVCacheStore& store) {
    for (int i = 0; i < 300; ++i) {
      auto [done, failed, pending, existing, unregistered] =
          store.PollRemoteWriteStatus();
      if (pending.empty() && (!done.empty() || !failed.empty())) {
        return {done, failed, existing, unregistered};
      }
      absl::SleepFor(absl::Milliseconds(10));
    }
    return {{}, {}, {}, {}};
  }

  // Stands the fake up and publishes it under `dst` so the source resolves it
  // through the registry exactly as it would a real peer.
  void StartFakeDestination(const RaidenId& dst) {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                             &port);
    builder.RegisterService(&fake_destination_);
    fake_destination_server_ = builder.BuildAndStart();
    ASSERT_NE(fake_destination_server_, nullptr);
    ASSERT_TRUE(client_
                    ->RegisterStore(dst, "127.0.0.1:" + std::to_string(port),
                                    /*controller_address=*/"")
                    .ok());
  }

  void TearDown() override {
    if (fake_destination_server_) fake_destination_server_->Shutdown();
    StoreDiscoveryTest::TearDown();
  }

  FakeDestinationService fake_destination_;
  std::unique_ptr<grpc::Server> fake_destination_server_;
  std::unique_ptr<::tpu_raiden::controller::TestWorkerServer> worker_server_;
  std::unique_ptr<::tpu_raiden::controller::ShardAwareMockTransferManager>
      transfer_mock_;
};

// The registry precondition, reported by the one component that knows: the
// backend, which is what resolves peers.
TEST_F(RemoteWriteSourceTest, RefusesWithoutAGlobalRegistry) {
  RaidenId src{"rw_src_noreg", "0", "kv", 0};
  auto store = MakeStore(src, /*with_registry=*/false);
  Populate(*store, src, {"a"});

  auto status = store->WriteRemote({"a"}, RaidenId{"rw_dst", "0", "kv", 0});
  EXPECT_TRUE(absl::IsFailedPrecondition(status)) << status.ToString();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("registry"));
}

// A missing registration means gone, not still-starting: every store that
// exists with a registry is registered before its constructor returns. So
// this fails fast rather than retrying.
TEST_F(RemoteWriteSourceTest, AnUnresolvableDestinationFailsImmediately) {
  RaidenId src{"rw_src_unresolvable", "0", "kv", 0};
  auto store = MakeStore(src);
  Populate(*store, src, {"a"});

  auto status = store->WriteRemote({"a"}, RaidenId{"nobody", "0", "kv", 0});
  EXPECT_TRUE(absl::IsNotFound(status)) << status.ToString();

  auto [done, failed, pending, existing, unregistered] =
      store->PollRemoteWriteStatus();
  EXPECT_TRUE(pending.empty()) << "a rejected offer must not stay active";
}

TEST_F(RemoteWriteSourceTest, RefusesToOfferBlocksItDoesNotHold) {
  RaidenId src{"rw_src_missing", "0", "kv", 0};
  auto store = MakeStore(src);

  auto status = store->WriteRemote({"absent"}, RaidenId{"d", "0", "kv", 0});
  EXPECT_TRUE(absl::IsNotFound(status)) << status.ToString();
}

TEST_F(RemoteWriteSourceTest, RefusesToOfferToItself) {
  RaidenId src{"rw_src_self", "0", "kv", 0};
  auto store = MakeStore(src);
  Populate(*store, src, {"a"});

  EXPECT_TRUE(absl::IsInvalidArgument(store->WriteRemote({"a"}, src)));
}

// The destination already had everything. A SUCCESS that moves no bytes and
// never creates an operation to poll.
TEST_F(RemoteWriteSourceTest, AllExistSettlesDoneWithoutATransfer) {
  RaidenId src{"rw_src_allexist", "0", "kv", 0};
  RaidenId dst{"rw_dst_allexist", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a", "b"});
  ASSERT_TRUE(dst_store->backend()->InsertAllOrNothing(
      {"a", "b"}, {RaidenBlockID(dst, 5, BlockStatus::HOST),
                   RaidenBlockID(dst, 6, BlockStatus::HOST)}));

  ASSERT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok());

  auto [done, failed, pending, existing, unregistered] =
      src_store->PollRemoteWriteStatus();
  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(failed.empty());
  EXPECT_TRUE(pending.empty());
  EXPECT_TRUE(existing.empty());
}

// The destination held a strict subset. This is a FAILURE, the list reaches
// the caller, and the store does NOT reissue with the difference -- that is
// the caller's decision, and it needs the list to make it.
TEST_F(RemoteWriteSourceTest, PartialExistIsReportedAndNotRetried) {
  RaidenId src{"rw_src_partial", "0", "kv", 0};
  RaidenId dst{"rw_dst_partial", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a", "b"});
  ASSERT_TRUE(dst_store->backend()->InsertAllOrNothing(
      {"a"}, {RaidenBlockID(dst, 5, BlockStatus::HOST)}));

  ASSERT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok());

  auto [done, failed, pending, existing, unregistered] =
      src_store->PollRemoteWriteStatus();
  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(pending.empty());
  EXPECT_THAT(existing, ::testing::UnorderedElementsAre("a"));

  // No hidden second offer: the destination still holds only what it had.
  EXPECT_EQ(dst_store->backend()->GetSize(), 1);
}

// The full source loop: offer, get an operation id back, poll it to a verdict
// through the store's own poller, and release the internal pin. The transfer
// itself fails here (this destination has no workers registered), which is
// the outcome under test -- what matters is that the source reaches a
// terminal answer and lets go.
TEST_F(RemoteWriteSourceTest, AnAcceptedOfferIsPolledToATerminalVerdict) {
  RaidenId src{"rw_src_accept", "0", "kv", 0};
  RaidenId dst{"rw_dst_accept", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a", "b"});

  ASSERT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok());

  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);
  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a", "b"));

  // The internal pin is gone and the hashes are no longer marked as writing,
  // so the same blocks can be offered again.
  EXPECT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok())
      << "the first operation never released its claim on these hashes";
  AwaitWriteSettled(*src_store);
}

// Two offers of the same hash at once would send two sets of authoritative
// block ids for one pin.
TEST_F(RemoteWriteSourceTest, RefusesASecondConcurrentOfferOfTheSameHash) {
  RaidenId src{"rw_src_concurrent", "0", "kv", 0};
  RaidenId dst{"rw_dst_concurrent", "0", "kv", 0};
  auto src_store = MakeStore(src);
  auto dst_store = MakeStore(dst);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a"});

  ASSERT_TRUE(src_store->WriteRemote({"a"}, dst).ok());
  auto second = src_store->WriteRemote({"a"}, dst);
  EXPECT_TRUE(absl::IsFailedPrecondition(second)) << second.ToString();
  AwaitWriteSettled(*src_store);
}

// The verdict this whole state exists for: the peer HAS the bytes but could
// not publish them.
//
// The source must not report it as done -- done is what tells a caller it may
// drop its own copy, and dropping it here moves the block from findable-here
// to findable-nowhere. It must report failed AND hand over the list, so a
// caller that only needed the peer to hold the bytes can decide otherwise.
TEST_F(RemoteWriteSourceTest, StoredUnregisteredIsFailedAndNamesTheBlocks) {
  RaidenId src{"rw_src_unreg", "0", "kv", 0};
  RaidenId dst{"rw_dst_unreg", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::STORED_UNREGISTERED);
  verdict.add_unregistered_hashes("a");
  verdict.add_unregistered_hashes("b");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_TRUE(done.empty())
      << "reporting this as done would let a caller free its only findable "
         "copy";
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_THAT(unregistered, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(existing.empty());

  // The internal pin is released either way, so the caller can act on the
  // list -- including by offering the same blocks somewhere else.
  EXPECT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok());
  AwaitWriteSettled(*src_store);
}

// The ordinary success, driven through the same seam so the two are directly
// comparable: same offer, different verdict, opposite bucket.
TEST_F(RemoteWriteSourceTest, CommittedIsReportedAsDone) {
  RaidenId src{"rw_src_committed", "0", "kv", 0};
  RaidenId dst{"rw_dst_committed", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a", "b"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  verdict.add_committed_hashes("b");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->WriteRemote({"a", "b"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_THAT(done, ::testing::UnorderedElementsAre("a", "b"));
  EXPECT_TRUE(failed.empty());
  EXPECT_TRUE(unregistered.empty());
  EXPECT_TRUE(existing.empty());
}

// Aged out, or the destination restarted. Indistinguishable from "never
// happened", so it must land as a plain failure with no annotations -- in
// particular it must NOT be mistaken for the stored-but-unpublished case,
// where the peer really does hold the data.
TEST_F(RemoteWriteSourceTest, UnknownIsReportedAsAPlainFailure) {
  RaidenId src{"rw_src_unknown", "0", "kv", 0};
  RaidenId dst{"rw_dst_unknown", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::UNKNOWN);
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->WriteRemote({"a"}, dst).ok());
  auto [done, failed, existing, unregistered] = AwaitWriteSettled(*src_store);

  EXPECT_TRUE(done.empty());
  EXPECT_THAT(failed, ::testing::UnorderedElementsAre("a"));
  EXPECT_TRUE(unregistered.empty());
}

// The deadline the source asks for is HOLD minus the margin, so the
// destination's grant can never outlive the source's pin. Checked here because
// the fake records what actually arrived on the wire.
TEST_F(RemoteWriteSourceTest, TheOfferAsksForLessThanTheSourceWillHold) {
  RaidenId src{"rw_src_deadline", "0", "kv", 0};
  RaidenId dst{"rw_dst_deadline", "0", "kv", 0};
  auto src_store = MakeStore(src);
  Populate(*src_store, src, {"a"});
  StartFakeDestination(dst);

  proto::PollWriteRemoteResponse verdict;
  verdict.set_state(proto::PollWriteRemoteResponse::COMMITTED);
  verdict.add_committed_hashes("a");
  fake_destination_.SetPollResponse(verdict);

  ASSERT_TRUE(src_store->WriteRemote({"a"}, dst).ok());
  AwaitWriteSettled(*src_store);

  EXPECT_EQ(fake_destination_.write_calls(), 1);
  // Default HOLD is 30s and the margin 5s.
  EXPECT_EQ(fake_destination_.requested_deadline_ms(),
            absl::ToInt64Milliseconds(absl::Seconds(25)));
}

// A destination that vanished after registering. The source must get a prompt
// error rather than waiting, and must drop the cached client so a restarted
// peer is reachable.
TEST_F(RemoteWriteSourceTest, AStaleButRegisteredDestinationFailsPromptly) {
  RaidenId src{"rw_src_stale", "0", "kv", 0};
  RaidenId dst{"rw_dst_stale", "0", "kv", 0};
  auto src_store = MakeStore(src);
  RegisterWorker(*src_store);
  Populate(*src_store, src, {"a"});

  {
    auto dst_store = MakeStore(dst);
    ASSERT_FALSE(dst_store->store_server_address().empty());
  }
  // The registration outlives the store: store entries never expire, and
  // nothing unpublishes on the way out except the store itself, which has
  // already gone.
  auto status = src_store->WriteRemote({"a"}, dst);
  EXPECT_FALSE(status.ok())
      << "offering to a dead peer should fail rather than hang";
}

// ---------------------------------------------------------------------------
// Construction rules (kv_cache_store_construction_rules.md): Create() rejects
// a missing/wildcard store_server_ip and a controller-less configuration with
// InvalidArgument; the raw constructors LOG(FATAL) on the same violations.
// ---------------------------------------------------------------------------

BackendConfig MakeHostBackendConfig() {
  BackendConfig config;
  config.type = "HostOffloadBackend";
  config.capacity = 4;
  return config;
}

TEST(KVCacheStoreConstructionRulesTest, CreateRejectsEmptyStoreServerIp) {
  auto store_or = KVCacheStore::Create(
      MakeHostBackendConfig(), /*capacity=*/4,
      /*global_registry_address=*/"", RaidenId{}, /*num_shards=*/1,
      /*shard_size_bytes=*/512, /*raiden_orchestrator_address=*/"",
      /*store_server_ip=*/"");
  EXPECT_TRUE(absl::IsInvalidArgument(store_or.status())) << store_or.status();
}

TEST(KVCacheStoreConstructionRulesTest, CreateRejectsWildcardStoreServerIp) {
  for (const char* wildcard : {"[::]", "0.0.0.0", "::"}) {
    auto store_or = KVCacheStore::Create(
        MakeHostBackendConfig(), /*capacity=*/4,
        /*global_registry_address=*/"", RaidenId{}, /*num_shards=*/1,
        /*shard_size_bytes=*/512, /*raiden_orchestrator_address=*/"",
        /*store_server_ip=*/wildcard);
    EXPECT_TRUE(absl::IsInvalidArgument(store_or.status()))
        << "wildcard \"" << wildcard << "\": " << store_or.status();
  }
}

TEST(KVCacheStoreConstructionRulesTest, CreateRejectsZeroShards) {
  auto store_or = KVCacheStore::Create(
      MakeHostBackendConfig(), /*capacity=*/4,
      /*global_registry_address=*/"", RaidenId{}, /*num_shards=*/0,
      /*shard_size_bytes=*/512, /*raiden_orchestrator_address=*/"",
      /*store_server_ip=*/"127.0.0.1");
  EXPECT_TRUE(absl::IsInvalidArgument(store_or.status())) << store_or.status();
}

TEST(KVCacheStoreConstructionRulesDeathTest, CapacityCtorDiesOnEmptyIp) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      KVCacheStore store(/*capacity=*/4, /*global_registry_address=*/"",
                         RaidenId{}, /*num_shards=*/1,
                         /*shard_size_bytes=*/512,
                         /*raiden_orchestrator_address=*/"",
                         /*store_server_ip=*/""),
      "construction validation failed");
}

TEST(KVCacheStoreConstructionRulesDeathTest, CapacityCtorDiesOnZeroShards) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      KVCacheStore store(/*capacity=*/4, /*global_registry_address=*/"",
                         RaidenId{}, /*num_shards=*/0,
                         /*shard_size_bytes=*/512,
                         /*raiden_orchestrator_address=*/"",
                         /*store_server_ip=*/"127.0.0.1"),
      "construction validation failed");
}

// Regression test: Create() must return a Status on a RegisterStore
// failure, not FATAL. This is the one failure Create() cannot pre-validate --
// it only surfaces once the store actually tries to register itself, so it is
// the whole reason Create() exists as a distinct, recoverable-error path from
// the raw constructors. This caught a real bug once: the constructor Create()
// calls internally already wired the controller (and FATALed on failure)
// before Create() got a chance to do its own recoverable wiring, making
// Create()'s Status-return dead code for every RegisterStore failure.
TEST(KVCacheStoreConstructionRulesTest, CreateFailsWhenRegistryPublishFails) {
  auto service = std::make_unique<global_registry::GlobalRegistryServiceImpl>();
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(service.get());
  auto server = builder.BuildAndStart();
  std::string server_address = "localhost:" + std::to_string(port);

  // A reachable registry that genuinely rejects RegisterStore (empty
  // job_name) -- registered, valid construction args, but a real runtime
  // publish failure.
  auto store_or = KVCacheStore::Create(
      MakeHostBackendConfig(), /*capacity=*/4, server_address, RaidenId{},
      /*num_shards=*/1, /*shard_size_bytes=*/512,
      /*raiden_orchestrator_address=*/"", /*store_server_ip=*/"127.0.0.1");
  EXPECT_FALSE(store_or.ok()) << "expected RegisterStore's rejection to "
                                 "surface as a Create() failure";

  server->Shutdown();
}

class ErrorLookupBackend : public KVCacheStoreBackend {
 public:
  std::string name() const override { return "ErrorLookupBackend"; }
  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override {
    return absl::InternalError("Backend lookup failed");
  }
  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids = {},
                     absl::Span<const RaidenBlockID> slices = {}) override {
    return {};
  }
  std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockID> slices, bool on_host) override {
    return {false, {}};
  }
  bool InsertAndLock(absl::Span<const std::string> block_hashes,
                     absl::Span<const RaidenBlockID> slices,
                     bool on_host) override {
    return false;
  }
  size_t ReleaseAndDelete(absl::Span<const std::string> block_hashes) override {
    return 0;
  }
  void Delete(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockID> slices) override {}
  bool Pin(absl::Span<const std::string> block_hashes) override {
    return false;
  }
  void Release(absl::Span<const std::string> block_hashes) override {}
  int GetPinCount(const std::string& hash) const override { return 0; }
  size_t GetCapacity() const override { return 0; }
  size_t GetSize() const override { return 0; }
  size_t GetAvailableSpace() const override { return 0; }
};

TEST(KVCacheStoreTest, LookupAndPinWorkflow) {
  auto b = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

  RaidenId id{"job", "0", "cache", 0};
  std::vector<std::string> hashes = {"h1", "h2"};
  std::vector<RaidenBlockID> slices = {RaidenBlockID(id, 1, BlockStatus::HOST),
                                       RaidenBlockID(id, 2, BlockStatus::HOST)};
  store.Insert(hashes, slices, /*on_host=*/true);

  auto res = store.Lookup(hashes, LookupOptions{.pin_found = true});
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(res->size(), 2);
  EXPECT_EQ(store.GetPinCount("h1"), 1);
  EXPECT_EQ(store.GetPinCount("h2"), 1);

  // Attempting to insert h3 when all blocks are pinned should not evict h1 or
  // h2
  std::vector<std::string> new_hash = {"h3"};
  std::vector<RaidenBlockID> new_slice = {
      RaidenBlockID(id, 3, BlockStatus::HOST)};
  auto insert_res = store.Insert(new_hash, new_slice, /*on_host=*/true);
  EXPECT_TRUE(insert_res.second.empty());  // No evictions occurred

  // Release pins
  store.Release(hashes);
  EXPECT_EQ(store.GetPinCount("h1"), 0);
  EXPECT_EQ(store.GetPinCount("h2"), 0);

  // Now inserting h3 evicts unpinned h2 (tail of sequence)
  insert_res = store.Insert(new_hash, new_slice, /*on_host=*/true);
  ASSERT_EQ(insert_res.second.size(), 1);
  EXPECT_EQ(insert_res.second[0].first, "h2");
}

TEST(KVCacheStoreTest, LookupAndPinErrorRollback) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1"}, {RaidenBlockID(id, 1, BlockStatus::HOST)},
             /*on_host=*/true);

  auto b2 = std::make_shared<ErrorLookupBackend>();

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

  auto res = store.Lookup({"h1", "h2"}, LookupOptions{.pin_found = true});
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(b1->GetPinCount("h1"), 0);
}

TEST(KVCacheStoreTest, LookupAndPinCapacityTruncation) {
  auto b1 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);
  auto b2 = std::make_shared<HostOffloadBackend>(/*capacity=*/2);

  RaidenId id{"job", "0", "cache", 0};
  b1->Insert({"h1", "h2"},
             {RaidenBlockID(id, 1, BlockStatus::HOST),
              RaidenBlockID(id, 2, BlockStatus::HOST)},
             /*on_host=*/true);
  b2->Insert({"h3"}, {RaidenBlockID(id, 3, BlockStatus::HOST)},
             /*on_host=*/true);

  KVCacheStore store(std::vector<std::shared_ptr<KVCacheStoreBackend>>{b1, b2},
                     RaidenId{}, /*num_shards=*/1, /*shard_size_bytes=*/512,
                     /*raiden_orchestrator_address=*/"",
                     /*store_server_ip=*/"127.0.0.1");

  EXPECT_EQ(store.capacity(), 2);

  auto res = store.Lookup({"h1", "h2", "h3"}, LookupOptions{.pin_found = true});
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res->size(), 2);
  EXPECT_EQ((*res)[0].first, "h1");
  EXPECT_EQ((*res)[1].first, "h2");

  EXPECT_EQ(b1->GetPinCount("h1"), 1);
  EXPECT_EQ(b1->GetPinCount("h2"), 1);
  EXPECT_EQ(b2->GetPinCount("h3"), 0);
}

}  // namespace
}  // namespace kv_cache
}  // namespace tpu_raiden
